//go:build goexperiment.simd && amd64

package utlpfor

import (
	"simd/archsimd"
	"unsafe"
)

// unpackLanesUTLAVX2 unpacks UTL payload using AVX2 (Uint32x8).
// Implements FastLanes Algorithm 2 with 256-bit vectors (2 loads per super-word).
func unpackLanesUTLAVX2(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	mask := archsimd.BroadcastUint32x8(uint32((1 << bitWidth) - 1))
	if bitWidth == 32 {
		mask = archsimd.BroadcastUint32x8(0xFFFFFFFF)
	}

	bitOffset := 0
	for v := 0; v < utlValuesPerLane; v++ {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		lo := archsimd.LoadUint32x8(
			(*[8]uint32)(unsafe.Pointer(&payload[base])))
		hi := archsimd.LoadUint32x8(
			(*[8]uint32)(unsafe.Pointer(&payload[base+32])))

		rLo := lo.ShiftAllRight(shift).And(mask)
		rHi := hi.ShiftAllRight(shift).And(mask)

		if int(shift)+bitWidth > 32 {
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			nextLo := archsimd.LoadUint32x8(
				(*[8]uint32)(unsafe.Pointer(&payload[nextBase])))
			nextHi := archsimd.LoadUint32x8(
				(*[8]uint32)(unsafe.Pointer(&payload[nextBase+32])))
			leftShift := uint64(32) - shift
			rLo = rLo.Or(nextLo.ShiftAllLeft(leftShift).And(mask))
			rHi = rHi.Or(nextHi.ShiftAllLeft(leftShift).And(mask))
		}

		outBase := v * utlLaneCount
		rLo.StoreSlice(dst[outBase : outBase+8])
		rHi.StoreSlice(dst[outBase+8 : outBase+16])

		bitOffset += bitWidth
	}
}

// packLanesUTLAVX2 packs values into UTL payload using AVX2 (Uint32x8).
// Implements FastLanes Algorithm 1 with 256-bit vectors (2 stores per super-word).
func packLanesUTLAVX2(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	mask := archsimd.BroadcastUint32x8(uint32((1 << bitWidth) - 1))
	if bitWidth == 32 {
		mask = archsimd.BroadcastUint32x8(0xFFFFFFFF)
	}

	zero := archsimd.BroadcastUint32x8(0)
	accLo := zero
	accHi := zero
	bitOffset := 0

	for v := 0; v < utlValuesPerLane; v++ {
		inBase := v * utlLaneCount
		vLo := archsimd.LoadUint32x8Slice(values[inBase : inBase+8])
		vHi := archsimd.LoadUint32x8Slice(values[inBase+8 : inBase+16])

		vLo = vLo.And(mask)
		vHi = vHi.And(mask)

		shift := uint64(bitOffset % 32)
		accLo = accLo.Or(vLo.ShiftAllLeft(shift))
		accHi = accHi.Or(vHi.ShiftAllLeft(shift))

		if int(shift)+bitWidth >= 32 {
			wordIdx := bitOffset / 32
			base := wordIdx * utlSuperWordBytes
			accLo.Store((*[8]uint32)(unsafe.Pointer(&dst[base])))
			accHi.Store((*[8]uint32)(unsafe.Pointer(&dst[base+32])))

			overflow := int(shift) + bitWidth - 32
			if overflow > 0 {
				rightShift := uint64(bitWidth - overflow)
				accLo = vLo.ShiftAllRight(rightShift)
				accHi = vHi.ShiftAllRight(rightShift)
			} else {
				accLo = zero
				accHi = zero
			}
		}
		bitOffset += bitWidth
	}

	if bitOffset%32 != 0 {
		wordIdx := bitOffset / 32
		base := wordIdx * utlSuperWordBytes
		accLo.Store((*[8]uint32)(unsafe.Pointer(&dst[base])))
		accHi.Store((*[8]uint32)(unsafe.Pointer(&dst[base+32])))
	}
}

// packUint32AVX2 is the full AVX2 packing pipeline.
// Uses AVX2 for bit-packing, scalar for delta/zigzag/exceptions.
func packUint32AVX2(flag byte, dst []byte, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	headerFlags := headerTypeUint32Flag
	workValues := values

	if flag&Delta != 0 {
		needZZ := deltaEncodePerLaneScalar(values, values)
		if needZZ {
			headerFlags |= headerZigZagFlag
		}
		headerFlags |= headerDeltaFlag
		workValues = values
	}

	bitWidth, excCount := selectBitWidth(workValues)
	payloadBytes := utlPayloadBytesLUT[bitWidth]
	hasExceptions := excCount > 0

	// TODO-PERF: Try to avoid the allocation and copy.
	var padded [blockSize]uint32
	copy(padded[:], workValues)

	if !hasExceptions {
		totalLen := headerBytes + payloadBytes
		dst = ensureLen(dst, totalLen)
		bo.PutUint32(dst, encodeHeader(len(values), bitWidth, 0, headerFlags))
		packLanesUTLAVX2(dst[headerBytes:headerBytes+payloadBytes], padded[:], bitWidth)
		return dst[:totalLen], nil
	}

	// TODO-PERF: Try to avoid the allocation and copy.
	var positions [blockSize]byte
	var bitmap [16]byte
	var highBits [blockSize]uint32
	collectExceptionsDirect(workValues, bitWidth, positions[:], bitmap[:], highBits[:])

	svbData := encodeExceptionHighBits(highBits[:excCount])
	svbLen := len(svbData)

	excIdxSize := excCount
	if excCount > excBitmapThreshold {
		excIdxSize = 16
	}

	pOff := payloadOffset(false, true)
	totalLen := pOff + payloadBytes + excIdxSize + svbLen

	dst = ensureLen(dst, totalLen)
	bo.PutUint32(dst, encodeHeader(len(values), bitWidth, excCount, headerFlags))
	bo.PutUint16(dst[headerBytes:], uint16(svbLen))

	packLanesUTLAVX2(dst[pOff:pOff+payloadBytes], padded[:], bitWidth)
	writeExceptionsDirect(dst[pOff+payloadBytes:], positions[:], bitmap[:], excCount, svbData)

	return dst[:totalLen], nil
}

// unpackUint32AVX2 is the full AVX2 unpacking pipeline.
// Uses AVX2 for bit-unpacking, scalar for delta/zigzag/exceptions.
func unpackUint32AVX2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	if len(buf) < headerBytes {
		return nil, 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, intType, excCount, hasExceptions, hasDelta, hasZigZag, hasFOR := decodeHeader(header)

	if err := validateIntType(intType); err != nil {
		return nil, 0, err
	}

	if count == 0 {
		return dst[:0], headerBytes, nil
	}
	if count > blockSize {
		return nil, 0, ErrInvalidBlockLength
	}
	if bitWidth > 32 {
		return nil, 0, ErrInvalidBuffer
	}

	pOff := payloadOffset(hasFOR, hasExceptions)
	payloadBytes := utlPayloadBytesLUT[bitWidth]

	if len(buf) < pOff+payloadBytes {
		return nil, 0, ErrInvalidBuffer
	}

	if cap(dst) < blockSize {
		dst = make([]uint32, blockSize)
	}
	dst = dst[:blockSize]

	payload := buf[pOff : pOff+payloadBytes]
	unpackLanesUTLAVX2(dst, payload, blockSize, bitWidth)

	dst = dst[:count]

	consumed := pOff + payloadBytes

	if hasExceptions {
		excStart := pOff + payloadBytes
		var err error
		consumed, err = applyExceptions(dst, buf, excStart, count, bitWidth, excCount, hasFOR, scratch)
		if err != nil {
			return nil, 0, err
		}
	}

	if hasDelta {
		overflowPos := deltaDecodePerLaneWithOverflowScalar(dst, dst, hasZigZag)
		if overflowPos > 0 {
			return nil, 0, &ErrOverflow{Position: overflowPos}
		}
	}

	return dst, consumed, nil
}

// getUint32AVX2 delegates to scalar for random access.
func getUint32AVX2(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
