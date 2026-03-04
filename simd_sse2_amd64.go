//go:build goexperiment.simd && amd64

package utlpfor

import (
	"simd/archsimd"
	"unsafe"
)

// unpackLanesUTLSSE2 unpacks UTL payload using SSE2 (Uint32x4).
// Implements FastLanes Algorithm 2 with 128-bit vectors (4 loads per super-word).
func unpackLanesUTLSSE2(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	mask := archsimd.BroadcastUint32x4(uint32((1 << bitWidth) - 1))
	if bitWidth == 32 {
		mask = archsimd.BroadcastUint32x4(0xFFFFFFFF)
	}

	bitOffset := 0
	for v := 0; v < utlValuesPerLane; v++ {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		// TODO-PERF: unroll these 4 load groups
		for group := 0; group < 4; group++ {
			off := base + group*16
			vec := archsimd.LoadUint32x4(
				(*[4]uint32)(unsafe.Pointer(&payload[off])))
			result := vec.ShiftAllRight(shift).And(mask)

			if int(shift)+bitWidth > 32 {
				nextOff := (wordIdx+1)*utlSuperWordBytes + group*16
				nextVec := archsimd.LoadUint32x4(
					(*[4]uint32)(unsafe.Pointer(&payload[nextOff])))
				result = result.Or(
					nextVec.ShiftAllLeft(uint64(32) - shift).And(mask))
			}

			outBase := v*utlLaneCount + group*4
			result.StoreSlice(dst[outBase : outBase+4])
		}
		bitOffset += bitWidth
	}
}

// packLanesUTLSSE2 packs values into UTL payload using SSE2 (Uint32x4).
// Implements FastLanes Algorithm 1 with 128-bit vectors (4 stores per super-word).
func packLanesUTLSSE2(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	mask := archsimd.BroadcastUint32x4(uint32((1 << bitWidth) - 1))
	if bitWidth == 32 {
		mask = archsimd.BroadcastUint32x4(0xFFFFFFFF)
	}

	var acc [4]archsimd.Uint32x4
	zero := archsimd.BroadcastUint32x4(0)
	for i := range acc {
		acc[i] = zero
	}
	bitOffset := 0

	for v := 0; v < utlValuesPerLane; v++ {
		shift := uint64(bitOffset % 32)

		for group := 0; group < 4; group++ {
			inBase := v*utlLaneCount + group*4
			val := archsimd.LoadUint32x4Slice(values[inBase : inBase+4])
			val = val.And(mask)
			acc[group] = acc[group].Or(val.ShiftAllLeft(shift))
		}

		if int(shift)+bitWidth >= 32 {
			wordIdx := bitOffset / 32
			base := wordIdx * utlSuperWordBytes
			for group := 0; group < 4; group++ {
				acc[group].Store(
					(*[4]uint32)(unsafe.Pointer(&dst[base+group*16])))
			}

			overflow := int(shift) + bitWidth - 32
			if overflow > 0 {
				rightShift := uint64(bitWidth - overflow)
				for group := 0; group < 4; group++ {
					inBase := v*utlLaneCount + group*4
					val := archsimd.LoadUint32x4Slice(values[inBase : inBase+4])
					val = val.And(mask)
					acc[group] = val.ShiftAllRight(rightShift)
				}
			} else {
				for group := range acc {
					acc[group] = zero
				}
			}
		}
		bitOffset += bitWidth
	}

	if bitOffset%32 != 0 {
		wordIdx := bitOffset / 32
		base := wordIdx * utlSuperWordBytes
		for group := 0; group < 4; group++ {
			acc[group].Store(
				(*[4]uint32)(unsafe.Pointer(&dst[base+group*16])))
		}
	}
}

// packUint32SSE2 is the full SSE2 packing pipeline.
// Uses SSE2 for bit-packing, scalar for delta/zigzag/exceptions.
func packUint32SSE2(flag byte, dst []byte, values []uint32) ([]byte, error) {
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

	var padded [blockSize]uint32
	copy(padded[:], workValues)

	if !hasExceptions {
		totalLen := headerBytes + payloadBytes
		dst = ensureLen(dst, totalLen)
		bo.PutUint32(dst, encodeHeader(len(values), bitWidth, 0, headerFlags))
		packLanesUTLSSE2(dst[headerBytes:headerBytes+payloadBytes], padded[:], bitWidth)
		return dst[:totalLen], nil
	}

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

	packLanesUTLSSE2(dst[pOff:pOff+payloadBytes], padded[:], bitWidth)
	writeExceptionsDirect(dst[pOff+payloadBytes:], positions[:], bitmap[:], excCount, svbData)

	return dst[:totalLen], nil
}

// unpackUint32SSE2 is the full SSE2 unpacking pipeline.
// Uses SSE2 for bit-unpacking, scalar for delta/zigzag/exceptions.
func unpackUint32SSE2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
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
	unpackLanesUTLSSE2(dst, payload, blockSize, bitWidth)

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

// getUint32SSE2 delegates to scalar for random access.
func getUint32SSE2(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
