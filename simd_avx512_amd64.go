//go:build goexperiment.simd && amd64

package utlpfor

import (
	"simd/archsimd"
	"unsafe"
)

// unpackLanesUTLAVX512 unpacks UTL payload using AVX-512 (Uint32x16).
// Implements FastLanes Algorithm 2 with 512-bit vectors (1 load per super-word).
// TODO-PERF: per-bitwidth specialization may enable inlining (current cost exceeds budget).
func unpackLanesUTLAVX512(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	mask := archsimd.BroadcastUint32x16(uint32((1 << bitWidth) - 1))
	if bitWidth == 32 {
		mask = archsimd.BroadcastUint32x16(0xFFFFFFFF)
	}

	bitOffset := 0
	for v := 0; v < utlValuesPerLane; v++ {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		vec := archsimd.LoadUint32x16(
			(*[16]uint32)(unsafe.Pointer(&payload[base])))
		result := vec.ShiftAllRight(shift).And(mask)

		if int(shift)+bitWidth > 32 {
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			nextVec := archsimd.LoadUint32x16(
				(*[16]uint32)(unsafe.Pointer(&payload[nextBase])))
			result = result.Or(
				nextVec.ShiftAllLeft(uint64(32) - shift).And(mask))
		}

		outBase := v * utlLaneCount
		result.StoreSlice(dst[outBase : outBase+16])
		bitOffset += bitWidth
	}
}

// packLanesUTLAVX512 packs values into UTL payload using AVX-512 (Uint32x16).
// Implements FastLanes Algorithm 1 with 512-bit vectors (1 load/store per super-word).
// Uses a load-OR-store pattern to avoid persistent accumulator register spilling.
// TODO-PERF: per-bitwidth specialization may enable inlining (current cost exceeds budget).
func packLanesUTLAVX512(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	if bitWidth == 32 {
		for v := 0; v < utlValuesPerLane; v++ {
			inBase := v * utlLaneCount
			base := v * utlSuperWordBytes
			archsimd.LoadUint32x16Slice(values[inBase : inBase+16]).Store(
				(*[16]uint32)(unsafe.Pointer(&dst[base])))
		}
		return
	}

	mask := archsimd.BroadcastUint32x16(uint32((1 << bitWidth) - 1))

	clear(dst)

	bitOffset := 0
	for v := 0; v < utlValuesPerLane; v++ {
		inBase := v * utlLaneCount
		val := archsimd.LoadUint32x16Slice(values[inBase : inBase+16]).And(mask)

		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		cur := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Pointer(&dst[base])))
		cur.Or(val.ShiftAllLeft(shift)).Store((*[16]uint32)(unsafe.Pointer(&dst[base])))

		if int(shift)+bitWidth > 32 {
			rightShift := uint64(32) - shift
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			next := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Pointer(&dst[nextBase])))
			next.Or(val.ShiftAllRight(rightShift)).Store((*[16]uint32)(unsafe.Pointer(&dst[nextBase])))
		}

		bitOffset += bitWidth
	}
}

// packUint32AVX512 is the full AVX-512 packing pipeline.
// Uses AVX-512 for bit-packing; AVX2 for delta/zigzag; scalar for exceptions.
func packUint32AVX512(flag byte, dst []byte, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	headerFlags := headerTypeUint32Flag
	workValues := values

	if flag&Delta != 0 {
		var needZZ bool
		if len(values) == blockSize {
			needZZ = deltaEncodePerLaneAVX2(values, values)
		} else {
			needZZ = deltaEncodePerLaneScalar(values, values)
		}
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
		packLanesUTLAVX512(dst[headerBytes:headerBytes+payloadBytes], padded[:], bitWidth)
		archsimd.ClearAVXUpperBits()
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

	packLanesUTLAVX512(dst[pOff:pOff+payloadBytes], padded[:], bitWidth)
	archsimd.ClearAVXUpperBits()
	writeExceptionsDirect(dst[pOff+payloadBytes:], positions[:], bitmap[:], excCount, svbData)

	return dst[:totalLen], nil
}

// unpackUint32AVX512 is the full AVX-512 unpacking pipeline.
// Uses AVX-512 for bit-unpacking; AVX2 for delta/zigzag; scalar for exceptions.
func unpackUint32AVX512(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
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
	unpackLanesUTLAVX512(dst, payload, blockSize, bitWidth)
	archsimd.ClearAVXUpperBits()

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
		var overflowPos int
		if count == blockSize {
			overflowPos = deltaDecodePerLaneWithOverflowAVX2(dst, dst, hasZigZag)
			archsimd.ClearAVXUpperBits()
		} else {
			overflowPos = deltaDecodePerLaneWithOverflowScalar(dst, dst, hasZigZag)
		}
		if overflowPos > 0 {
			return nil, 0, &ErrOverflow{Position: overflowPos}
		}
	}

	return dst, consumed, nil
}

// getUint32AVX512 delegates to scalar for random access.
func getUint32AVX512(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
