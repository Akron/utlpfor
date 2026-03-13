//go:build goexperiment.simd && amd64

package utlpfor

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

// unpackLanesUTLAVX512 unpacks UTL payload using AVX-512 (Uint32x16).
// Dispatches to per-bitwidth specialized native archsimd functions.
func unpackLanesUTLAVX512(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	switch bitWidth {
	case 4:
		unpackAVX512BW4(&dst[0], &payload[0])
	case 8:
		unpackAVX512BW8(&dst[0], &payload[0])
	case 12:
		unpackAVX512BW12(&dst[0], &payload[0])
	case 16:
		unpackAVX512BW16(&dst[0], &payload[0])
	case 20:
		unpackAVX512BW20(&dst[0], &payload[0])
	case 24:
		unpackAVX512BW24(&dst[0], &payload[0])
	case 28:
		unpackAVX512BW28(&dst[0], &payload[0])
	case 32:
		unpackAVX512BW32(&dst[0], &payload[0])
	default:
		unpackLanesUTLAVX512Generic(dst, payload, count, bitWidth)
	}
}

// unpackLanesUTLAVX512Generic is the generic loop-based AVX-512 unpacker.
// Used as fallback for non-step bitwidths (should not be reached in practice).
func unpackLanesUTLAVX512Generic(dst []uint32, payload []byte, count, bitWidth int) {
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
// Dispatches to per-bitwidth specialized native archsimd functions.
// Caller must ensure values has at least blockSize elements (zero-padded).
func packLanesUTLAVX512(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	if bitWidth != 32 {
		clear(dst)
	}
	switch bitWidth {
	case 4:
		packAVX512BW4(&dst[0], &values[0])
	case 8:
		packAVX512BW8(&dst[0], &values[0])
	case 12:
		packAVX512BW12(&dst[0], &values[0])
	case 16:
		packAVX512BW16(&dst[0], &values[0])
	case 20:
		packAVX512BW20(&dst[0], &values[0])
	case 24:
		packAVX512BW24(&dst[0], &values[0])
	case 28:
		packAVX512BW28(&dst[0], &values[0])
	case 32:
		packAVX512BW32(&dst[0], &values[0])
	default:
		packLanesUTLAVX512Generic(dst, values, bitWidth)
	}
}

// packLanesUTLAVX512Generic is the generic loop-based AVX-512 packer.
// Used as fallback for non-step bitwidths (should not be reached in practice).
func packLanesUTLAVX512Generic(dst []byte, values []uint32, bitWidth int) {
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
// Uses AVX-512 for bit-packing; AVX2 for delta/zigzag; scalar for exceptions and FOR.
func packUint32AVX512(flag byte, dst []byte, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	headerFlags := headerTypeUint32Flag

	useFOR, baseValue, forW := selectBitWidthWithFORAVX512(values)

	if useFOR {
		forSubtractAVX512(values, values, baseValue)
		headerFlags |= uint32(forW) << forWidthShift
	}

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
	}

	bitWidth, excCount := selectBitWidthAVX512(values)
	payloadBytes := utlPayloadBytesLUT[bitWidth]
	hasExceptions := excCount > 0
	forBaseBytes := forBaseBytesLUT[forW]

	var padded [blockSize]uint32
	packInput := values
	if len(values) < blockSize {
		copy(padded[:], values)
		packInput = padded[:]
	}

	if !hasExceptions {
		pOff := payloadOffset(forBaseBytes, false)
		totalLen := pOff + payloadBytes
		dst = ensureLen(dst, totalLen)
		bo.PutUint32(dst, encodeHeader(len(values), bitWidth, 0, headerFlags))
		if useFOR {
			writeFORBase(dst, baseValue, forW, false)
		}
		packLanesUTLAVX512(dst[pOff:pOff+payloadBytes], packInput, bitWidth)
		archsimd.ClearAVXUpperBits()
		return dst[:totalLen], nil
	}

	var positions [blockSize]byte
	var bitmap [16]byte
	var highBits [blockSize]uint32
	collectExceptionsDirect(values, bitWidth, positions[:], bitmap[:], highBits[:])

	svbData := encodeExceptionHighBits(highBits[:excCount])
	svbLen := len(svbData)

	excIdxSize := excCount
	if excCount > excBitmapThreshold {
		excIdxSize = 16
	}

	pOff := payloadOffset(forBaseBytes, true)
	totalLen := pOff + payloadBytes + excIdxSize + svbLen

	dst = ensureLen(dst, totalLen)
	bo.PutUint32(dst, encodeHeader(len(values), bitWidth, excCount, headerFlags))
	bo.PutUint16(dst[headerBytes:], uint16(svbLen))
	if useFOR {
		writeFORBase(dst, baseValue, forW, true)
	}

	packLanesUTLAVX512(dst[pOff:pOff+payloadBytes], packInput, bitWidth)
	archsimd.ClearAVXUpperBits()
	writeExceptionsDirect(dst[pOff+payloadBytes:], positions[:], bitmap[:], excCount, svbData)

	return dst[:totalLen], nil
}

// unpackUint32AVX512 is the full AVX-512 unpacking pipeline.
// Uses AVX-512 for bit-unpacking; AVX2 for delta/zigzag; scalar for exceptions and FOR.
func unpackUint32AVX512(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	if len(buf) < headerBytes {
		return nil, 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, intType, excCount, forWidth, hasExceptions, hasDelta, hasZigZag := decodeHeader(header)
	hasFOR := forWidth > 0

	if err := validateIntType(intType); err != nil {
		return nil, 0, err
	}

	if uint(count-1) >= blockSize || uint(bitWidth) > 32 {
		if count == 0 {
			return dst[:0], headerBytes, nil
		}
		if count > blockSize {
			return nil, 0, ErrInvalidBlockLength
		}
		return nil, 0, ErrInvalidBuffer
	}

	pOff := payloadOffset(0, hasExceptions)
	var forBase uint32
	if hasFOR {
		forBase = readFORBase(buf, pOff, forWidth)
		pOff += forBaseBytesLUT[forWidth]
	}

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
		consumed, err = applyExceptions(dst, buf, excStart, count, bitWidth, excCount, scratch)
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

	if hasFOR {
		forAddAVX512(dst, count, forBase)
		archsimd.ClearAVXUpperBits()
	}

	return dst, consumed, nil
}

// selectBitWidthAVX512 computes the optimal step bitwidth using SIMD threshold
// comparisons. Instead of computing bits.Len32 per value and incrementing a
// histogram bin (scatter-add), this approach compares all values against each
// step threshold simultaneously. The 8 thresholds are split into two groups
// of 4 to avoid register pressure (5 SIMD registers per pass: 4 thresholds
// + 1 value vector). AVX-512 uses a single pass (9 of 32 ZMM registers).
func selectBitWidthAVX512(values []uint32) (width int, excCount int) {
	width, excCount, _ = chooseBestFromExcCounts(buildExcCountsAVX512(values))
	return
}

// selectBitWidthWithFORSSE2 uses SIMD-accelerated findMinMax and SIMD
// threshold comparisons for the standard histogram. The two-level rejection
// strategy (min==0, forMaxStep >= rawMaxStep/stdWidth) is preserved.
func selectBitWidthWithFORAVX512(values []uint32) (useFOR bool, baseValue uint32, forWidth int) {
	minVal, maxVal := findMinMaxAVX512(values)
	if minVal == 0 {
		return false, 0, forWidthNone
	}
	forMaxStep := roundUpToStep(bits.Len32(maxVal - minVal))
	rawMaxStep := roundUpToStep(bits.Len32(maxVal))
	if forMaxStep >= rawMaxStep {
		return false, 0, forWidthNone
	}

	stdWidth, _, _ := chooseBestFromExcCounts(buildExcCountsAVX512(values))
	if forMaxStep >= stdWidth {
		return false, 0, forWidthNone
	}
	return true, minVal, selectFORWidth(minVal)
}

// buildExcCountsAVX512 computes cumulative exception counts using AVX-512
// threshold comparisons. All 8 thresholds fit in a single pass because
// AVX-512 has 32 ZMM registers (only 9 needed: 8 thresholds + 1 value).
func buildExcCountsAVX512(values []uint32) (exc [9]int) {
	t0 := archsimd.BroadcastUint32x16(0)
	t1 := archsimd.BroadcastUint32x16(0xF)
	t2 := archsimd.BroadcastUint32x16(0xFF)
	t3 := archsimd.BroadcastUint32x16(0xFFF)
	t4 := archsimd.BroadcastUint32x16(0xFFFF)
	t5 := archsimd.BroadcastUint32x16(0xFFFFF)
	t6 := archsimd.BroadcastUint32x16(0xFFFFFF)
	t7 := archsimd.BroadcastUint32x16(0xFFFFFFF)

	i := 0
	for ; i+16 <= len(values); i += 16 {
		v := archsimd.LoadUint32x16Slice(values[i:])
		exc[0] += bits.OnesCount16(v.Greater(t0).ToBits())
		exc[1] += bits.OnesCount16(v.Greater(t1).ToBits())
		exc[2] += bits.OnesCount16(v.Greater(t2).ToBits())
		exc[3] += bits.OnesCount16(v.Greater(t3).ToBits())
		exc[4] += bits.OnesCount16(v.Greater(t4).ToBits())
		exc[5] += bits.OnesCount16(v.Greater(t5).ToBits())
		exc[6] += bits.OnesCount16(v.Greater(t6).ToBits())
		exc[7] += bits.OnesCount16(v.Greater(t7).ToBits())
	}
	for ; i < len(values); i++ {
		v := values[i]
		// Branchless scalar tail: each comparison contributes 0 or 1 to the cumulative exception counters.
		exc[0] += gtCountU32(v, 0)
		exc[1] += gtCountU32(v, 0xF)
		exc[2] += gtCountU32(v, 0xFF)
		exc[3] += gtCountU32(v, 0xFFF)
		exc[4] += gtCountU32(v, 0xFFFF)
		exc[5] += gtCountU32(v, 0xFFFFF)
		exc[6] += gtCountU32(v, 0xFFFFFF)
		exc[7] += gtCountU32(v, 0xFFFFFFF)
	}
	return
}

// findMinMaxAVX512 computes min/max using AVX-512 16-wide operations.
func findMinMaxAVX512(values []uint32) (uint32, uint32) {
	if len(values) < 16 {
		return findMinMaxAVX512(values)
	}
	minVec := archsimd.LoadUint32x16Slice(values[:16])
	maxVec := minVec
	i := 16
	for ; i+16 <= len(values); i += 16 {
		chunk := archsimd.LoadUint32x16Slice(values[i:])
		minVec = minVec.Min(chunk)
		maxVec = maxVec.Max(chunk)
	}
	var minLanes, maxLanes [16]uint32
	minVec.Store(&minLanes)
	maxVec.Store(&maxLanes)
	minResult, maxResult := minLanes[0], maxLanes[0]
	for _, v := range minLanes[1:] {
		if v < minResult {
			minResult = v
		}
	}
	for _, v := range maxLanes[1:] {
		if v > maxResult {
			maxResult = v
		}
	}
	for ; i < len(values); i++ {
		if values[i] < minResult {
			minResult = values[i]
		}
		if values[i] > maxResult {
			maxResult = values[i]
		}
	}
	return minResult, maxResult
}

// forSubtractAVX512 subtracts baseValue from each element using AVX-512.
func forSubtractAVX512(dst, src []uint32, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x16(baseValue)
	i := 0
	for ; i+16 <= len(src); i += 16 {
		v := archsimd.LoadUint32x16Slice(src[i:])
		v = v.Sub(baseVec)
		v.StoreSlice(dst[i:])
	}
	for ; i < len(src); i++ {
		dst[i] = src[i] - baseValue
	}
}

// forAddAVX512 adds baseValue to each of the first count elements using AVX-512.
func forAddAVX512(output []uint32, count int, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x16(baseValue)
	i := 0
	for ; i+16 <= count; i += 16 {
		v := archsimd.LoadUint32x16Slice(output[i:])
		v = v.Add(baseVec)
		v.StoreSlice(output[i:])
	}
	for ; i < count; i++ {
		output[i] += baseValue
	}
}

// getUint32AVX512 delegates to scalar for random access.
func getUint32AVX512(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
