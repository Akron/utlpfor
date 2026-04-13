//go:build goexperiment.simd && amd64

package utlpfor

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

// unpackLanesUTLSSE2 unpacks UTL payload using SSE2 (Uint32x4).
// Dispatches to per-bitwidth specialized functions for step bitwidths.
func unpackLanesUTLSSE2(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	switch bitWidth {
	case 4:
		unpackSSE2BW4(&dst[0], &payload[0])
	case 8:
		unpackSSE2BW8(&dst[0], &payload[0])
	case 12:
		unpackSSE2BW12(&dst[0], &payload[0])
	case 16:
		unpackSSE2BW16(&dst[0], &payload[0])
	case 20:
		unpackSSE2BW20(&dst[0], &payload[0])
	case 24:
		unpackSSE2BW24(&dst[0], &payload[0])
	case 28:
		unpackSSE2BW28(&dst[0], &payload[0])
	case 32:
		unpackSSE2BW32(&dst[0], &payload[0])
	default:
		unpackLanesUTLSSE2Generic(dst, payload, count, bitWidth)
	}
}

// unpackLanesUTLSSE2Generic is the generic loop-based SSE2 unpacker.
// Used as fallback for non-step bitwidths.
func unpackLanesUTLSSE2Generic(dst []uint32, payload []byte, count, bitWidth int) {
	mask := archsimd.BroadcastUint32x4(uint32((1 << bitWidth) - 1))
	if bitWidth == 32 {
		mask = archsimd.BroadcastUint32x4(0xFFFFFFFF)
	}

	bitOffset := 0
	for v := 0; v < utlValuesPerLane; v++ {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

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
// Dispatches to per-bitwidth specialized functions for step bitwidths.
// Caller must ensure values has at least blockSize elements (zero-padded).
func packLanesUTLSSE2(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	if bitWidth != 32 {
		clear(dst)
	}
	switch bitWidth {
	case 4:
		packSSE2BW4(&dst[0], &values[0])
	case 8:
		packSSE2BW8(&dst[0], &values[0])
	case 12:
		packSSE2BW12(&dst[0], &values[0])
	case 16:
		packSSE2BW16(&dst[0], &values[0])
	case 20:
		packSSE2BW20(&dst[0], &values[0])
	case 24:
		packSSE2BW24(&dst[0], &values[0])
	case 28:
		packSSE2BW28(&dst[0], &values[0])
	case 32:
		packSSE2BW32(&dst[0], &values[0])
	default:
		packLanesUTLSSE2Generic(dst, values, bitWidth)
	}
}

// packLanesUTLSSE2Generic is the generic loop-based SSE2 packer.
// Used as fallback for non-step bitwidths.
func packLanesUTLSSE2Generic(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 32 {
		for v := 0; v < utlValuesPerLane; v++ {
			for group := 0; group < 4; group++ {
				inBase := v*utlLaneCount + group*4
				off := v*utlSuperWordBytes + group*16
				archsimd.LoadUint32x4Slice(values[inBase : inBase+4]).Store(
					(*[4]uint32)(unsafe.Pointer(&dst[off])))
			}
		}
		return
	}

	mask := archsimd.BroadcastUint32x4(uint32((1 << bitWidth) - 1))

	clear(dst)

	bitOffset := 0
	for v := 0; v < utlValuesPerLane; v++ {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		for group := 0; group < 4; group++ {
			inBase := v*utlLaneCount + group*4
			val := archsimd.LoadUint32x4Slice(values[inBase : inBase+4]).And(mask)

			off := base + group*16
			cur := archsimd.LoadUint32x4((*[4]uint32)(unsafe.Pointer(&dst[off])))
			cur.Or(val.ShiftAllLeft(shift)).Store((*[4]uint32)(unsafe.Pointer(&dst[off])))

			if int(shift)+bitWidth > 32 {
				rightShift := uint64(32) - shift
				nextOff := (wordIdx+1)*utlSuperWordBytes + group*16
				next := archsimd.LoadUint32x4((*[4]uint32)(unsafe.Pointer(&dst[nextOff])))
				next.Or(val.ShiftAllRight(rightShift)).Store((*[4]uint32)(unsafe.Pointer(&dst[nextOff])))
			}
		}

		bitOffset += bitWidth
	}
}

// --- SSE2 Zigzag Encode/Decode ---

// zigzagEncodeSSE2 applies zigzag encoding using SSE2 (Uint32x4).
// Formula: (n << 1) ^ (n >> 31) with arithmetic right shift.
func zigzagEncodeSSE2(buf []uint32, n int) {
	for i := 0; i <= n-4; i += 4 {
		v := archsimd.LoadUint32x4Slice(buf[i : i+4])
		shifted := v.ShiftAllLeft(1)
		sign := v.AsInt32x4().ShiftAllRight(31).AsUint32x4()
		result := shifted.Xor(sign)
		result.StoreSlice(buf[i : i+4])
	}
	for i := (n / 4) * 4; i < n; i++ {
		buf[i] = zigzagEncode32(int32(buf[i]))
	}
}

// zigzagDecodeSSE2 applies zigzag decoding using SSE2 (Uint32x4).
// Formula: (n >>> 1) ^ -(n & 1).
func zigzagDecodeSSE2(dst, src []uint32) {
	one := archsimd.BroadcastUint32x4(1)
	zero := archsimd.BroadcastUint32x4(0)
	for i := 0; i <= len(src)-4; i += 4 {
		v := archsimd.LoadUint32x4Slice(src[i : i+4])
		half := v.ShiftAllRight(1)
		signBit := v.And(one)
		negSign := zero.Sub(signBit)
		result := half.Xor(negSign)
		result.StoreSlice(dst[i : i+4])
	}
	for i := (len(src) / 4) * 4; i < len(src); i++ {
		dst[i] = uint32(zigzagDecode32(src[i]))
	}
}

// --- SSE2 Per-Lane Delta Encode/Decode ---

// deltaEncodePerLaneSSE2 computes per-lane deltas using SSE2 (Uint32x4).
// Processes all 16 lanes in four register groups.
// Returns true if zigzag encoding was needed (negative deltas detected).
// Borrow detection is batched to eliminate branches from the inner loop.
func deltaEncodePerLaneSSE2(dst, src []uint32) bool {
	var anyBorrow uint8

	for v := utlValuesPerLane - 1; v > 0; v-- {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		// TODO-PERF: Maybe unroll
		for group := 0; group < 4; group++ {
			off := group * 4
			cur := archsimd.LoadUint32x4Slice(src[curBase+off : curBase+off+4])
			prev := archsimd.LoadUint32x4Slice(src[prevBase+off : prevBase+off+4])
			delta := cur.Sub(prev)

			anyBorrow |= prev.Greater(cur).ToBits()

			delta.StoreSlice(dst[curBase+off : curBase+off+4])
		}
	}

	copy(dst[:utlLaneCount], src[:utlLaneCount])

	needZigZag := anyBorrow != 0
	if needZigZag {
		zigzagEncodeSSE2(dst, len(src))
	}
	return needZigZag
}

// deltaDecodePerLaneSSE2 performs per-lane prefix sums using SSE2 (Uint32x4).
func deltaDecodePerLaneSSE2(dst, deltas []uint32, useZigZag bool) {
	if useZigZag {
		zigzagDecodeSSE2(dst, deltas)
	} else {
		copy(dst, deltas)
	}

	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		// TODO-PERF: Maybe unroll
		for group := 0; group < 4; group++ {
			off := group * 4
			prev := archsimd.LoadUint32x4Slice(dst[prevBase+off : prevBase+off+4])
			cur := archsimd.LoadUint32x4Slice(dst[curBase+off : curBase+off+4])
			sum := prev.Add(cur)
			sum.StoreSlice(dst[curBase+off : curBase+off+4])
		}
	}
}

// deltaDecodePerLaneWithOverflowSSE2 performs prefix sum with overflow check
// using SSE2 (Uint32x4). Returns position of first overflow (0 = none).
func deltaDecodePerLaneWithOverflowSSE2(dst, deltas []uint32, useZigZag bool) int {
	if useZigZag {
		deltaDecodePerLaneSSE2(dst, deltas, true)
		return 0
	}

	copy(dst, deltas)
	var overflowPos int

	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		for group := 0; group < 4; group++ {
			off := group * 4
			prev := archsimd.LoadUint32x4Slice(dst[prevBase+off : prevBase+off+4])
			cur := archsimd.LoadUint32x4Slice(dst[curBase+off : curBase+off+4])
			sum := prev.Add(cur)

			overflow := sum.Less(prev)
			if overflowPos == 0 && overflow.ToBits() != 0 {
				lane := bits.TrailingZeros8(overflow.ToBits())
				overflowPos = curBase + off + lane
			}

			sum.StoreSlice(dst[curBase+off : curBase+off+4])
		}
	}

	return overflowPos
}

// packUint32SSE2 is the full SSE2 packing pipeline.
// Uses SSE2 for bit-packing and delta/zigzag; scalar for exceptions and FOR.
func packUint32SSE2(flag byte, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	headerFlags := headerTypeUint32Flag

	useFOR, baseValue, forW := selectBitWidthWithFORSSE2(values)

	if useFOR {
		forSubtractSSE2(values, values, baseValue)
		headerFlags |= uint32(forW) << forWidthShift
	}

	if flag&Delta != 0 {
		var needZZ bool
		if len(values) == blockSize {
			needZZ = deltaEncodePerLaneSSE2(values, values)
		} else {
			needZZ = deltaEncodePerLaneScalar(values, values)
		}
		if needZZ {
			headerFlags |= headerZigZagFlag
		}
		headerFlags |= headerDeltaFlag
	}

	bitWidth, excCount := selectBitWidthSSE2(values)
	payloadBytes := utlPayloadBytesLUT[bitWidth]
	hasExceptions := excCount > 0
	forBaseBytes := forBaseBytesLUT[forW]

	packInput := values
	if len(values) < blockSize {
		copy(scratch[:len(values)], values)
		clear(scratch[len(values):blockSize])
		packInput = scratch[:blockSize]
	}

	if !hasExceptions {
		pOff := payloadOffset(forBaseBytes, false)
		totalLen := pOff + payloadBytes
		dst = ensureLen(dst, totalLen)
		bo.PutUint32(dst, encodeHeader(len(values), bitWidth, 0, headerFlags))
		if useFOR {
			writeFORBase(dst, baseValue, forW, false)
		}
		packLanesUTLSSE2(dst[pOff:pOff+payloadBytes], packInput, bitWidth)
		return dst[:totalLen], nil
	}

	excIdxSize := excIndexSize(excCount)
	maxSvbLen := maxSVBEncodedLen(excCount)
	pOff := payloadOffset(forBaseBytes, true)
	maxTotalLen := pOff + payloadBytes + excIdxSize + maxSvbLen

	dst = ensureLen(dst, maxTotalLen)
	bo.PutUint32(dst, encodeHeader(len(values), bitWidth, excCount, headerFlags))
	if useFOR {
		writeFORBase(dst, baseValue, forW, true)
	}

	packLanesUTLSSE2(dst[pOff:pOff+payloadBytes], packInput, bitWidth)

	highBits := scratch[:blockSize]

	excOff := pOff + payloadBytes
	collectAndWriteExceptions(values, bitWidth, dst[excOff:], excCount, highBits)

	svbOffset := excOff + excIdxSize
	svbLen := encodeSVBIntoDst(dst[svbOffset:maxTotalLen], highBits[:excCount])

	bo.PutUint16(dst[headerBytes:], uint16(svbLen))
	return dst[:svbOffset+svbLen], nil
}

// unpackUint32SSE2 is the full SSE2 unpacking pipeline.
// Uses SSE2 for bit-unpacking and delta/zigzag; scalar for exceptions and FOR.
func unpackUint32SSE2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
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
	unpackLanesUTLSSE2(dst, payload, blockSize, bitWidth)

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
		if hasZigZag {
			if count == blockSize {
				deltaDecodePerLaneSSE2(dst, dst, true)
			} else {
				deltaDecodePerLaneScalar(dst, dst, true)
			}
		} else {
			var overflowPos int
			if count == blockSize {
				overflowPos = deltaDecodePerLaneWithOverflowSSE2(dst, dst, false)
			} else {
				overflowPos = deltaDecodePerLaneWithOverflowScalar(dst, dst, false)
			}
			if overflowPos > 0 {
				return nil, 0, &ErrOverflow{Position: overflowPos}
			}
		}
	}

	if hasFOR {
		forAddSSE2(dst, count, forBase)
	}

	return dst, consumed, nil
}

// selectBitWidthSSE2 computes the optimal step bitwidth using SIMD threshold
// comparisons. Instead of computing bits.Len32 per value and incrementing a
// histogram bin (scatter-add), this approach compares all values against each
// step threshold simultaneously. The 8 thresholds are split into two groups
// of 4 to avoid register pressure (5 SIMD registers per pass: 4 thresholds
// + 1 value vector). AVX-512 uses a single pass (9 of 32 ZMM registers).
func selectBitWidthSSE2(values []uint32) (width int, excCount int) {
	width, excCount, _ = chooseBestFromExcCounts(buildExcCountsSSE2(values))
	return
}

// selectBitWidthWithFORSSE2 uses SIMD-accelerated findMinMax and SIMD
// threshold comparisons for the standard histogram. The two-level rejection
// strategy (min==0, forMaxStep >= rawMaxStep/stdWidth) is preserved.
func selectBitWidthWithFORSSE2(values []uint32) (useFOR bool, baseValue uint32, forWidth int) {
	minVal, maxVal := findMinMaxSSE2(values)
	if minVal == 0 {
		return false, 0, forWidthNone
	}
	forMaxStep := roundUpToStep(bits.Len32(maxVal - minVal))
	rawMaxStep := roundUpToStep(bits.Len32(maxVal))
	if forMaxStep >= rawMaxStep {
		return false, 0, forWidthNone
	}

	stdWidth, _, _ := chooseBestFromExcCounts(buildExcCountsSSE2(values))
	if forMaxStep >= stdWidth {
		return false, 0, forWidthNone
	}
	return true, minVal, selectFORWidth(minVal)
}

// buildExcCountsSSE2 computes cumulative exception counts using SSE2
// threshold comparisons in a single pass.
func buildExcCountsSSE2(values []uint32) (exc [9]int) {
	tA0 := archsimd.BroadcastUint32x4(0)
	tA1 := archsimd.BroadcastUint32x4(0xF)
	tA2 := archsimd.BroadcastUint32x4(0xFF)
	tA3 := archsimd.BroadcastUint32x4(0xFFF)
	tB0 := archsimd.BroadcastUint32x4(0xFFFF)
	tB1 := archsimd.BroadcastUint32x4(0xFFFFF)
	tB2 := archsimd.BroadcastUint32x4(0xFFFFFF)
	tB3 := archsimd.BroadcastUint32x4(0xFFFFFFF)

	i := 0
	for ; i+4 <= len(values); i += 4 {
		v := archsimd.LoadUint32x4Slice(values[i:])
		exc[0] += bits.OnesCount8(v.Greater(tA0).ToBits())
		exc[1] += bits.OnesCount8(v.Greater(tA1).ToBits())
		exc[2] += bits.OnesCount8(v.Greater(tA2).ToBits())
		exc[3] += bits.OnesCount8(v.Greater(tA3).ToBits())
		exc[4] += bits.OnesCount8(v.Greater(tB0).ToBits())
		exc[5] += bits.OnesCount8(v.Greater(tB1).ToBits())
		exc[6] += bits.OnesCount8(v.Greater(tB2).ToBits())
		exc[7] += bits.OnesCount8(v.Greater(tB3).ToBits())
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

// findMinMaxSSE2 computes min/max using SSE2 4-wide operations.
// Uses pointer-based loads to avoid per-iteration slice bounds checking.
func findMinMaxSSE2(values []uint32) (uint32, uint32) {
	n := len(values)
	if n < 4 {
		return findMinMaxScalar(values)
	}

	p := unsafe.Pointer(&values[0])
	minVec := archsimd.LoadUint32x4((*[4]uint32)(p))
	maxVec := minVec

	end := uintptr(n) * 4
	for off := uintptr(16); off+16 <= end; off += 16 {
		chunk := archsimd.LoadUint32x4((*[4]uint32)(unsafe.Pointer(uintptr(p) + off)))
		minVec = minVec.Min(chunk)
		maxVec = maxVec.Max(chunk)
	}

	var minLanes, maxLanes [4]uint32
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

	tail := (n / 4) * 4
	for i := tail; i < n; i++ {
		if values[i] < minResult {
			minResult = values[i]
		}
		if values[i] > maxResult {
			maxResult = values[i]
		}
	}
	return minResult, maxResult
}

// forSubtractSSE2 subtracts baseValue from each element using SSE2.
func forSubtractSSE2(dst, src []uint32, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x4(baseValue)
	i := 0
	for ; i+4 <= len(src); i += 4 {
		v := archsimd.LoadUint32x4Slice(src[i:])
		v = v.Sub(baseVec)
		v.StoreSlice(dst[i:])
	}
	for ; i < len(src); i++ {
		dst[i] = src[i] - baseValue
	}
}

// forAddSSE2 adds baseValue to each of the first count elements using SSE2.
func forAddSSE2(output []uint32, count int, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x4(baseValue)
	i := 0
	for ; i+4 <= count; i += 4 {
		v := archsimd.LoadUint32x4Slice(output[i:])
		v = v.Add(baseVec)
		v.StoreSlice(output[i:])
	}
	for ; i < count; i++ {
		output[i] += baseValue
	}
}

// getUint32SSE2 delegates to scalar for random access.
func getUint32SSE2(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
