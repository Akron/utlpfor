//go:build goexperiment.simd && amd64

package utlpfor

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

// unpackLanesUTLAVX2 unpacks UTL payload using AVX2 (Uint32x8).
// Dispatches to per-bitwidth specialized functions for step bitwidths.
func unpackLanesUTLAVX2(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	switch bitWidth {
	case 4:
		unpackAVX2BW4(&dst[0], &payload[0])
	case 8:
		unpackAVX2BW8(&dst[0], &payload[0])
	case 12:
		unpackAVX2BW12(&dst[0], &payload[0])
	case 16:
		unpackAVX2BW16(&dst[0], &payload[0])
	case 20:
		unpackAVX2BW20(&dst[0], &payload[0])
	case 24:
		unpackAVX2BW24(&dst[0], &payload[0])
	case 28:
		unpackAVX2BW28(&dst[0], &payload[0])
	case 32:
		unpackAVX2BW32(&dst[0], &payload[0])
	default:
		unpackLanesUTLAVX2Generic(dst, payload, count, bitWidth)
	}
}

// unpackLanesUTLAVX2Generic is the generic loop-based AVX2 unpacker.
// Used as fallback for non-step bitwidths (should not be reached in practice).
func unpackLanesUTLAVX2Generic(dst []uint32, payload []byte, count, bitWidth int) {
	mask := archsimd.BroadcastUint32x8(uint32((1 << bitWidth) - 1))
	if bitWidth == 32 {
		mask = archsimd.BroadcastUint32x8(0xFFFFFFFF)
	}

	bitOffset := 0
	for v := range utlValuesPerLane {
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
// Dispatches to per-bitwidth specialized native archsimd functions.
// Caller must ensure values has at least blockSize elements (zero-padded).
func packLanesUTLAVX2(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	if bitWidth != 32 {
		clear(dst)
	}
	switch bitWidth {
	case 4:
		packAVX2BW4(&dst[0], &values[0])
	case 8:
		packAVX2BW8(&dst[0], &values[0])
	case 12:
		packAVX2BW12(&dst[0], &values[0])
	case 16:
		packAVX2BW16(&dst[0], &values[0])
	case 20:
		packAVX2BW20(&dst[0], &values[0])
	case 24:
		packAVX2BW24(&dst[0], &values[0])
	case 28:
		packAVX2BW28(&dst[0], &values[0])
	case 32:
		packAVX2BW32(&dst[0], &values[0])
	default:
		packLanesUTLAVX2Generic(dst, values, bitWidth)
	}
}

// packLanesUTLAVX2Generic is the generic loop-based AVX2 packer.
// Used as fallback for non-step bitwidths (should not be reached in practice).
func packLanesUTLAVX2Generic(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 32 {
		for v := range utlValuesPerLane {
			inBase := v * utlLaneCount
			base := v * utlSuperWordBytes
			archsimd.LoadUint32x8Slice(values[inBase : inBase+8]).Store(
				(*[8]uint32)(unsafe.Pointer(&dst[base])))
			archsimd.LoadUint32x8Slice(values[inBase+8 : inBase+16]).Store(
				(*[8]uint32)(unsafe.Pointer(&dst[base+32])))
		}
		return
	}

	mask := archsimd.BroadcastUint32x8(uint32((1 << bitWidth) - 1))

	bitOffset := 0
	for v := range utlValuesPerLane {
		inBase := v * utlLaneCount
		vLo := archsimd.LoadUint32x8Slice(values[inBase : inBase+8]).And(mask)
		vHi := archsimd.LoadUint32x8Slice(values[inBase+8 : inBase+16]).And(mask)

		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		curLo := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&dst[base])))
		curHi := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&dst[base+32])))
		curLo.Or(vLo.ShiftAllLeft(shift)).Store((*[8]uint32)(unsafe.Pointer(&dst[base])))
		curHi.Or(vHi.ShiftAllLeft(shift)).Store((*[8]uint32)(unsafe.Pointer(&dst[base+32])))

		if int(shift)+bitWidth > 32 {
			rightShift := uint64(32) - shift
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			nextLo := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&dst[nextBase])))
			nextHi := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&dst[nextBase+32])))
			nextLo.Or(vLo.ShiftAllRight(rightShift)).Store((*[8]uint32)(unsafe.Pointer(&dst[nextBase])))
			nextHi.Or(vHi.ShiftAllRight(rightShift)).Store((*[8]uint32)(unsafe.Pointer(&dst[nextBase+32])))
		}

		bitOffset += bitWidth
	}
}

// zigzagEncodeAVX2 applies zigzag encoding to all values using AVX2.
// Formula: (n << 1) ^ (n >> 31) where >> is arithmetic right shift.
func zigzagEncodeAVX2(buf []uint32, n int) {

	// TODO-PERF: Unroll?
	for i := 0; i <= n-8; i += 8 {
		v := archsimd.LoadUint32x8Slice(buf[i : i+8])
		shifted := v.ShiftAllLeft(1)
		sign := v.AsInt32x8().ShiftAllRight(31).AsUint32x8()
		result := shifted.Xor(sign)
		result.StoreSlice(buf[i : i+8])
	}
	// TODO-PERF: Use SSE for the tail initially
	for i := (n / 8) * 8; i < n; i++ {
		buf[i] = zigzagEncode32(int32(buf[i]))
	}
}

// zigzagDecodeAVX2 applies zigzag decoding in-place using AVX2.
// Formula: (n >>> 1) ^ -(n & 1) where >>> is logical right shift.
func zigzagDecodeAVX2(values []uint32) {
	one := archsimd.BroadcastUint32x8(1)
	zero := archsimd.BroadcastUint32x8(0)
	// TODO-PERF: Unroll?
	for i := 0; i <= len(values)-8; i += 8 {
		v := archsimd.LoadUint32x8Slice(values[i : i+8])
		half := v.ShiftAllRight(1)
		signBit := v.And(one)
		negSign := zero.Sub(signBit)
		result := half.Xor(negSign)
		result.StoreSlice(values[i : i+8])
	}
	// TODO-PERF: Use SSE for the tail initially
	for i := (len(values) / 8) * 8; i < len(values); i++ {
		values[i] = uint32(zigzagDecode32(values[i]))
	}
}

// deltaEncodePerLaneAVX2 computes per-lane deltas using AVX2.
// Processes all 16 lanes in two register groups (lanes 0-7 and 8-15).
// Returns true if zigzag encoding was needed (negative deltas detected).
// Borrow detection is batched: all borrow masks are ORed into a single uint8
// and checked once after the loop, eliminating branches from the inner loop.
func deltaEncodePerLaneAVX2(values []uint32) bool {
	var anyBorrow uint8

	// TODO-PERF: Unroll!
	for v := utlValuesPerLane - 1; v > 0; v-- {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		cur0 := archsimd.LoadUint32x8Slice(values[curBase : curBase+8])
		prev0 := archsimd.LoadUint32x8Slice(values[prevBase : prevBase+8])
		delta0 := cur0.Sub(prev0)

		cur1 := archsimd.LoadUint32x8Slice(values[curBase+8 : curBase+16])
		prev1 := archsimd.LoadUint32x8Slice(values[prevBase+8 : prevBase+16])
		delta1 := cur1.Sub(prev1)

		anyBorrow |= prev0.Greater(cur0).ToBits()
		anyBorrow |= prev1.Greater(cur1).ToBits()

		delta0.StoreSlice(values[curBase : curBase+8])
		delta1.StoreSlice(values[curBase+8 : curBase+16])
	}

	needZigZag := anyBorrow != 0
	if needZigZag {
		zigzagEncodeAVX2(values, len(values))
	}
	return needZigZag
}

// deltaDecodePerLaneAVX2 performs in-place per-lane prefix sums using AVX2.
func deltaDecodePerLaneAVX2(values []uint32, useZigZag bool) {
	if useZigZag {
		zigzagDecodeAVX2(values)
	}

	// TODO-PERF: Unroll!
	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		prev0 := archsimd.LoadUint32x8Slice(values[prevBase : prevBase+8])
		cur0 := archsimd.LoadUint32x8Slice(values[curBase : curBase+8])
		sum0 := prev0.Add(cur0)
		sum0.StoreSlice(values[curBase : curBase+8])

		prev1 := archsimd.LoadUint32x8Slice(values[prevBase+8 : prevBase+16])
		cur1 := archsimd.LoadUint32x8Slice(values[curBase+8 : curBase+16])
		sum1 := prev1.Add(cur1)
		sum1.StoreSlice(values[curBase+8 : curBase+16])
	}
}

// deltaDecodePerLaneWithOverflowAVX2 performs prefix sum with overflow check.
// Returns the position of the first overflow (0 = no overflow).
func deltaDecodePerLaneWithOverflowAVX2(values []uint32, useZigZag bool) int {
	if useZigZag {
		deltaDecodePerLaneAVX2(values, true)
		return 0
	}

	var overflowPos int

	// TODO-PERF: Unroll!
	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		prev0 := archsimd.LoadUint32x8Slice(values[prevBase : prevBase+8])
		cur0 := archsimd.LoadUint32x8Slice(values[curBase : curBase+8])
		sum0 := prev0.Add(cur0)

		overflow0 := sum0.Less(prev0)
		if overflowPos == 0 && overflow0.ToBits() != 0 {
			lane := bits.TrailingZeros8(overflow0.ToBits())
			overflowPos = curBase + lane
		}

		sum0.StoreSlice(values[curBase : curBase+8])

		prev1 := archsimd.LoadUint32x8Slice(values[prevBase+8 : prevBase+16])
		cur1 := archsimd.LoadUint32x8Slice(values[curBase+8 : curBase+16])
		sum1 := prev1.Add(cur1)

		overflow1 := sum1.Less(prev1)
		if overflowPos == 0 && overflow1.ToBits() != 0 {
			lane := bits.TrailingZeros8(overflow1.ToBits())
			overflowPos = curBase + 8 + lane
		}

		sum1.StoreSlice(values[curBase+8 : curBase+16])
	}

	return overflowPos
}

// packUint32AVX2 is the full AVX2 packing pipeline.
// Uses AVX2 for bit-packing and delta/zigzag encoding; scalar for exceptions and FOR.
func packUint32AVX2(flag byte, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	headerFlags := headerTypeUint32Flag

	var useFOR bool
	var baseValue uint32
	var forW int
	if flag&NoFOR == 0 {
		useFOR, baseValue, forW = selectBitWidthWithFORAVX2(values)
		if useFOR {
			forSubtractAVX2(values, values, baseValue)
			headerFlags |= uint32(forW) << forWidthShift
		}
	}

	if flag&Delta != 0 {
		var needZZ bool
		if len(values) == blockSize {
			needZZ = deltaEncodePerLaneAVX2(values)
		} else {
			needZZ = deltaEncodePerLaneScalar(values)
		}
		if needZZ {
			headerFlags |= headerZigZagFlag
		}
		headerFlags |= headerDeltaFlag
	}

	var bitWidth, excCount int
	if flag&NoPatch != 0 {
		bitWidth = selectBitWidthNoPatchAVX2(values)
	} else {
		bitWidth, excCount = selectBitWidthAVX2(values)
	}
	payloadBytes := utlPayloadBytes(bitWidth)
	hasExceptions := excCount > 0
	forBaseBytes := forBaseBytes(forW)

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
		packLanesUTLAVX2(dst[pOff:pOff+payloadBytes], packInput, bitWidth)
		archsimd.ClearAVXUpperBits()
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

	packLanesUTLAVX2(dst[pOff:pOff+payloadBytes], packInput, bitWidth)
	archsimd.ClearAVXUpperBits()

	highBits := scratch[:blockSize]

	excOff := pOff + payloadBytes
	collectAndWriteExceptions(values, bitWidth, dst[excOff:], excCount, highBits)

	svbOffset := excOff + excIdxSize
	svbLen := encodeSVBIntoDst(dst[svbOffset:maxTotalLen], highBits[:excCount])

	bo.PutUint16(dst[headerBytes:], uint16(svbLen))
	return dst[:svbOffset+svbLen], nil
}

// unpackUint32AVX2 is the AVX2 unpacking pipeline.
// It uses SSE2 per-bitwidth unpack kernels for the lane unpack step because
// AVX2 right-shift codegen is still suboptimal on current toolchains.
// AVX2 remains active for delta/zigzag and FOR operations.
func unpackUint32AVX2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
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
		pOff += forBaseBytes(forWidth)
	}

	payloadBytes := utlPayloadBytes(bitWidth)

	if len(buf) < pOff+payloadBytes {
		return nil, 0, ErrInvalidBuffer
	}

	if cap(dst) < blockSize {
		dst = make([]uint32, blockSize)
	}
	dst = dst[:blockSize]

	payload := buf[pOff : pOff+payloadBytes]

	// This requires https://github.com/golang/go/commit/aa80d7a7e6bf97aa27a74cc5056ef270a2a0c2f4
	// which will likely be in Go 1.27.
	// Until then, we may need SSE2 unpacker or gotip
	unpackLanesUTLAVX2(dst, payload, blockSize, bitWidth)

	dst = dst[:count]

	consumed := pOff + payloadBytes

	if hasExceptions {
		archsimd.ClearAVXUpperBits()
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
				deltaDecodePerLaneAVX2(dst, true)
			} else {
				archsimd.ClearAVXUpperBits()
				deltaDecodePerLaneScalar(dst, true)
			}
		} else {
			var overflowPos int
			if count == blockSize {
				overflowPos = deltaDecodePerLaneWithOverflowAVX2(dst, false)
			} else {
				archsimd.ClearAVXUpperBits()
				overflowPos = deltaDecodePerLaneWithOverflowScalar(dst, false)
			}
			if overflowPos > 0 {
				archsimd.ClearAVXUpperBits()
				return nil, 0, &ErrOverflow{Position: overflowPos}
			}
		}
	}

	if hasFOR {
		forAddAVX2(dst, count, forBase)
	}

	archsimd.ClearAVXUpperBits()
	return dst, consumed, nil
}

// selectBitWidthAVX2 computes the optimal step bitwidth using SIMD threshold
// comparisons. Instead of computing bits.Len32 per value and incrementing a
// histogram bin (scatter-add), this approach compares all values against each
// step threshold simultaneously. The 8 thresholds are split into two groups
// of 4 to avoid register pressure (5 SIMD registers per pass: 4 thresholds
// + 1 value vector). AVX-512 uses a single pass (9 of 32 ZMM registers).
func selectBitWidthAVX2(values []uint32) (width int, excCount int) {
	width, excCount, _ = chooseBestFromExcCounts(buildExcCountsAVX2(values))
	return
}

// selectBitWidthWithFORAVX2 uses the optimized AVX2 findMinMax (2 accumulators,
// pointer-based loads) followed by the optimized buildExcCountsAVX2. Same
// rationale as SSE2: separate passes outperform the fused 4+4 split approach.
func selectBitWidthWithFORAVX2(values []uint32) (useFOR bool, baseValue uint32, forWidth int) {
	minVal, maxVal := findMinMaxAVX2(values)
	if minVal == 0 {
		return false, 0, forWidthNone
	}
	forMaxStep := roundUpToStep(bits.Len32(maxVal - minVal))
	rawMaxStep := roundUpToStep(bits.Len32(maxVal))
	if forMaxStep >= rawMaxStep {
		return false, 0, forWidthNone
	}

	exc := buildExcCountsAVX2(values)
	stdWidth, _, _ := chooseBestFromExcCounts(exc)
	if forMaxStep >= stdWidth {
		return false, 0, forWidthNone
	}
	return true, minVal, selectFORWidth(minVal)
}

// buildExcCountsAVX2 computes cumulative exception counts using AVX2
// threshold comparisons in a single pass over the data. All 8 thresholds
// are compared per iteration, requiring 9 YMM registers (8 thresholds +
// 1 value vector) of 16 available -- matching the SSE2 single-pass
// approach but with 8-wide operations.
func buildExcCountsAVX2(values []uint32) (exc [9]int) {
	t0 := archsimd.BroadcastUint32x8(0)
	t1 := archsimd.BroadcastUint32x8(0xF)
	t2 := archsimd.BroadcastUint32x8(0xFF)
	t3 := archsimd.BroadcastUint32x8(0xFFF)
	t4 := archsimd.BroadcastUint32x8(0xFFFF)
	t5 := archsimd.BroadcastUint32x8(0xFFFFF)
	t6 := archsimd.BroadcastUint32x8(0xFFFFFF)
	t7 := archsimd.BroadcastUint32x8(0xFFFFFFF)

	i := 0
	for ; i+8 <= len(values); i += 8 {
		v := archsimd.LoadUint32x8Slice(values[i:])
		exc[0] += bits.OnesCount8(v.Greater(t0).ToBits())
		exc[1] += bits.OnesCount8(v.Greater(t1).ToBits())
		exc[2] += bits.OnesCount8(v.Greater(t2).ToBits())
		exc[3] += bits.OnesCount8(v.Greater(t3).ToBits())
		exc[4] += bits.OnesCount8(v.Greater(t4).ToBits())
		exc[5] += bits.OnesCount8(v.Greater(t5).ToBits())
		exc[6] += bits.OnesCount8(v.Greater(t6).ToBits())
		exc[7] += bits.OnesCount8(v.Greater(t7).ToBits())
	}
	// TODO-PERF: Use SSE for the tail initially
	for ; i < len(values); i++ {
		v := values[i]
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

// findMinMaxAVX2 computes min/max using AVX2 8-wide operations with 2
// independent accumulator pairs for instruction-level parallelism.
// Uses pointer-based loads to avoid per-iteration slice bounds checking.
// Register budget: 4 accumulators + 2 chunk temps = 6 YMM (of 16 available).
func findMinMaxAVX2(values []uint32) (uint32, uint32) {
	n := len(values)
	if n < 16 {
		return findMinMaxSSE2(values)
	}

	p := unsafe.Pointer(&values[0])
	min0 := archsimd.LoadUint32x8((*[8]uint32)(p))
	min1 := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Add(p, 32)))
	max0, max1 := min0, min1

	end := uintptr(n) * 4
	for off := uintptr(64); off+64 <= end; off += 64 {
		c0 := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Add(p, off)))
		c1 := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(uintptr(p) + off + 32)))
		min0 = min0.Min(c0)
		max0 = max0.Max(c0)
		min1 = min1.Min(c1)
		max1 = max1.Max(c1)
	}

	minVec := min0.Min(min1)
	maxVec := max0.Max(max1)

	var minLanes, maxLanes [8]uint32
	minVec.Store(&minLanes)
	maxVec.Store(&maxLanes)
	minResult, maxResult := minLanes[0], maxLanes[0]

	// TODO-PERF: Unroll?
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

	// TODO-PERF: Use SSE for the tail initially
	tail := (n / 16) * 16
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

// selectBitWidthNoPatchAVX2 computes the minimum step bitwidth using AVX2
// OR-reduction. No exception analysis is performed.
func selectBitWidthNoPatchAVX2(values []uint32) int {
	orVec := archsimd.BroadcastUint32x8(0)
	i := 0
	for ; i+8 <= len(values); i += 8 {
		orVec = orVec.Or(archsimd.LoadUint32x8Slice(values[i:]))
	}
	var lanes [8]uint32
	orVec.Store(&lanes)
	var orAll uint32
	for _, v := range lanes {
		orAll |= v
	}
	// TODO-PERF: Fallback to SSE2 for the tail
	for ; i < len(values); i++ {
		orAll |= values[i]
	}
	return roundUpToStep(bits.Len32(orAll))
}

// forSubtractAVX2 subtracts baseValue from each element using AVX2.
func forSubtractAVX2(dst, src []uint32, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x8(baseValue)
	i := 0
	// TODO-PERF: Unroll?
	for ; i+8 <= len(src); i += 8 {
		archsimd.LoadUint32x8Slice(src[i:]).Sub(baseVec).StoreSlice(dst[i:])
	}
	for ; i < len(src); i++ {
		dst[i] = src[i] - baseValue
	}
}

// forAddAVX2 adds baseValue to each of the first count elements using AVX2.
func forAddAVX2(output []uint32, count int, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x8(baseValue)
	i := 0
	// TODO-PERF: Unroll?
	for ; i+8 <= count; i += 8 {
		archsimd.LoadUint32x8Slice(output[i:]).Add(baseVec).StoreSlice(output[i:])
	}
	for ; i < count; i++ {
		output[i] += baseValue
	}
}

// getUint32AVX2 delegates to scalar for random access.
func getUint32AVX2(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
