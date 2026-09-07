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

		lo := archsimd.LoadUint32x8Array(
			(*[8]uint32)(unsafe.Pointer(&payload[base])))
		hi := archsimd.LoadUint32x8Array(
			(*[8]uint32)(unsafe.Pointer(&payload[base+32])))

		rLo := lo.ShiftAllRight(shift).And(mask)
		rHi := hi.ShiftAllRight(shift).And(mask)

		if int(shift)+bitWidth > 32 {
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			nextLo := archsimd.LoadUint32x8Array(
				(*[8]uint32)(unsafe.Pointer(&payload[nextBase])))
			nextHi := archsimd.LoadUint32x8Array(
				(*[8]uint32)(unsafe.Pointer(&payload[nextBase+32])))
			leftShift := uint64(32) - shift
			rLo = rLo.Or(nextLo.ShiftAllLeft(leftShift).And(mask))
			rHi = rHi.Or(nextHi.ShiftAllLeft(leftShift).And(mask))
		}

		outBase := v * utlLaneCount
		rLo.Store(dst[outBase : outBase+8])
		rHi.Store(dst[outBase+8 : outBase+16])

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
			archsimd.LoadUint32x8(values[inBase : inBase+8]).StoreArray(
				(*[8]uint32)(unsafe.Pointer(&dst[base])))
			archsimd.LoadUint32x8(values[inBase+8 : inBase+16]).StoreArray(
				(*[8]uint32)(unsafe.Pointer(&dst[base+32])))
		}
		return
	}

	mask := archsimd.BroadcastUint32x8(uint32((1 << bitWidth) - 1))

	bitOffset := 0
	for v := range utlValuesPerLane {
		inBase := v * utlLaneCount
		vLo := archsimd.LoadUint32x8(values[inBase : inBase+8]).And(mask)
		vHi := archsimd.LoadUint32x8(values[inBase+8 : inBase+16]).And(mask)

		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		curLo := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&dst[base])))
		curHi := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&dst[base+32])))
		curLo.Or(vLo.ShiftAllLeft(shift)).StoreArray((*[8]uint32)(unsafe.Pointer(&dst[base])))
		curHi.Or(vHi.ShiftAllLeft(shift)).StoreArray((*[8]uint32)(unsafe.Pointer(&dst[base+32])))

		if int(shift)+bitWidth > 32 {
			rightShift := uint64(32) - shift
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			nextLo := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&dst[nextBase])))
			nextHi := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&dst[nextBase+32])))
			nextLo.Or(vLo.ShiftAllRight(rightShift)).StoreArray((*[8]uint32)(unsafe.Pointer(&dst[nextBase])))
			nextHi.Or(vHi.ShiftAllRight(rightShift)).StoreArray((*[8]uint32)(unsafe.Pointer(&dst[nextBase+32])))
		}

		bitOffset += bitWidth
	}
}

// zigzagEncodeAVX2 applies zigzag encoding to all values using AVX2.
// Formula: (n << 1) ^ (n >> 31) where >> is arithmetic right shift.
func zigzagEncodeAVX2(buf []uint32, n int) {
	// Pointer-based loads/stores: eliminates per-iteration bounds checks.
	if n >= 8 {
		p := unsafe.Pointer(&buf[0])
		end := uintptr(n) * 4
		// TODO-PERF: Unroll?
		for off := uintptr(0); off+32 <= end; off += 32 {
			v := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(p, off)))
			shifted := v.ShiftAllLeft(1)
			sign := v.AsInt32x8().ShiftAllRight(31).AsUint32x8()
			shifted.Xor(sign).StoreArray((*[8]uint32)(unsafe.Add(p, off)))
		}
		// TODO-PERF: Use SSE for the tail initially
	}
	// Scalar tail for n < 8 or leftover values.
	for i := (n / 8) * 8; i < n; i++ {
		buf[i] = zigzagEncode32(int32(buf[i]))
	}
}

// zigzagDecodeAVX2 applies zigzag decoding in-place using AVX2.
// Formula: (n >>> 1) ^ -(n & 1) where >>> is logical right shift.
func zigzagDecodeAVX2(values []uint32) {
	// Broadcast constants hoisted out of the loop; pointer loads avoid
	// per-iteration bounds checks.
	one := archsimd.BroadcastUint32x8(1)
	zero := archsimd.BroadcastUint32x8(0)
	n := len(values)
	if n >= 8 {
		p := unsafe.Pointer(&values[0])
		end := uintptr(n) * 4
		// TODO-PERF: Unroll?
		for off := uintptr(0); off+32 <= end; off += 32 {
			v := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(p, off)))
			half := v.ShiftAllRight(1)   // n >>> 1 (logical right shift)
			signBit := v.And(one)        // n & 1: extract sign from LSB
			negSign := zero.Sub(signBit) // 0 - signBit: yields 0x00000000 or 0xFFFFFFFF
			half.Xor(negSign).StoreArray((*[8]uint32)(unsafe.Add(p, off)))
		}
	}
	// Scalar tail for n < 8 or leftover values.
	// TODO-PERF: Unroll?
	for i := (n / 8) * 8; i < n; i++ {
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

	// Iterate backwards: each position stores (cur - prev) for all 16 lanes.
	// Two 8-wide groups cover lanes 0-7 and 8-15 per value position.
	// TODO-PERF: Unroll!
	for v := utlValuesPerLane - 1; v > 0; v-- {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		cur0 := archsimd.LoadUint32x8(values[curBase : curBase+8])
		prev0 := archsimd.LoadUint32x8(values[prevBase : prevBase+8])
		delta0 := cur0.Sub(prev0)

		cur1 := archsimd.LoadUint32x8(values[curBase+8 : curBase+16])
		prev1 := archsimd.LoadUint32x8(values[prevBase+8 : prevBase+16])
		delta1 := cur1.Sub(prev1)

		// Detect unsigned underflow (prev > cur means negative delta)
		anyBorrow |= prev0.Greater(cur0).ToBits()
		anyBorrow |= prev1.Greater(cur1).ToBits()

		delta0.Store(values[curBase : curBase+8])
		delta1.Store(values[curBase+8 : curBase+16])
	}

	// If any lane had a negative delta, apply zigzag to make all values positive
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

		prev0 := archsimd.LoadUint32x8(values[prevBase : prevBase+8])
		cur0 := archsimd.LoadUint32x8(values[curBase : curBase+8])
		sum0 := prev0.Add(cur0)
		sum0.Store(values[curBase : curBase+8])

		prev1 := archsimd.LoadUint32x8(values[prevBase+8 : prevBase+16])
		cur1 := archsimd.LoadUint32x8(values[curBase+8 : curBase+16])
		sum1 := prev1.Add(cur1)
		sum1.Store(values[curBase+8 : curBase+16])
	}
}

// deltaDecodePerLaneWithOverflowAVX2 performs per-lane prefix sum (delta
// decode) using AVX2 YMM registers and detects unsigned overflow.
//
// Returns the flat index (row*16 + lane) of the first addition that wrapped
// past 2^32, or 0 if no overflow. The scan is v-major: lowest row wins,
// ties broken by lowest lane - matching the scalar reference exactly.
//
// Design: instead of checking for overflow inside the SIMD loop (which
// requires a data-dependent branch per row), the per-row overflow masks are
// collected into a [7]uint16 array and the first overflow is resolved in a
// short scan after the loop. This gives the CPU better ILP because the
// compare + mask-OR is off the store dependency chain.
func deltaDecodePerLaneWithOverflowAVX2(values []uint32, useZigZag bool) int {
	if useZigZag {
		deltaDecodePerLaneAVX2(values, true)
		return 0
	}

	// masks[v] holds the overflow mask of row v: bits 0-7 from vector 0
	// (lanes 0-7), bits 8-15 from vector 1 (lanes 8-15). Row 0 has no
	// previous element, so v starts at 1 and masks has 7 entries.
	var masks [utlValuesPerLane - 1]uint16

	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		prev0 := archsimd.LoadUint32x8(values[prevBase : prevBase+8])
		cur0 := archsimd.LoadUint32x8(values[curBase : curBase+8])
		sum0 := prev0.Add(cur0)

		prev1 := archsimd.LoadUint32x8(values[prevBase+8 : prevBase+16])
		cur1 := archsimd.LoadUint32x8(values[curBase+8 : curBase+16])
		sum1 := prev1.Add(cur1)

		// Sum < prev means the unsigned addition wrapped (overflow).
		m0 := sum0.Less(prev0).ToBits()
		m1 := sum1.Less(prev1).ToBits()

		sum0.Store(values[curBase : curBase+8])
		sum1.Store(values[curBase+8 : curBase+16])

		// Collect the mask; the branch is data-dependent but at most once
		// per row and not on the store path (better ILP than lane-major).
		// uint16: m1 needs bits 8-15, above the uint8 range of m0.
		if m0|m1 != 0 {
			masks[v-1] = uint16(m0) | uint16(m1)<<8
		}
	}

	// Resolve the first overflow in v-major order after the loop: the
	// smallest row v with any overflow bit wins; within the row, the lowest
	// set bit (lowest lane) wins. lane-order index = v*16 + lane.
	for v, m := range masks {
		if m != 0 {
			return (v+1)*utlLaneCount + bits.TrailingZeros16(m)
		}
	}
	return 0
}

// packUint32AVX2 is the AVX2 packing pipeline. Delegates to packBlockAVX2
// with uint32 type flags and Append handling.
func packUint32AVX2(flag Flag, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}
	if flag&Append == 0 {
		return packBlockAVX2(flag, values, dst, scratch, headerTypeUint32Flag, false)
	}
	off := len(dst)
	dst = ensureCapacity32(dst, off)
	block, err := packBlockAVX2(flag&^Append, values, dst[off:off], scratch, headerTypeUint32Flag, false)
	if err != nil {
		return nil, err
	}
	return dst[:off+len(block)], nil
}

// unpackUint32AVX2 is the AVX2 unpacking pipeline. Delegates to
// unpackBlockAVX2 with uint32 context (forUint64=false).
func unpackUint32AVX2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackBlockAVX2(dst, scratch, buf, false)
}

// unpackBlockAVX2 with uint64 sub-block context (forUint64=true).
// Used by GetUint64 for SIMD-accelerated delta full-unpack fallback.
func unpackBlockForUint64AVX2(dst, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackBlockAVX2(dst, scratch, buf, true)
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

// findMinMaxAVX2 computes min/max using AVX2 8-wide operations with 2
// independent accumulator pairs for instruction-level parallelism.
// Uses pointer-based loads to avoid per-iteration slice bounds checking.
// Register budget: 4 accumulators + 2 chunk temps = 6 YMM (of 16 available).
func findMinMaxAVX2(values []uint32) (uint32, uint32) {
	n := len(values)
	if n < 16 {
		return findMinMaxSSE2(values)
	}

	// Prime two independent 8-lane accumulators from the first 16 values.
	p := unsafe.Pointer(&values[0])
	min0 := archsimd.LoadUint32x8Array((*[8]uint32)(p))
	min1 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(p, 32)))
	max0, max1 := min0, min1

	// Process two vectors per iteration to keep both accumulator chains busy.
	end := uintptr(n) * 4
	for off := uintptr(64); off+64 <= end; off += 64 {
		c0 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(p, off)))
		c1 := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(uintptr(p) + off + 32)))
		min0 = min0.Min(c0)
		max0 = max0.Max(c0)
		min1 = min1.Min(c1)
		max1 = max1.Max(c1)
	}

	minVec := min0.Min(min1)
	maxVec := max0.Max(max1)

	// Horizontally reduce 8-lane vectors down to scalar min/max values.
	minR4 := minVec.GetLo().Min(minVec.GetHi())
	maxR4 := maxVec.GetLo().Max(maxVec.GetHi())
	var minL, maxL [4]uint32
	minR4.StoreArray(&minL)
	maxR4.StoreArray(&maxL)
	minResult := min(min(minL[0], minL[1]), min(minL[2], minL[3]))
	maxResult := max(max(maxL[0], maxL[1]), max(maxL[2], maxL[3]))

	// TODO-PERF: Use SSE2 for the tail
	// Handle the leftover elements that do not fill a 16-value AVX2 block.
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
	// Pointer-based loads: eliminates the per-iteration slice bounds check.
	orVec := archsimd.BroadcastUint32x8(0)
	n := len(values)
	if n >= 8 {
		p := unsafe.Pointer(&values[0])
		end := uintptr(n) * 4
		for off := uintptr(0); off+32 <= end; off += 32 {
			orVec = orVec.Or(archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(p, off))))
		}
	}
	var lanes [8]uint32
	orVec.StoreArray(&lanes)
	var orAll uint32
	for _, v := range lanes {
		orAll |= v
	}
	// TODO-PERF: Fallback to SSE2 for the tail
	// Scalar tail (n < 8 or leftover values after the last full vector).
	for i := (n / 8) * 8; i < n; i++ {
		orAll |= values[i]
	}
	return roundUpToStep(bits.Len32(orAll))
}

// forSubtractAVX2 subtracts baseValue from each element using AVX2.
func forSubtractAVX2(dst, src []uint32, baseValue uint32) {
	// Broadcast once outside the loop; pointer loads avoid bounds checks.
	baseVec := archsimd.BroadcastUint32x8(baseValue)
	n := len(src)
	if n >= 8 {
		sp := unsafe.Pointer(&src[0])
		dp := unsafe.Pointer(&dst[0])
		end := uintptr(n) * 4
		// TODO-PERF: Unroll?
		for off := uintptr(0); off+32 <= end; off += 32 {
			archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(sp, off))).
				Sub(baseVec).
				StoreArray((*[8]uint32)(unsafe.Add(dp, off)))
		}
	}
	// Scalar tail for n < 8 or leftover values.
	for i := (n / 8) * 8; i < n; i++ {
		dst[i] = src[i] - baseValue
	}
}

// forAddAVX2 adds baseValue to each of the first count elements using AVX2.
func forAddAVX2(output []uint32, count int, baseValue uint32) {
	// Broadcast once outside the loop; pointer loads avoid bounds checks.
	baseVec := archsimd.BroadcastUint32x8(baseValue)
	if count >= 8 {
		p := unsafe.Pointer(&output[0])
		end := uintptr(count) * 4
		// TODO-PERF: Unroll?
		for off := uintptr(0); off+32 <= end; off += 32 {
			archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Add(p, off))).
				Add(baseVec).
				StoreArray((*[8]uint32)(unsafe.Add(p, off)))
		}
	}
	// Scalar tail for count < 8 or leftover elements.
	for i := (count / 8) * 8; i < count; i++ {
		output[i] += baseValue
	}
}

// deinterleaveIdx selects even-indexed dwords (lower halves) into the
// low 128-bit lane and odd-indexed dwords (upper halves) into the high
// lane: [lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3] -> [lo0,lo1,lo2,lo3,hi0,hi1,hi2,hi3].
var deinterleaveIdx = [8]uint32{0, 2, 4, 6, 1, 3, 5, 7}

// analyzeUint64AVX2 computes OR-accumulator and min/max of uint64 values
// in a single fused pass. SIMD Uint64x4 handles OR-accumulation (4 values
// per iteration) while scalar pair-comparison handles min/max (VPMINUQ
// requires AVX-512). No lower/upper extraction is performed.
//
// Pair-comparison: each iteration loads 4 values as 2 pairs, sorts each
// pair with one comparison, then updates min from the smaller candidates
// and max from the larger candidates. This reduces branches from 8 to 6
// per 4-value chunk (2 pair sorts + 2 min updates + 2 max updates).
func analyzeUint64AVX2(values []uint64) (min64, max64, acc uint64) {
	n := len(values)
	if n < 4 {
		return analyzeUint64SSE2(values)
	}

	// Pointer loads: bounds checks in the 4-value loop dominate this pass.
	p := unsafe.Pointer(&values[0])

	accVec := archsimd.LoadUint64x4Array((*[4]uint64)(p))

	a0, b0 := values[0], values[1]
	if a0 > b0 {
		a0, b0 = b0, a0
	}
	a1, b1 := values[2], values[3]
	if a1 > b1 {
		a1, b1 = b1, a1
	}
	min64 = a0
	if a1 < min64 {
		min64 = a1
	}
	max64 = b0
	if b1 > max64 {
		max64 = b1
	}

	end := uintptr(n) * 8
	for off := uintptr(32); off+32 <= end; off += 32 {
		accVec = accVec.Or(archsimd.LoadUint64x4Array((*[4]uint64)(unsafe.Add(p, off))))
		// 32-bit pair sort keeps the min/max branch count at 6 per chunk.
		a0, b0 = *(*uint64)(unsafe.Add(p, off)), *(*uint64)(unsafe.Add(p, off+8))
		if a0 > b0 {
			a0, b0 = b0, a0
		}
		a1, b1 = *(*uint64)(unsafe.Add(p, off+16)), *(*uint64)(unsafe.Add(p, off+24))
		if a1 > b1 {
			a1, b1 = b1, a1
		}
		if a0 < min64 {
			min64 = a0
		}
		if a1 < min64 {
			min64 = a1
		}
		if b0 > max64 {
			max64 = b0
		}
		if b1 > max64 {
			max64 = b1
		}
	}

	accR := accVec.GetLo().Or(accVec.GetHi())
	var accL [2]uint64
	accR.StoreArray(&accL)
	acc = accL[0] | accL[1]

	tail := (n / 4) * 4
	for i := tail; i < n; i++ {
		v := values[i]
		acc |= v
		if v < min64 {
			min64 = v
		}
		if v > max64 {
			max64 = v
		}
	}
	return
}

// splitUint64OnlyAVX2 extracts lower/upper uint32 halves using VPERMD
// deinterleave (4 uint64 per iteration) without computing min/max or
// OR-accumulator. Used after analyzeUint64AVX2 has determined that the
// two-block path is needed.
func splitUint64OnlyAVX2(lower, upper []uint32, values []uint64) {
	n := len(values)
	if n < 4 {
		splitUint64Only(lower, upper, values)
		return
	}

	perm := archsimd.LoadUint32x8Array(&deinterleaveIdx)

	for i := 0; i+4 <= n; i += 4 {
		shuffled := archsimd.LoadUint64x4(values[i : i+4]).AsUint32x8().Permute(perm)
		shuffled.GetLo().Store(lower[i : i+4])
		shuffled.GetHi().Store(upper[i : i+4])
	}

	tail := (n / 4) * 4
	for i := tail; i < n; i++ {
		v := values[i]
		lower[i] = uint32(v)
		upper[i] = uint32(v >> 32)
	}
}

// allFitIn32BitsAVX2 checks if all uint64 values fit in 32 bits by
// OR-accumulating with Uint64x4 (4 values per iteration) and checking
// the upper 32 bits.
func allFitIn32BitsAVX2(values []uint64) bool {
	n := len(values)
	// Empty input trivially fits in 32 bits (vacuous OR-accumulator);
	// the scalar loop below already handles n in [0, 4).
	if n < 4 {
		var acc uint64
		for _, v := range values {
			acc |= v
		}
		return acc>>32 == 0
	}

	// Pointer loads avoid per-iteration bounds checks in the OR loop.
	p := unsafe.Pointer(&values[0])
	accVec := archsimd.LoadUint64x4Array((*[4]uint64)(p))

	end := uintptr(n) * 8
	for off := uintptr(32); off+32 <= end; off += 32 {
		accVec = accVec.Or(archsimd.LoadUint64x4Array((*[4]uint64)(unsafe.Add(p, off))))
	}

	reduced := accVec.GetLo().Or(accVec.GetHi())
	var lanes [2]uint64
	reduced.Store(lanes[:])
	acc := lanes[0] | lanes[1]

	tail := (n / 4) * 4
	for i := tail; i < n; i++ {
		acc |= values[i]
	}
	return acc>>32 == 0
}

// packBlockAVX2 packs uint32 values into a UTL block using AVX2-accelerated
// kernels for bit-packing, delta encoding, FOR selection, and bitwidth selection.
// Accepts configurable typeFlags and hasCombine for use by uint64 sub-block paths.
// When NoInPlace is set, the input values slice is not modified.
func packBlockAVX2(flag Flag, values []uint32, dst []byte, scratch []uint32, typeFlags uint32, hasCombine bool) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	noInPlace := flag&NoInPlace != 0
	if noInPlace && len(scratch) < ScratchLenNoInPlace {
		scratch = make([]uint32, ScratchLenNoInPlace)
	}

	headerFlags := typeFlags
	headerFlags |= uint32(flag&Special) << 14

	workValues := values
	workRedirected := false

	var useFOR bool
	var baseValue uint32
	var forW int
	if flag&NoFOR == 0 {
		useFOR, baseValue, forW = selectBitWidthWithFORAVX2(values)
		if useFOR {
			if noInPlace {
				workValues = scratch[:len(values)]
				forSubtractAVX2(workValues, values, baseValue)
				workRedirected = true
			} else {
				forSubtractAVX2(values, values, baseValue)
			}
			headerFlags |= uint32(forW) << forWidthShift
		}
	}

	if flag&Delta != 0 {
		if noInPlace && !workRedirected {
			workValues = scratch[:len(values)]
			copy(workValues, values)
		}
		var needZZ bool
		if len(workValues) == blockSize {
			needZZ = deltaEncodePerLaneAVX2(workValues)
		} else {
			needZZ = deltaEncodePerLaneScalar(workValues)
		}
		if needZZ {
			headerFlags |= headerZigZagFlag
		}
		headerFlags |= headerDeltaFlag
	}

	var bitWidth, excCount int
	if flag&NoPatch != 0 {
		bitWidth = selectBitWidthNoPatchAVX2(workValues)
	} else {
		bitWidth, excCount = selectBitWidthAVX2(workValues)
	}
	payloadSize := utlPayloadBytes(bitWidth)
	hasExceptions := excCount > 0
	forBBytes := forBaseBytes(forW)

	packInput := workValues
	if len(workValues) < blockSize {
		if noInPlace {
			// workValues already lives in scratch[0:len(values)]; extend and zero-pad.
			if !workRedirected && flag&Delta == 0 {
				copy(scratch[:len(values)], values)
			}
			clear(scratch[len(values):blockSize])
			packInput = scratch[:blockSize]
		} else {
			copy(scratch[:len(values)], values)
			clear(scratch[len(values):blockSize])
			packInput = scratch[:blockSize]
		}
	}

	if !hasExceptions {
		pOff := payloadOffset(forBBytes, false, hasCombine)
		totalLen := pOff + payloadSize
		dst = ensureLen(dst, totalLen)
		bo.PutUint32(dst, encodeHeader(len(values), bitWidth, 0, headerFlags))
		if useFOR {
			writeFORBase(dst, baseValue, forW, false)
		}
		packLanesUTLAVX2(dst[pOff:pOff+payloadSize], packInput, bitWidth)
		archsimd.ClearAVXUpperBits()
		return dst[:totalLen], nil
	}

	excIdxSize := excIndexSize(excCount)
	maxSvbLen := maxSVBEncodedLen(excCount)
	pOff := payloadOffset(forBBytes, true, hasCombine)
	maxTotalLen := pOff + payloadSize + excIdxSize + maxSvbLen

	dst = ensureLen(dst, maxTotalLen)
	bo.PutUint32(dst, encodeHeader(len(values), bitWidth, excCount, headerFlags))
	if useFOR {
		writeFORBase(dst, baseValue, forW, true)
	}

	packLanesUTLAVX2(dst[pOff:pOff+payloadSize], packInput, bitWidth)
	archsimd.ClearAVXUpperBits()

	var highBits []uint32
	if noInPlace {
		highBits = scratch[blockSize : 2*blockSize]
	} else {
		highBits = scratch[:blockSize]
	}
	excOff := pOff + payloadSize
	collectAndWriteExceptions(workValues, bitWidth, dst[excOff:], excCount, highBits)
	svbOffset := excOff + excIdxSize
	svbLen := encodeSVBIntoDst(dst[svbOffset:maxTotalLen], highBits[:excCount])

	bo.PutUint16(dst[headerBytes:], uint16(svbLen))
	return dst[:svbOffset+svbLen], nil
}

// packUint64AVX2 is the AVX2 implementation of PackUint64.
// Uses a fused analysis pass (SIMD OR-acc + scalar min/max) followed by
// a single path-specific data pass. This eliminates wasted VPERMD
// extraction when the FOR64 path is taken.
func packUint64AVX2(flag Flag, values []uint64, dst []byte, scratch []uint32) ([]byte, error) {
	count := len(values)
	if count == 0 || count > blockSize {
		return nil, ErrInvalidBuffer
	}

	off := len(dst) * int((flag&Append)>>4)
	innerFlag := flag &^ Append

	dst = ensureCapacity64(dst, off, innerFlag)

	min64, max64, acc := analyzeUint64AVX2(values)

	if innerFlag&NoFOR == 0 && min64 > 0 && (max64-min64) < (1<<32) {
		forSubtract64AVX2(scratch[:count], values, min64)
		result, err := packFor64(innerFlag, dst, scratch, off, count, min64, packBlockAVX2)
		archsimd.ClearAVXUpperBits()
		return result, err
	}

	if acc>>32 == 0 {
		narrowToUint32AVX2(scratch, values, count)
		block, err := packBlockAVX2(innerFlag, scratch[:count], dst[off:off], scratch[blockSize:2*blockSize], headerTypeUint64Flag, false)
		if err != nil {
			return nil, err
		}
		archsimd.ClearAVXUpperBits()
		return dst[:off+len(block)], nil
	}

	upper := scratch[2*blockSize:]
	splitUint64OnlyAVX2(scratch, upper, values)
	result, err := packUint64TwoBlock(innerFlag, dst, scratch, off, count, packBlockAVX2)
	archsimd.ClearAVXUpperBits()
	return result, err
}

// unpackBlockAVX2 unpacks a uint32 block using AVX2-accelerated kernels.
// Accepts forUint64 to handle uint64 sub-block context (type validation,
// FOR64 single-block handling, combine flag recognition).
func unpackBlockAVX2(dst []uint32, scratch []uint32, buf []byte, forUint64 bool) ([]uint32, int, error) {
	// Shared prologue written through a pointer: no fat struct return on
	// the hot path, and the flag byte replaces six bool fields.
	var h unpackHeaderOut
	buf, err := decodeUnpackHeader(buf, forUint64, &h)
	if err != nil {
		return nil, 0, err
	}
	// Empty block: header consumed, no values.
	if h.empty() {
		return dst[:0], headerBytes, nil
	}

	dst = growUnpackDst(dst)

	unpackLanesUTLAVX2(dst, h.payloadAt(buf), blockSize, h.bitWidth)

	dst = dst[:h.count]

	consumed := h.consumed

	if h.hasExceptions() {
		archsimd.ClearAVXUpperBits()
		// The exception region starts right after the payload, which is
		// exactly the consumed offset (h.consumed = pOff + payloadBytes).
		var err error
		consumed, err = applyExceptions(dst, buf, h.consumed, h.count, h.bitWidth, h.excCount, scratch)
		if err != nil {
			return nil, 0, err
		}
	}

	if h.hasDelta() {
		if h.hasZigZag() {
			// Zigzag mode cannot overflow: SIMD kernel for full blocks.
			if h.count == blockSize {
				deltaDecodePerLaneAVX2(dst, true)
			} else {
				archsimd.ClearAVXUpperBits()
				deltaDecodePerLaneScalar(dst, true)
			}
		} else {
			// Explicit dispatch (no func values): full blocks run the AVX2
			// overflow kernel, partial blocks the scalar reference.
			var overflowPos int
			if h.count == blockSize {
				overflowPos = deltaDecodePerLaneWithOverflowAVX2(dst, false)
			} else {
				archsimd.ClearAVXUpperBits()
				overflowPos = deltaDecodePerLaneWithOverflowScalar(dst, false)
			}
			// Wrap a detected overflow into the documented error type.
			if overflowPos > 0 {
				archsimd.ClearAVXUpperBits()
				return nil, 0, &ErrOverflow{Position: overflowPos}
			}
		}
	}

	if h.hasFOR() && !h.u64Single() {
		forAddAVX2(dst, h.count, h.forBase)
	}

	archsimd.ClearAVXUpperBits()
	return dst, consumed, nil
}

// combineUint64AVX2 merges lower and upper uint32 halves into uint64 values
// using AVX2 VPMOVZXDQ (ExtendToUint64) + ShiftAllLeft + Or (4 values per iteration).
func combineUint64AVX2(dst []uint64, lower, upper []uint32, count int) {
	// Pointer loads/stores eliminate per-iteration bounds checks; the
	// count check skips pointer setup for empty slices (corrupt input).
	if count >= 4 {
		dp := unsafe.Pointer(&dst[0])
		lp := unsafe.Pointer(&lower[0])
		up := unsafe.Pointer(&upper[0])
		// off iterates in output bytes (8 bytes/uint64 x 4 values = 32 per step).
		// Input arrays are uint32, so their byte offset is off/2
		// (4 bytes/uint32 x 4 values = 16 per step).
		end := uintptr(count) * 8
		for off := uintptr(0); off+32 <= end; off += 32 {
			loVec := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(lp, off/2))).ExtendToUint64()
			hiVec := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(up, off/2))).ExtendToUint64()
			loVec.Or(hiVec.ShiftAllLeft(32)).StoreArray((*[4]uint64)(unsafe.Add(dp, off)))
		}
	}
	for i := (count / 4) * 4; i < count; i++ {
		dst[i] = uint64(upper[i])<<32 | uint64(lower[i])
	}
}

// forAdd64AVX2 adds a uint64 base to each uint32 value and stores as uint64.
// Uses AVX2 VPMOVZXDQ (ExtendToUint64) + Add (4 values per iteration).
func forAdd64AVX2(dst []uint64, values []uint32, base uint64, count int) {
	// Broadcast hoisted; pointer loads eliminate bounds checks.
	baseVec := archsimd.BroadcastUint64x4(base)
	// Corrupt-input paths can pass count == 0 with an empty dst slice;
	// skip pointer setup in that case (adding to 0 elements is a no-op).
	if count >= 4 {
		dp := unsafe.Pointer(&dst[0])
		vp := unsafe.Pointer(&values[0])
		// off iterates in output bytes (uint64); input is uint32, so off/2.
		end := uintptr(count) * 8
		for off := uintptr(0); off+32 <= end; off += 32 {
			wide := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(vp, off/2))).ExtendToUint64().Add(baseVec)
			wide.StoreArray((*[4]uint64)(unsafe.Add(dp, off)))
		}
	}
	for i := (count / 4) * 4; i < count; i++ {
		dst[i] = uint64(values[i]) + base
	}
}

// narrowToUint32AVX2 copies uint64 values to uint32 by truncation using
// VPERMD deinterleave (4 values per iteration). Extracts the lower dword
// of each uint64 by shuffling [lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3] ->
// [lo0,lo1,lo2,lo3,...] and storing the low 128-bit lane.
func narrowToUint32AVX2(dst []uint32, values []uint64, count int) {
	// Deinterleave index loaded once; pointer loads avoid bounds checks.
	// The count check skips pointer setup for empty slices (corrupt input).
	perm := archsimd.LoadUint32x8Array(&deinterleaveIdx)
	if count >= 4 {
		dp := unsafe.Pointer(&dst[0])
		vp := unsafe.Pointer(&values[0])
		// off iterates in input bytes (uint64); output is uint32, so off/2.
		end := uintptr(count) * 8
		for off := uintptr(0); off+32 <= end; off += 32 {
			archsimd.LoadUint64x4Array((*[4]uint64)(unsafe.Add(vp, off))).
				AsUint32x8().Permute(perm).GetLo().StoreArray((*[4]uint32)(unsafe.Add(dp, off/2)))
		}
	}
	for i := (count / 4) * 4; i < count; i++ {
		dst[i] = uint32(values[i])
	}
}

// forSubtract64AVX2 subtracts base from each uint64 value and stores as uint32.
// Uses Uint64x4 Sub + VPERMD truncation (4 values per iteration).
func forSubtract64AVX2(dst []uint32, values []uint64, base uint64) {
	// Broadcast + shuffle index hoisted; pointer loads avoid bounds checks.
	baseVec := archsimd.BroadcastUint64x4(base)
	perm := archsimd.LoadUint32x8Array(&deinterleaveIdx)
	n := len(values)
	if n >= 4 {
		dp := unsafe.Pointer(&dst[0])
		vp := unsafe.Pointer(&values[0])
		// off iterates in input bytes (uint64); output is uint32, so off/2.
		end := uintptr(n) * 8
		for off := uintptr(0); off+32 <= end; off += 32 {
			diff := archsimd.LoadUint64x4Array((*[4]uint64)(unsafe.Add(vp, off))).Sub(baseVec)
			diff.AsUint32x8().Permute(perm).GetLo().StoreArray((*[4]uint32)(unsafe.Add(dp, off/2)))
		}
	}
	for i := (n / 4) * 4; i < n; i++ {
		dst[i] = uint32(values[i] - base)
	}
}

// unpackUint64AVX2 is the AVX2 implementation of UnpackUint64.
// Uses AVX2 SIMD for combine and forAdd64 operations.
func unpackUint64AVX2(buf []byte, dst []uint64, scratch []uint32) ([]uint64, int, error) {
	dst, consumed, err := unpackUint64Block(buf, dst, scratch, unpackBlockAVX2, forAdd64AVX2, combineUint64AVX2)
	archsimd.ClearAVXUpperBits()
	return dst, consumed, err
}
