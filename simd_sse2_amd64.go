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
	for v := range utlValuesPerLane {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		for group := range 4 {
			off := base + group*16
			vec := archsimd.LoadUint32x4Array(
				(*[4]uint32)(unsafe.Pointer(&payload[off])))
			result := vec.ShiftAllRight(shift).And(mask)

			if int(shift)+bitWidth > 32 {
				nextOff := (wordIdx+1)*utlSuperWordBytes + group*16
				nextVec := archsimd.LoadUint32x4Array(
					(*[4]uint32)(unsafe.Pointer(&payload[nextOff])))
				result = result.Or(
					nextVec.ShiftAllLeft(uint64(32) - shift).And(mask))
			}

			outBase := v*utlLaneCount + group*4
			result.Store(dst[outBase : outBase+4])
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
		for v := range utlValuesPerLane {
			for group := range 4 {
				inBase := v*utlLaneCount + group*4
				off := v*utlSuperWordBytes + group*16
				archsimd.LoadUint32x4(values[inBase : inBase+4]).StoreArray(
					(*[4]uint32)(unsafe.Pointer(&dst[off])))
			}
		}
		return
	}

	mask := archsimd.BroadcastUint32x4(uint32((1 << bitWidth) - 1))

	clear(dst)

	bitOffset := 0
	for v := range utlValuesPerLane {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		for group := range 4 {
			inBase := v*utlLaneCount + group*4
			val := archsimd.LoadUint32x4(values[inBase : inBase+4]).And(mask)

			off := base + group*16
			cur := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&dst[off])))
			cur.Or(val.ShiftAllLeft(shift)).StoreArray((*[4]uint32)(unsafe.Pointer(&dst[off])))

			if int(shift)+bitWidth > 32 {
				rightShift := uint64(32) - shift
				nextOff := (wordIdx+1)*utlSuperWordBytes + group*16
				next := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Pointer(&dst[nextOff])))
				next.Or(val.ShiftAllRight(rightShift)).StoreArray((*[4]uint32)(unsafe.Pointer(&dst[nextOff])))
			}
		}

		bitOffset += bitWidth
	}
}

// zigzagEncodeSSE2 applies zigzag encoding using SSE2 (Uint32x4).
// Formula: (n << 1) ^ (n >> 31) with arithmetic right shift.
func zigzagEncodeSSE2(buf []uint32, n int) {
	// Pointer-based loads/stores: eliminates per-iteration bounds checks.
	if n >= 4 {
		p := unsafe.Pointer(&buf[0])
		end := uintptr(n) * 4
		for off := uintptr(0); off+16 <= end; off += 16 {
			v := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(p, off)))
			shifted := v.ShiftAllLeft(1)
			sign := v.AsInt32x4().ShiftAllRight(31).AsUint32x4()
			shifted.Xor(sign).StoreArray((*[4]uint32)(unsafe.Add(p, off)))
		}
	}
	// Scalar tail for n < 4 or leftover values.
	for i := (n / 4) * 4; i < n; i++ {
		buf[i] = zigzagEncode32(int32(buf[i]))
	}
}

// zigzagDecodeSSE2 applies zigzag decoding in-place using SSE2 (Uint32x4).
// Formula: (n >>> 1) ^ -(n & 1).
func zigzagDecodeSSE2(values []uint32) {
	// Broadcast constants hoisted out of the loop; pointer loads avoid
	// per-iteration bounds checks.
	one := archsimd.BroadcastUint32x4(1)
	zero := archsimd.BroadcastUint32x4(0)
	n := len(values)
	if n >= 4 {
		p := unsafe.Pointer(&values[0])
		end := uintptr(n) * 4
		for off := uintptr(0); off+16 <= end; off += 16 {
			v := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(p, off)))
			half := v.ShiftAllRight(1)   // n >>> 1 (logical right shift)
			signBit := v.And(one)        // n & 1: extract sign from LSB
			negSign := zero.Sub(signBit) // 0 - signBit: yields 0x00000000 or 0xFFFFFFFF
			half.Xor(negSign).StoreArray((*[4]uint32)(unsafe.Add(p, off)))
		}
	}
	// Scalar tail for n < 4 or leftover values.
	for i := (n / 4) * 4; i < n; i++ {
		values[i] = uint32(zigzagDecode32(values[i]))
	}
}

// deltaEncodePerLaneSSE2 computes per-lane deltas using SSE2 (Uint32x4).
// Processes all 16 lanes in four register groups.
// Returns true if zigzag encoding was needed (negative deltas detected).
// Borrow detection is batched to eliminate branches from the inner loop.
func deltaEncodePerLaneSSE2(values []uint32) bool {
	var anyBorrow uint8

	for v := utlValuesPerLane - 1; v > 0; v-- {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		// TODO-PERF: Maybe unroll
		for group := range 4 {
			off := group * 4
			cur := archsimd.LoadUint32x4(values[curBase+off : curBase+off+4])
			prev := archsimd.LoadUint32x4(values[prevBase+off : prevBase+off+4])
			delta := cur.Sub(prev)

			anyBorrow |= prev.Greater(cur).ToBits()

			delta.Store(values[curBase+off : curBase+off+4])
		}
	}

	needZigZag := anyBorrow != 0
	if needZigZag {
		zigzagEncodeSSE2(values, len(values))
	}
	return needZigZag
}

// deltaDecodePerLaneSSE2 performs in-place per-lane prefix sums using SSE2 (Uint32x4).
func deltaDecodePerLaneSSE2(values []uint32, useZigZag bool) {
	if useZigZag {
		zigzagDecodeSSE2(values)
	}

	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		// TODO-PERF: Maybe unroll
		for group := range 4 {
			off := group * 4
			prev := archsimd.LoadUint32x4(values[prevBase+off : prevBase+off+4])
			cur := archsimd.LoadUint32x4(values[curBase+off : curBase+off+4])
			sum := prev.Add(cur)
			sum.Store(values[curBase+off : curBase+off+4])
		}
	}
}

// deltaDecodePerLaneWithOverflowSSE2 performs per-lane prefix sum (delta
// decode) using SSE2 XMM registers (Uint32x4) and detects unsigned overflow.
//
// Returns the flat index (row*16 + lane) of the first addition that wrapped
// past 2^32, or 0 if no overflow. Same v-major scan order as the scalar
// reference and the AVX2 variant (see deltaDecodePerLaneWithOverflowAVX2
// for the mask-collect design rationale).
//
// SSE2 processes 4 lanes per vector, so each row requires 4 vectors (two
// groups of two). The per-row mask is a uint16 assembled from four 4-bit
// ToBits results - NOT a uint8, because 16 lanes need 16 mask bits (a
// uint8 would silently lose overflows in lanes 8-15).
func deltaDecodePerLaneWithOverflowSSE2(values []uint32, useZigZag bool) int {
	if useZigZag {
		deltaDecodePerLaneSSE2(values, true)
		return 0
	}

	// masks[v] holds the overflow mask of row v: bit 0-3 from vector 0
	// (lanes 0-3), bits 4-7 from vector 1 (lanes 4-7), bits 8-11 from
	// group-1 vector 0 (lanes 8-11), bits 12-15 from group-1 vector 1
	// (lanes 12-15). Row 0 has no previous element, so v starts at 1 and
	// masks has 7 entries. uint16 is required: 16 lanes need 16 mask bits
	// (a uint8 would silently drop the lane 8-15 bits).
	var masks [utlValuesPerLane - 1]uint16

	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		m := uint16(0)
		for group := range 2 {
			off := group * 8
			prev0 := archsimd.LoadUint32x4(values[prevBase+off : prevBase+off+4])
			cur0 := archsimd.LoadUint32x4(values[curBase+off : curBase+off+4])
			sum0 := prev0.Add(cur0)
			// Sum < prev means the unsigned addition wrapped (overflow).
			// Each group covers 8 lanes: bits group*8+0..3 from vector 0,
			// bits group*8+4..7 from vector 1.
			m |= uint16(sum0.Less(prev0).ToBits()) << (group * 8)

			prev1 := archsimd.LoadUint32x4(values[prevBase+off+4 : prevBase+off+8])
			cur1 := archsimd.LoadUint32x4(values[curBase+off+4 : curBase+off+8])
			sum1 := prev1.Add(cur1)
			m |= uint16(sum1.Less(prev1).ToBits()) << (group*8 + 4)

			sum0.Store(values[curBase+off : curBase+off+4])
			sum1.Store(values[curBase+off+4 : curBase+off+8])
		}
		// Collect the mask; at most once per row (see AVX2 comment).
		if m != 0 {
			masks[v-1] = m
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

// packUint32SSE2 is the SSE2 packing pipeline. Delegates to packBlockSSE2
// with uint32 type flags and Append handling.
func packUint32SSE2(flag Flag, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}
	if flag&Append == 0 {
		return packBlockSSE2(flag, values, dst, scratch, headerTypeUint32Flag, false)
	}
	off := len(dst)
	dst = ensureCapacity32(dst, off)
	block, err := packBlockSSE2(flag&^Append, values, dst[off:off], scratch, headerTypeUint32Flag, false)
	if err != nil {
		return nil, err
	}
	return dst[:off+len(block)], nil
}

// unpackUint32SSE2 is the SSE2 unpacking pipeline. Delegates to
// unpackBlockSSE2 with uint32 context (forUint64=false).
func unpackUint32SSE2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackBlockSSE2(dst, scratch, buf, false)
}

// unpackBlockSSE2 with uint64 sub-block context (forUint64=true).
// Used by GetUint64 for SIMD-accelerated delta full-unpack fallback.
func unpackBlockForUint64SSE2(dst, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackBlockSSE2(dst, scratch, buf, true)
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

// findMinMaxSSE2 computes min/max using SSE2 4-wide operations.
// Uses pointer-based loads to avoid per-iteration slice bounds checking.
func findMinMaxSSE2(values []uint32) (uint32, uint32) {
	n := len(values)
	if n < 4 {
		return findMinMaxScalar(values)
	}

	// Seed both SIMD accumulators from the first 4-value chunk.
	p := unsafe.Pointer(&values[0])
	minVec := archsimd.LoadUint32x4Array((*[4]uint32)(p))
	maxVec := minVec

	// Scan full SSE2-width chunks and keep running lane-wise min/max.
	end := uintptr(n) * 4
	for off := uintptr(16); off+16 <= end; off += 16 {
		chunk := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(p, off)))
		minVec = minVec.Min(chunk)
		maxVec = maxVec.Max(chunk)
	}

	// Collapse the vector accumulators into scalar candidates.
	var minL, maxL [4]uint32
	minVec.StoreArray(&minL)
	maxVec.StoreArray(&maxL)
	minResult := min(min(minL[0], minL[1]), min(minL[2], minL[3]))
	maxResult := max(max(maxL[0], maxL[1]), max(maxL[2], maxL[3]))

	// Finish any remaining elements that did not fill a full SIMD chunk.
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

// selectBitWidthNoPatchSSE2 computes the minimum step bitwidth using SSE2
// OR-reduction. No exception analysis is performed.
func selectBitWidthNoPatchSSE2(values []uint32) int {
	// Pointer-based loads: eliminates the per-iteration slice bounds check.
	orVec := archsimd.BroadcastUint32x4(0)
	n := len(values)
	if n >= 4 {
		p := unsafe.Pointer(&values[0])
		end := uintptr(n) * 4
		for off := uintptr(0); off+16 <= end; off += 16 {
			orVec = orVec.Or(archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(p, off))))
		}
	}
	var lanes [4]uint32
	orVec.StoreArray(&lanes)
	var orAll uint32
	for _, v := range lanes {
		orAll |= v
	}
	// Scalar tail for n < 4 or leftover values after the last full vector.
	for i := (n / 4) * 4; i < n; i++ {
		orAll |= values[i]
	}
	return roundUpToStep(bits.Len32(orAll))
}

// forSubtractSSE2 subtracts baseValue from each element using SSE2.
func forSubtractSSE2(dst, src []uint32, baseValue uint32) {
	// Broadcast once outside the loop; pointer loads avoid bounds checks.
	baseVec := archsimd.BroadcastUint32x4(baseValue)
	n := len(src)
	if n >= 4 {
		sp := unsafe.Pointer(&src[0])
		dp := unsafe.Pointer(&dst[0])
		end := uintptr(n) * 4
		for off := uintptr(0); off+16 <= end; off += 16 {
			archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(sp, off))).
				Sub(baseVec).
				StoreArray((*[4]uint32)(unsafe.Add(dp, off)))
		}
	}
	// Scalar tail for n < 4 or leftover values.
	for i := (n / 4) * 4; i < n; i++ {
		dst[i] = src[i] - baseValue
	}
}

// forAddSSE2 adds baseValue to each of the first count elements using SSE2.
func forAddSSE2(output []uint32, count int, baseValue uint32) {
	// Broadcast once outside the loop; pointer loads avoid bounds checks.
	baseVec := archsimd.BroadcastUint32x4(baseValue)
	if count >= 4 {
		p := unsafe.Pointer(&output[0])
		end := uintptr(count) * 4
		for off := uintptr(0); off+16 <= end; off += 16 {
			archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(p, off))).
				Add(baseVec).
				StoreArray((*[4]uint32)(unsafe.Add(p, off)))
		}
	}
	// Scalar tail for count < 4 or leftover elements.
	for i := (count / 4) * 4; i < count; i++ {
		output[i] += baseValue
	}
}

// analyzeUint64SSE2 computes OR-accumulator and min/max of uint64 values
// in a single fused pass. SIMD Uint64x2 handles OR-accumulation while
// scalar pair-comparison handles min/max (VPMINUQ requires AVX-512).
// No lower/upper extraction is performed, avoiding wasted writes when
// the FOR64 or all-fit-32 path is taken.
//
// Pair-comparison: each iteration loads 2 values, sorts the pair with
// one comparison, then updates min from the smaller and max from the
// larger candidate. This reduces branches from 4 to 3 per pair.
func analyzeUint64SSE2(values []uint64) (min64, max64, acc uint64) {
	n := len(values)
	if n < 2 {
		return analyzeUint64(values)
	}

	// Pointer loads: bounds checks in the 2-value loop dominate this pass.
	p := unsafe.Pointer(&values[0])
	accVec := archsimd.LoadUint64x2Array((*[2]uint64)(p))
	a, b := values[0], values[1]
	if a > b {
		a, b = b, a
	}
	min64, max64 = a, b

	end := uintptr(n) * 8
	for off := uintptr(16); off+16 <= end; off += 16 {
		accVec = accVec.Or(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Add(p, off))))
		a, b = *(*uint64)(unsafe.Add(p, off)), *(*uint64)(unsafe.Add(p, off+8))
		if a > b {
			a, b = b, a
		}
		if a < min64 {
			min64 = a
		}
		if b > max64 {
			max64 = b
		}
	}

	var accL [2]uint64
	accVec.StoreArray(&accL)
	acc = accL[0] | accL[1]

	if n%2 != 0 {
		v := values[n-1]
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

// allFitIn32BitsSSE2 checks if all uint64 values fit in 32 bits by
// OR-accumulating and testing the upper 32 bits. Processes 2 values per
// iteration using Uint64x2.
func allFitIn32BitsSSE2(values []uint64) bool {
	n := len(values)
	// Empty input trivially fits in 32 bits (vacuous OR-accumulator).
	if n < 2 {
		if n == 0 {
			return true
		}
		return values[0]>>32 == 0
	}

	// Pointer loads avoid per-iteration bounds checks in the OR loop.
	p := unsafe.Pointer(&values[0])
	accVec := archsimd.LoadUint64x2Array((*[2]uint64)(p))

	end := uintptr(n) * 8
	for off := uintptr(16); off+16 <= end; off += 16 {
		accVec = accVec.Or(archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Add(p, off))))
	}

	var lanes [2]uint64
	accVec.Store(lanes[:])
	acc := lanes[0] | lanes[1]

	tail := (n / 2) * 2
	for i := tail; i < n; i++ {
		acc |= values[i]
	}
	return acc>>32 == 0
}

// packBlockSSE2 packs uint32 values into a UTL block using SSE2-accelerated
// kernels for bit-packing, delta encoding, FOR selection, and bitwidth selection.
// Accepts configurable typeFlags and hasCombine for use by uint64 sub-block paths.
// When NoInPlace is set, the input values slice is not modified.
func packBlockSSE2(flag Flag, values []uint32, dst []byte, scratch []uint32, typeFlags uint32, hasCombine bool) ([]byte, error) {
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
		useFOR, baseValue, forW = selectBitWidthWithFORSSE2(values)
		if useFOR {
			if noInPlace {
				workValues = scratch[:len(values)]
				forSubtractSSE2(workValues, values, baseValue)
				workRedirected = true
			} else {
				forSubtractSSE2(values, values, baseValue)
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
			needZZ = deltaEncodePerLaneSSE2(workValues)
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
		bitWidth = selectBitWidthNoPatchSSE2(workValues)
	} else {
		bitWidth, excCount = selectBitWidthSSE2(workValues)
	}
	payloadSize := utlPayloadBytes(bitWidth)
	hasExceptions := excCount > 0
	forBBytes := forBaseBytes(forW)

	packInput := workValues
	if len(workValues) < blockSize {
		if noInPlace {
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
		packLanesUTLSSE2(dst[pOff:pOff+payloadSize], packInput, bitWidth)
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

	packLanesUTLSSE2(dst[pOff:pOff+payloadSize], packInput, bitWidth)

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

// packUint64SSE2 is the SSE2 implementation of PackUint64.
// Uses a fused analysis pass (SIMD OR-acc + scalar min/max) followed by
// a single path-specific data pass. This eliminates wasted lower/upper
// extraction when the FOR64 path is taken.
func packUint64SSE2(flag Flag, values []uint64, dst []byte, scratch []uint32) ([]byte, error) {
	count := len(values)
	if count == 0 || count > blockSize {
		return nil, ErrInvalidBuffer
	}

	off := len(dst) * int((flag&Append)>>4)
	innerFlag := flag &^ Append

	dst = ensureCapacity64(dst, off, innerFlag)

	min64, max64, acc := analyzeUint64SSE2(values)

	if innerFlag&NoFOR == 0 && min64 > 0 && (max64-min64) < (1<<32) {
		forSubtract64SSE2(scratch[:count], values, min64)
		return packFor64(innerFlag, dst, scratch, off, count, min64, packBlockSSE2)
	}

	if acc>>32 == 0 {
		narrowToUint32(scratch, values, count)
		block, err := packBlockSSE2(innerFlag, scratch[:count], dst[off:off], scratch[blockSize:2*blockSize], headerTypeUint64Flag, false)
		if err != nil {
			return nil, err
		}
		return dst[:off+len(block)], nil
	}

	upper := scratch[2*blockSize:]
	splitUint64Only(scratch, upper, values)
	return packUint64TwoBlock(innerFlag, dst, scratch, off, count, packBlockSSE2)
}

// unpackBlockSSE2 unpacks a uint32 block using SSE2-accelerated kernels.
// Accepts forUint64 to handle uint64 sub-block context (type validation,
// FOR64 single-block handling, combine flag recognition).
func unpackBlockSSE2(dst []uint32, scratch []uint32, buf []byte, forUint64 bool) ([]uint32, int, error) {
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

	unpackLanesUTLSSE2(dst, h.payloadAt(buf), blockSize, h.bitWidth)
	dst = dst[:h.count]

	consumed := h.consumed

	if h.hasExceptions() {
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
				deltaDecodePerLaneSSE2(dst, true)
			} else {
				deltaDecodePerLaneScalar(dst, true)
			}
		} else {
			// Explicit dispatch (no func values): full blocks run the SSE2
			// overflow kernel, partial blocks the scalar reference.
			var overflowPos int
			if h.count == blockSize {
				overflowPos = deltaDecodePerLaneWithOverflowSSE2(dst, false)
			} else {
				overflowPos = deltaDecodePerLaneWithOverflowScalar(dst, false)
			}
			// Wrap a detected overflow into the documented error type.
			if overflowPos > 0 {
				return nil, 0, &ErrOverflow{Position: overflowPos}
			}
		}
	}

	if h.hasFOR() && !h.u64Single() {
		forAddSSE2(dst, h.count, h.forBase)
	}

	return dst, consumed, nil
}

// combineUint64SSE2 merges lower and upper uint32 halves into uint64 values
// using SSE2 InterleaveLo/InterleaveHi (4 values per iteration).
func combineUint64SSE2(dst []uint64, lower, upper []uint32, count int) {
	// Pointer loads/stores eliminate per-iteration bounds checks; the
	// count check skips pointer setup for empty slices (corrupt input).
	if count >= 4 {
		dp := unsafe.Pointer(&dst[0])
		lp := unsafe.Pointer(&lower[0])
		up := unsafe.Pointer(&upper[0])
		// off iterates in output bytes (8 bytes/uint64 × 4 values = 32 per step).
		// Input arrays are uint32, so their byte offset is off/2
		// (4 bytes/uint32 × 4 values = 16 per step).
		end := uintptr(count) * 8
		for off := uintptr(0); off+32 <= end; off += 32 {
			lo := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(lp, off/2)))
			hi := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(up, off/2)))
			lo.InterleaveLo(hi).AsUint64x2().StoreArray((*[2]uint64)(unsafe.Add(dp, off)))
			lo.InterleaveHi(hi).AsUint64x2().StoreArray((*[2]uint64)(unsafe.Add(dp, off+16)))
		}
	}
	for i := (count / 4) * 4; i < count; i++ {
		dst[i] = uint64(upper[i])<<32 | uint64(lower[i])
	}
}

// forAdd64SSE2 adds a uint64 base to each uint32 value and stores as uint64.
// Uses InterleaveLo/Hi with zero vector for zero-extension (4 values per iteration).
// SSE2 lacks VPMOVZXDQ, so zero-extension is emulated by interleaving with
// a zero vector: InterleaveLo(vals, zero) produces [v0, 0, v1, 0] as uint64x2.
func forAdd64SSE2(dst []uint64, values []uint32, base uint64, count int) {
	// Broadcasts hoisted; pointer loads avoid bounds checks; the count
	// check skips pointer setup for empty slices (corrupt input).
	baseVec := archsimd.BroadcastUint64x2(base)
	zero := archsimd.BroadcastUint32x4(0)
	if count >= 4 {
		dp := unsafe.Pointer(&dst[0])
		vp := unsafe.Pointer(&values[0])
		// off iterates in output bytes (uint64); input is uint32, so off/2.
		end := uintptr(count) * 8
		for off := uintptr(0); off+32 <= end; off += 32 {
			vals := archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(vp, off/2)))
			vals.InterleaveLo(zero).AsUint64x2().Add(baseVec).StoreArray((*[2]uint64)(unsafe.Add(dp, off)))
			vals.InterleaveHi(zero).AsUint64x2().Add(baseVec).StoreArray((*[2]uint64)(unsafe.Add(dp, off+16)))
		}
	}
	for i := (count / 4) * 4; i < count; i++ {
		dst[i] = uint64(values[i]) + base
	}
}

// forSubtract64SSE2 subtracts base from each uint64 value and stores as uint32.
// Uses Uint64x2 Sub (2 values per iteration). SSE2 lacks a direct 64->32
// truncation instruction (VPMOVQD is AVX-512), so the results are extracted
// via a temporary array and stored as scalar uint32.
func forSubtract64SSE2(dst []uint32, values []uint64, base uint64) {
	// Broadcast hoisted; pointer loads avoid bounds checks.
	// The n >= 2 guard skips pointer setup for empty/single-element slices
	// (defense-in-depth; callers currently guarantee count > 0).
	baseVec := archsimd.BroadcastUint64x2(base)
	n := len(values)
	if n >= 2 {
		vp := unsafe.Pointer(&values[0])
		// off iterates in input bytes (uint64); off/8 converts to element index
		// for the uint32 destination (each uint64 yields one truncated uint32).
		end := uintptr(n) * 8
		for off := uintptr(0); off+16 <= end; off += 16 {
			diff := archsimd.LoadUint64x2Array((*[2]uint64)(unsafe.Add(vp, off))).Sub(baseVec)
			var tmp [2]uint64
			diff.StoreArray(&tmp)
			dst[off/8] = uint32(tmp[0])
			dst[(off/8)+1] = uint32(tmp[1])
		}
	}
	for i := (n / 2) * 2; i < n; i++ {
		dst[i] = uint32(values[i] - base)
	}
}

// unpackUint64SSE2 is the SSE2 implementation of UnpackUint64.
// Uses SSE2 SIMD for combine and forAdd64 operations.
func unpackUint64SSE2(buf []byte, dst []uint64, scratch []uint32) ([]uint64, int, error) {
	return unpackUint64Block(buf, dst, scratch, unpackBlockSSE2, forAdd64SSE2, combineUint64SSE2)
}
