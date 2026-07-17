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
	for v := range utlValuesPerLane {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		vec := archsimd.LoadUint32x16Array(
			(*[16]uint32)(unsafe.Pointer(&payload[base])))
		result := vec.ShiftAllRight(shift).And(mask)

		if int(shift)+bitWidth > 32 {
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			nextVec := archsimd.LoadUint32x16Array(
				(*[16]uint32)(unsafe.Pointer(&payload[nextBase])))
			result = result.Or(
				nextVec.ShiftAllLeft(uint64(32) - shift).And(mask))
		}

		outBase := v * utlLaneCount
		result.Store(dst[outBase : outBase+16])
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
		for v := range utlValuesPerLane {
			inBase := v * utlLaneCount
			base := v * utlSuperWordBytes
			archsimd.LoadUint32x16(values[inBase : inBase+16]).StoreArray(
				(*[16]uint32)(unsafe.Pointer(&dst[base])))
		}
		return
	}

	mask := archsimd.BroadcastUint32x16(uint32((1 << bitWidth) - 1))

	clear(dst)

	bitOffset := 0
	for v := range utlValuesPerLane {
		inBase := v * utlLaneCount
		val := archsimd.LoadUint32x16(values[inBase : inBase+16]).And(mask)

		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		cur := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(&dst[base])))
		cur.Or(val.ShiftAllLeft(shift)).StoreArray((*[16]uint32)(unsafe.Pointer(&dst[base])))

		if int(shift)+bitWidth > 32 {
			rightShift := uint64(32) - shift
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			next := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(&dst[nextBase])))
			next.Or(val.ShiftAllRight(rightShift)).StoreArray((*[16]uint32)(unsafe.Pointer(&dst[nextBase])))
		}

		bitOffset += bitWidth
	}
}

// packUint32AVX512 is the AVX-512 packing pipeline. Delegates to
// packBlockAVX512 with uint32 type flags and Append handling.
func packUint32AVX512(flag Flag, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}
	if flag&Append == 0 {
		return packBlockAVX512(flag, values, dst, scratch, headerTypeUint32Flag, false)
	}
	off := len(dst)
	dst = ensureCapacity32(dst, off)
	block, err := packBlockAVX512(flag&^Append, values, dst[off:off], scratch, headerTypeUint32Flag, false)
	if err != nil {
		return nil, err
	}
	return dst[:off+len(block)], nil
}

// unpackUint32AVX512 is the AVX-512 unpacking pipeline. Delegates to
// unpackBlockAVX512 with uint32 context (forUint64=false).
func unpackUint32AVX512(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackBlockAVX512(dst, scratch, buf, false)
}

// unpackBlockAVX512 with uint64 sub-block context (forUint64=true).
// Used by GetUint64 for SIMD-accelerated delta full-unpack fallback.
func unpackBlockForUint64AVX512(dst, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackBlockAVX512(dst, scratch, buf, true)
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

// selectBitWidthWithFORAVX512 fuses min/max computation with all 8 threshold
// comparisons in a single pass. AVX-512 has 32 ZMM registers, so 8 thresholds
// + 2 min/max + 1 value = 11 persistent ZMM registers fit easily.
func selectBitWidthWithFORAVX512(values []uint32) (useFOR bool, baseValue uint32, forWidth int) {
	n := len(values)

	var exc [9]int
	var minVal, maxVal uint32

	if n >= 16 {
		p := unsafe.Pointer(&values[0])
		t0 := archsimd.BroadcastUint32x16(0)
		t1 := archsimd.BroadcastUint32x16(0xF)
		t2 := archsimd.BroadcastUint32x16(0xFF)
		t3 := archsimd.BroadcastUint32x16(0xFFF)
		t4 := archsimd.BroadcastUint32x16(0xFFFF)
		t5 := archsimd.BroadcastUint32x16(0xFFFFF)
		t6 := archsimd.BroadcastUint32x16(0xFFFFFF)
		t7 := archsimd.BroadcastUint32x16(0xFFFFFFF)
		minVec := archsimd.LoadUint32x16Array((*[16]uint32)(p))
		maxVec := minVec

		end := uintptr(n) * 4
		for off := uintptr(0); off+64 <= end; off += 64 {
			v := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Add(p, off)))
			minVec = minVec.Min(v)
			maxVec = maxVec.Max(v)
			exc[0] += bits.OnesCount16(v.Greater(t0).ToBits())
			exc[1] += bits.OnesCount16(v.Greater(t1).ToBits())
			exc[2] += bits.OnesCount16(v.Greater(t2).ToBits())
			exc[3] += bits.OnesCount16(v.Greater(t3).ToBits())
			exc[4] += bits.OnesCount16(v.Greater(t4).ToBits())
			exc[5] += bits.OnesCount16(v.Greater(t5).ToBits())
			exc[6] += bits.OnesCount16(v.Greater(t6).ToBits())
			exc[7] += bits.OnesCount16(v.Greater(t7).ToBits())
		}

		minR8 := minVec.GetLo().Min(minVec.GetHi())
		maxR8 := maxVec.GetLo().Max(maxVec.GetHi())
		minR4 := minR8.GetLo().Min(minR8.GetHi())
		maxR4 := maxR8.GetLo().Max(maxR8.GetHi())
		var minL, maxL [4]uint32
		minR4.StoreArray(&minL)
		maxR4.StoreArray(&maxL)
		minVal = min(min(minL[0], minL[1]), min(minL[2], minL[3]))
		maxVal = max(max(maxL[0], maxL[1]), max(maxL[2], maxL[3]))
		tail := (n / 16) * 16
		for i := tail; i < n; i++ {
			v := values[i]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
			exc[0] += gtCountU32(v, 0)
			exc[1] += gtCountU32(v, 0xF)
			exc[2] += gtCountU32(v, 0xFF)
			exc[3] += gtCountU32(v, 0xFFF)
			exc[4] += gtCountU32(v, 0xFFFF)
			exc[5] += gtCountU32(v, 0xFFFFF)
			exc[6] += gtCountU32(v, 0xFFFFFF)
			exc[7] += gtCountU32(v, 0xFFFFFFF)
		}
	} else {
		minVal, maxVal = findMinMaxScalar(values)
		for _, v := range values {
			exc[0] += gtCountU32(v, 0)
			exc[1] += gtCountU32(v, 0xF)
			exc[2] += gtCountU32(v, 0xFF)
			exc[3] += gtCountU32(v, 0xFFF)
			exc[4] += gtCountU32(v, 0xFFFF)
			exc[5] += gtCountU32(v, 0xFFFFF)
			exc[6] += gtCountU32(v, 0xFFFFFF)
			exc[7] += gtCountU32(v, 0xFFFFFFF)
		}
	}

	if minVal == 0 {
		return false, 0, forWidthNone
	}
	forMaxStep := roundUpToStep(bits.Len32(maxVal - minVal))
	rawMaxStep := roundUpToStep(bits.Len32(maxVal))
	if forMaxStep >= rawMaxStep {
		return false, 0, forWidthNone
	}

	stdWidth, _, _ := chooseBestFromExcCounts(exc)
	if forMaxStep >= stdWidth {
		return false, 0, forWidthNone
	}
	return true, minVal, selectFORWidth(minVal)
}

// selectBitWidthNoPatchAVX512 computes the minimum step bitwidth using AVX-512
// OR-reduction. No exception analysis is performed.
func selectBitWidthNoPatchAVX512(values []uint32) int {
	orVec := archsimd.BroadcastUint32x16(0)
	i := 0
	for ; i+16 <= len(values); i += 16 {
		orVec = orVec.Or(archsimd.LoadUint32x16(values[i:]))
	}
	var lanes [16]uint32
	orVec.StoreArray(&lanes)
	var orAll uint32
	for _, v := range lanes {
		orAll |= v
	}

	// TODO-PERF: Fallback to AVX2 for the tail
	for ; i < len(values); i++ {
		orAll |= values[i]
	}
	return roundUpToStep(bits.Len32(orAll))
}

// findMinMaxAVX512 computes min/max using AVX-512 16-wide operations with 2
// independent accumulator pairs for instruction-level parallelism.
// Uses pointer-based loads to avoid per-iteration slice bounds checking.
// Register budget: 4 accumulators + 2 chunk temps = 6 ZMM (of 32 available).
func findMinMaxAVX512(values []uint32) (uint32, uint32) {
	n := len(values)
	if n < 32 {
		return findMinMaxScalar(values)
	}

	// Prime two independent 16-lane accumulators from the first 32 values.
	p := unsafe.Pointer(&values[0])
	min0 := archsimd.LoadUint32x16Array((*[16]uint32)(p))
	min1 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Add(p, 64)))
	max0, max1 := min0, min1

	// Process two 64-byte vectors per iteration to maximize throughput.
	end := uintptr(n) * 4
	for off := uintptr(128); off+128 <= end; off += 128 {
		c0 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Add(p, off)))
		c1 := archsimd.LoadUint32x16Array((*[16]uint32)(unsafe.Pointer(uintptr(p) + off + 64)))
		min0 = min0.Min(c0)
		max0 = max0.Max(c0)
		min1 = min1.Min(c1)
		max1 = max1.Max(c1)
	}

	minVec16 := min0.Min(min1)
	maxVec16 := max0.Max(max1)
	// Fold 16 lanes -> 8 lanes -> 4 lanes before final scalar reduction.
	minR8 := minVec16.GetLo().Min(minVec16.GetHi())
	maxR8 := maxVec16.GetLo().Max(maxVec16.GetHi())
	minR4 := minR8.GetLo().Min(minR8.GetHi())
	maxR4 := maxR8.GetLo().Max(maxR8.GetHi())
	var minL, maxL [4]uint32
	minR4.StoreArray(&minL)
	maxR4.StoreArray(&maxL)
	minResult := min(min(minL[0], minL[1]), min(minL[2], minL[3]))
	maxResult := max(max(maxL[0], maxL[1]), max(maxL[2], maxL[3]))

	// TODO-PERF: Use AVX2 for the tail!
	// Handle the remaining values that do not complete a 32-value block.
	tail := (n / 32) * 32
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

// forSubtractAVX512 subtracts baseValue from each element using AVX-512.
func forSubtractAVX512(dst, src []uint32, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x16(baseValue)
	i := 0
	for ; i+16 <= len(src); i += 16 {
		v := archsimd.LoadUint32x16(src[i:])
		v = v.Sub(baseVec)
		v.Store(dst[i:])
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
		v := archsimd.LoadUint32x16(output[i:])
		v = v.Add(baseVec)
		v.Store(output[i:])
	}
	for ; i < count; i++ {
		output[i] += baseValue
	}
}

// splitUint64AVX512 splits uint64 values into lower and upper uint32 halves
// using AVX-512 TruncToUint32 (8 values per iteration).
func splitUint64AVX512(lower, upper []uint32, values []uint64, count int) {
	i := 0
	for ; i+8 <= count; i += 8 {
		v := archsimd.LoadUint64x8(values[i : i+8])
		v.TruncToUint32().Store(lower[i : i+8])
		v.ShiftAllRight(32).TruncToUint32().Store(upper[i : i+8])
	}
	for ; i < count; i++ {
		lower[i] = uint32(values[i])
		upper[i] = uint32(values[i] >> 32)
	}
}

// allFitIn32BitsAVX512 checks if all uint64 values fit in 32 bits by
// OR-accumulating with Uint64x8 (8 values per iteration).
func allFitIn32BitsAVX512(values []uint64) bool {
	n := len(values)
	if n < 8 {
		return allFitIn32BitsAVX2(values)
	}

	accVec := archsimd.LoadUint64x8(values[:8])

	for i := 8; i+8 <= n; i += 8 {
		chunk := archsimd.LoadUint64x8(values[i : i+8])
		accVec = accVec.Or(chunk)
	}

	reduced4 := accVec.GetLo().Or(accVec.GetHi())
	reduced2 := reduced4.GetLo().Or(reduced4.GetHi())
	var lanes [2]uint64
	reduced2.Store(lanes[:])
	acc := lanes[0] | lanes[1]

	tail := (n / 8) * 8
	for i := tail; i < n; i++ {
		acc |= values[i]
	}
	return acc>>32 == 0
}

// findMinMax64AVX512 computes min/max of uint64 values using AVX-512
// Uint64x8.Min/Max (8 values per iteration).
func findMinMax64AVX512(values []uint64) (uint64, uint64) {
	n := len(values)
	if n < 8 {
		min64, max64 := values[0], values[0]
		for _, v := range values[1:] {
			min64 = min(min64, v)
			max64 = max(max64, v)
		}
		return min64, max64
	}

	minVec := archsimd.LoadUint64x8(values[:8])
	maxVec := minVec

	for i := 8; i+8 <= n; i += 8 {
		chunk := archsimd.LoadUint64x8(values[i : i+8])
		minVec = minVec.Min(chunk)
		maxVec = maxVec.Max(chunk)
	}

	min4 := minVec.GetLo().Min(minVec.GetHi())
	max4 := maxVec.GetLo().Max(maxVec.GetHi())
	min2 := min4.GetLo().Min(min4.GetHi())
	max2 := max4.GetLo().Max(max4.GetHi())

	var minL, maxL [2]uint64
	min2.Store(minL[:])
	max2.Store(maxL[:])
	minResult := min(minL[0], minL[1])
	maxResult := max(maxL[0], maxL[1])

	tail := (n / 8) * 8
	for i := tail; i < n; i++ {
		minResult = min(minResult, values[i])
		maxResult = max(maxResult, values[i])
	}
	return minResult, maxResult
}

// narrowToUint32AVX512 copies uint64 values to uint32 by truncation using
// AVX-512 TruncToUint32 (VPMOVQD, 8 values per iteration).
func narrowToUint32AVX512(dst []uint32, values []uint64, count int) {
	i := 0
	for ; i+8 <= count; i += 8 {
		archsimd.LoadUint64x8(values[i : i+8]).TruncToUint32().Store(dst[i : i+8])
	}
	for ; i < count; i++ {
		dst[i] = uint32(values[i])
	}
}

// forSubtract64AVX512 subtracts base from each uint64 value and stores as uint32.
// Uses AVX-512 Sub + TruncToUint32 (8 values per iteration).
func forSubtract64AVX512(dst []uint32, values []uint64, base uint64) {
	baseVec := archsimd.BroadcastUint64x8(base)
	n := len(values)
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadUint64x8(values[i : i+8])
		v.Sub(baseVec).TruncToUint32().Store(dst[i : i+8])
	}
	for ; i < n; i++ {
		dst[i] = uint32(values[i] - base)
	}
}

// forAdd64AVX512 adds a uint64 base to each uint32 value and stores as uint64.
// Uses AVX-512 Uint32x8.ExtendToUint64 + Add (8 values per iteration).
func forAdd64AVX512(dst []uint64, values []uint32, base uint64, count int) {
	baseVec := archsimd.BroadcastUint64x8(base)
	i := 0
	for ; i+8 <= count; i += 8 {
		lo := archsimd.LoadUint32x8(values[i : i+8])
		wide := lo.ExtendToUint64().Add(baseVec)
		wide.Store(dst[i : i+8])
	}
	for ; i < count; i++ {
		dst[i] = uint64(values[i]) + base
	}
}

// combineUint64AVX512 merges lower and upper uint32 halves into uint64 values.
// Uses AVX-512 ExtendToUint64 (8 values at a time), shifts uppers left by 32, ORs.
func combineUint64AVX512(dst []uint64, lower, upper []uint32, count int) {
	i := 0
	for ; i+8 <= count; i += 8 {
		loVec := archsimd.LoadUint32x8(lower[i : i+8]).ExtendToUint64()
		hiVec := archsimd.LoadUint32x8(upper[i : i+8]).ExtendToUint64()
		result := loVec.Or(hiVec.ShiftAllLeft(32))
		result.Store(dst[i : i+8])
	}
	for ; i < count; i++ {
		dst[i] = uint64(upper[i])<<32 | uint64(lower[i])
	}
}

// packBlockAVX512 packs uint32 values into a UTL block using AVX-512-accelerated
// kernels for bit-packing, delta encoding, FOR selection, and bitwidth selection.
// Accepts configurable typeFlags and hasCombine for use by uint64 sub-block paths.
// When NoInPlace is set, the input values slice is not modified.
func packBlockAVX512(flag Flag, values []uint32, dst []byte, scratch []uint32, typeFlags uint32, hasCombine bool) ([]byte, error) {
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
		useFOR, baseValue, forW = selectBitWidthWithFORAVX512(values)
		if useFOR {
			if noInPlace {
				workValues = scratch[:len(values)]
				forSubtractAVX512(workValues, values, baseValue)
				workRedirected = true
			} else {
				forSubtractAVX512(values, values, baseValue)
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
		bitWidth = selectBitWidthNoPatchAVX512(workValues)
	} else {
		bitWidth, excCount = selectBitWidthAVX512(workValues)
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
		packLanesUTLAVX512(dst[pOff:pOff+payloadSize], packInput, bitWidth)
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

	packLanesUTLAVX512(dst[pOff:pOff+payloadSize], packInput, bitWidth)
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

// analyzeUint64AVX512 computes OR-accumulator and min/max of uint64 values
// in a single fully-SIMD pass. Uses Uint64x8.Min/Max (VPMINUQ/VPMAXUQ)
// and Uint64x8.Or, processing 8 values per iteration.
func analyzeUint64AVX512(values []uint64) (min64, max64, acc uint64) {
	n := len(values)
	if n < 8 {
		return analyzeUint64AVX2(values)
	}

	chunk0 := archsimd.LoadUint64x8(values[:8])
	minVec, maxVec, accVec := chunk0, chunk0, chunk0

	for i := 8; i+8 <= n; i += 8 {
		v := archsimd.LoadUint64x8(values[i : i+8])
		minVec = minVec.Min(v)
		maxVec = maxVec.Max(v)
		accVec = accVec.Or(v)
	}

	min4 := minVec.GetLo().Min(minVec.GetHi())
	max4 := maxVec.GetLo().Max(maxVec.GetHi())
	acc4 := accVec.GetLo().Or(accVec.GetHi())
	min2 := min4.GetLo().Min(min4.GetHi())
	max2 := max4.GetLo().Max(max4.GetHi())
	acc2 := acc4.GetLo().Or(acc4.GetHi())

	var minL, maxL, accL [2]uint64
	min2.Store(minL[:])
	max2.Store(maxL[:])
	acc2.Store(accL[:])
	min64 = min(minL[0], minL[1])
	max64 = max(maxL[0], maxL[1])
	acc = accL[0] | accL[1]

	tail := (n / 8) * 8
	for i := tail; i < n; i++ {
		v := values[i]
		acc |= v
		min64 = min(min64, v)
		max64 = max(max64, v)
	}
	return
}

// packUint64AVX512 is the AVX-512 implementation of PackUint64.
// Uses a fused analysis pass (SIMD min/max + OR-acc via VPMINUQ/VPMAXUQ)
// followed by path-specific dispatch, consistent with the SSE2/AVX2 pattern.
func packUint64AVX512(flag Flag, values []uint64, dst []byte, scratch []uint32) ([]byte, error) {
	count := len(values)
	if count == 0 || count > blockSize {
		return nil, ErrInvalidBuffer
	}

	off := len(dst) * int((flag&Append)>>4)
	innerFlag := flag &^ Append

	dst = ensureCapacity64(dst, off, innerFlag)

	min64, max64, acc := analyzeUint64AVX512(values)

	if innerFlag&NoFOR == 0 && min64 > 0 && (max64-min64) < (1<<32) {
		forSubtract64AVX512(scratch[:count], values, min64)
		result, err := packFor64(innerFlag, dst, scratch, off, count, min64, packBlockAVX512)
		archsimd.ClearAVXUpperBits()
		return result, err
	}

	if acc>>32 == 0 {
		narrowToUint32AVX512(scratch, values, count)
		block, err := packBlockAVX512(innerFlag, scratch[:count], dst[off:off], scratch[blockSize:2*blockSize], headerTypeUint64Flag, false)
		if err != nil {
			return nil, err
		}
		archsimd.ClearAVXUpperBits()
		return dst[:off+len(block)], nil
	}

	upper := scratch[2*blockSize:]
	splitUint64AVX512(scratch, upper, values, count)
	result, err := packUint64TwoBlock(innerFlag, dst, scratch, off, count, packBlockAVX512)
	archsimd.ClearAVXUpperBits()
	return result, err
}

// unpackBlockAVX512 unpacks a uint32 block using AVX-512-accelerated kernels.
// Accepts forUint64 to handle uint64 sub-block context (type validation,
// FOR64 single-block handling, combine flag recognition).
func unpackBlockAVX512(dst []uint32, scratch []uint32, buf []byte, forUint64 bool) ([]uint32, int, error) {
	if len(buf) < headerBytes {
		return nil, 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, intType, excCount, forWidth, hasExceptions, hasDelta, hasZigZag, _, hasCombine := decodeHeader(header)
	hasFOR := forWidth > 0

	if forUint64 {
		if err := validateIntType64(intType); err != nil {
			return nil, 0, err
		}
	} else {
		if err := validateIntType(intType); err != nil {
			return nil, 0, err
		}
		hasCombine = false
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

	u64Single := forUint64 && isFor64SingleBlock(intType, forWidth, hasCombine)

	pOff := headerBytes
	if hasExceptions {
		pOff += svbLenBytes
	}
	var forBase uint32
	if hasFOR {
		if u64Single {
			pOff += for64BaseSize
		} else {
			forBase = readFORBase(buf, pOff, forWidth)
			pOff += forBaseBytes(forWidth)
		}
	}
	if hasCombine {
		pOff += block2LenBytes
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
	unpackLanesUTLAVX512(dst, payload, blockSize, bitWidth)

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

	if hasFOR && !u64Single {
		forAddAVX512(dst, count, forBase)
	}

	archsimd.ClearAVXUpperBits()
	return dst, consumed, nil
}

// unpackUint64AVX512 is the AVX-512 implementation of UnpackUint64.
func unpackUint64AVX512(buf []byte, dst []uint64, scratch []uint32) ([]uint64, int, error) {
	dst, consumed, err := unpackUint64Block(buf, dst, scratch, unpackBlockAVX512, forAdd64AVX512, combineUint64AVX512)
	archsimd.ClearAVXUpperBits()
	return dst, consumed, err
}
