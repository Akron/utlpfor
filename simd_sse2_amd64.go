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
	for i := 0; i <= n-4; i += 4 {
		v := archsimd.LoadUint32x4(buf[i : i+4])
		shifted := v.ShiftAllLeft(1)
		sign := v.AsInt32x4().ShiftAllRight(31).AsUint32x4()
		result := shifted.Xor(sign)
		result.Store(buf[i : i+4])
	}
	for i := (n / 4) * 4; i < n; i++ {
		buf[i] = zigzagEncode32(int32(buf[i]))
	}
}

// zigzagDecodeSSE2 applies zigzag decoding in-place using SSE2 (Uint32x4).
// Formula: (n >>> 1) ^ -(n & 1).
func zigzagDecodeSSE2(values []uint32) {
	one := archsimd.BroadcastUint32x4(1)
	zero := archsimd.BroadcastUint32x4(0)
	for i := 0; i <= len(values)-4; i += 4 {
		v := archsimd.LoadUint32x4(values[i : i+4])
		half := v.ShiftAllRight(1)
		signBit := v.And(one)
		negSign := zero.Sub(signBit)
		result := half.Xor(negSign)
		result.Store(values[i : i+4])
	}
	for i := (len(values) / 4) * 4; i < len(values); i++ {
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

// deltaDecodePerLaneWithOverflowSSE2 performs prefix sum with overflow check
// using SSE2 (Uint32x4). Returns position of first overflow (0 = none).
func deltaDecodePerLaneWithOverflowSSE2(values []uint32, useZigZag bool) int {
	if useZigZag {
		deltaDecodePerLaneSSE2(values, true)
		return 0
	}

	var overflowPos int

	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		for group := range 4 {
			off := group * 4
			prev := archsimd.LoadUint32x4(values[prevBase+off : prevBase+off+4])
			cur := archsimd.LoadUint32x4(values[curBase+off : curBase+off+4])
			sum := prev.Add(cur)

			overflow := sum.Less(prev)
			if overflowPos == 0 && overflow.ToBits() != 0 {
				lane := bits.TrailingZeros8(overflow.ToBits())
				overflowPos = curBase + off + lane
			}

			sum.Store(values[curBase+off : curBase+off+4])
		}
	}

	return overflowPos
}

// packUint32SSE2 is the full SSE2 packing pipeline.
// Uses SSE2 for bit-packing and delta/zigzag; scalar for exceptions and FOR.
func packUint32SSE2(flag Flag, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	off := 0
	if flag&Append != 0 {
		off = len(dst)
	}
	flag &^= Append

	headerFlags := headerTypeUint32Flag
	headerFlags |= uint32(flag&Special) << 14

	var useFOR bool
	var baseValue uint32
	var forW int
	if flag&NoFOR == 0 {
		useFOR, baseValue, forW = selectBitWidthWithFORSSE2(values)
		if useFOR {
			forSubtractSSE2(values, values, baseValue)
			headerFlags |= uint32(forW) << forWidthShift
		}
	}

	if flag&Delta != 0 {
		var needZZ bool
		if len(values) == blockSize {
			needZZ = deltaEncodePerLaneSSE2(values)
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
		bitWidth = selectBitWidthNoPatchSSE2(values)
	} else {
		bitWidth, excCount = selectBitWidthSSE2(values)
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
		if off > 0 {
			dst = ensureAppend(dst, off, totalLen)
		} else {
			dst = ensureLen(dst, totalLen)
		}
		block := dst[off:]
		bo.PutUint32(block, encodeHeader(len(values), bitWidth, 0, headerFlags))
		if useFOR {
			writeFORBase(block, baseValue, forW, false)
		}
		packLanesUTLSSE2(block[pOff:pOff+payloadBytes], packInput, bitWidth)
		return dst[:off+totalLen], nil
	}

	excIdxSize := excIndexSize(excCount)
	maxSvbLen := maxSVBEncodedLen(excCount)
	pOff := payloadOffset(forBaseBytes, true)
	maxTotalLen := pOff + payloadBytes + excIdxSize + maxSvbLen

	if off > 0 {
		dst = ensureAppend(dst, off, maxTotalLen)
	} else {
		dst = ensureLen(dst, maxTotalLen)
	}
	block := dst[off:]
	bo.PutUint32(block, encodeHeader(len(values), bitWidth, excCount, headerFlags))
	if useFOR {
		writeFORBase(block, baseValue, forW, true)
	}

	packLanesUTLSSE2(block[pOff:pOff+payloadBytes], packInput, bitWidth)

	highBits := scratch[:blockSize]

	excOff := pOff + payloadBytes
	collectAndWriteExceptions(values, bitWidth, block[excOff:], excCount, highBits)

	svbOffset := excOff + excIdxSize
	svbLen := encodeSVBIntoDst(block[svbOffset:maxTotalLen], highBits[:excCount])

	bo.PutUint16(block[headerBytes:], uint16(svbLen))
	return dst[:off+svbOffset+svbLen], nil
}

// unpackUint32SSE2 is the full SSE2 unpacking pipeline.
// Uses SSE2 for bit-unpacking and delta/zigzag; scalar for exceptions and FOR.
func unpackUint32SSE2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	if len(buf) < headerBytes {
		return nil, 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, intType, excCount, forWidth, hasExceptions, hasDelta, hasZigZag, _ := decodeHeader(header)
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
				deltaDecodePerLaneSSE2(dst, true)
			} else {
				deltaDecodePerLaneScalar(dst, true)
			}
		} else {
			var overflowPos int
			if count == blockSize {
				overflowPos = deltaDecodePerLaneWithOverflowSSE2(dst, false)
			} else {
				overflowPos = deltaDecodePerLaneWithOverflowScalar(dst, false)
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
	orVec := archsimd.BroadcastUint32x4(0)
	i := 0
	for ; i+4 <= len(values); i += 4 {
		orVec = orVec.Or(archsimd.LoadUint32x4(values[i:]))
	}
	var lanes [4]uint32
	orVec.StoreArray(&lanes)
	var orAll uint32
	for _, v := range lanes {
		orAll |= v
	}
	// Use scalar for the tail
	for ; i < len(values); i++ {
		orAll |= values[i]
	}
	return roundUpToStep(bits.Len32(orAll))
}

// forSubtractSSE2 subtracts baseValue from each element using SSE2.
func forSubtractSSE2(dst, src []uint32, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x4(baseValue)
	i := 0
	for ; i+4 <= len(src); i += 4 {
		v := archsimd.LoadUint32x4(src[i:])
		v = v.Sub(baseVec)
		v.Store(dst[i:])
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
		v := archsimd.LoadUint32x4(output[i:])
		v = v.Add(baseVec)
		v.Store(output[i:])
	}
	for ; i < count; i++ {
		output[i] += baseValue
	}
}
