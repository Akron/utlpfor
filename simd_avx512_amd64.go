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
		for v := range utlValuesPerLane {
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
	for v := range utlValuesPerLane {
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
func packUint32AVX512(flag byte, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	headerFlags := headerTypeUint32Flag
	headerFlags |= uint32(flag&Special) << 14

	var useFOR bool
	var baseValue uint32
	var forW int
	if flag&NoFOR == 0 {
		useFOR, baseValue, forW = selectBitWidthWithFORAVX512(values)
		if useFOR {
			forSubtractAVX512(values, values, baseValue)
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
		bitWidth = selectBitWidthNoPatchAVX512(values)
	} else {
		bitWidth, excCount = selectBitWidthAVX512(values)
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
		packLanesUTLAVX512(dst[pOff:pOff+payloadBytes], packInput, bitWidth)
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

	packLanesUTLAVX512(dst[pOff:pOff+payloadBytes], packInput, bitWidth)
	archsimd.ClearAVXUpperBits()

	highBits := scratch[:blockSize]

	excOff := pOff + payloadBytes
	collectAndWriteExceptions(values, bitWidth, dst[excOff:], excCount, highBits)

	svbOffset := excOff + excIdxSize
	svbLen := encodeSVBIntoDst(dst[svbOffset:maxTotalLen], highBits[:excCount])

	bo.PutUint16(dst[headerBytes:], uint16(svbLen))
	return dst[:svbOffset+svbLen], nil
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

	if hasFOR {
		forAddAVX512(dst, count, forBase)
	}

	archsimd.ClearAVXUpperBits()
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
		minVec := archsimd.LoadUint32x16((*[16]uint32)(p))
		maxVec := minVec

		end := uintptr(n) * 4
		for off := uintptr(0); off+64 <= end; off += 64 {
			v := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Add(p, off)))
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
		minR4.Store(&minL)
		maxR4.Store(&maxL)
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
		orVec = orVec.Or(archsimd.LoadUint32x16Slice(values[i:]))
	}
	var lanes [16]uint32
	orVec.Store(&lanes)
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
	min0 := archsimd.LoadUint32x16((*[16]uint32)(p))
	min1 := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Add(p, 64)))
	max0, max1 := min0, min1

	// Process two 64-byte vectors per iteration to maximize throughput.
	end := uintptr(n) * 4
	for off := uintptr(128); off+128 <= end; off += 128 {
		c0 := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Add(p, off)))
		c1 := archsimd.LoadUint32x16((*[16]uint32)(unsafe.Pointer(uintptr(p) + off + 64)))
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
	minR4.Store(&minL)
	maxR4.Store(&maxL)
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
