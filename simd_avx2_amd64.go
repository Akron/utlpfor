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
	for v := 0; v < utlValuesPerLane; v++ {
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
		for v := 0; v < utlValuesPerLane; v++ {
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
	for v := 0; v < utlValuesPerLane; v++ {
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
	for i := 0; i <= n-8; i += 8 {
		v := archsimd.LoadUint32x8Slice(buf[i : i+8])
		shifted := v.ShiftAllLeft(1)
		sign := v.AsInt32x8().ShiftAllRight(31).AsUint32x8()
		result := shifted.Xor(sign)
		result.StoreSlice(buf[i : i+8])
	}
	for i := (n / 8) * 8; i < n; i++ {
		buf[i] = zigzagEncode32(int32(buf[i]))
	}
}

// zigzagDecodeAVX2 applies zigzag decoding using AVX2.
// Formula: (n >>> 1) ^ -(n & 1) where >>> is logical right shift.
func zigzagDecodeAVX2(dst, src []uint32) {
	one := archsimd.BroadcastUint32x8(1)
	zero := archsimd.BroadcastUint32x8(0)
	for i := 0; i <= len(src)-8; i += 8 {
		v := archsimd.LoadUint32x8Slice(src[i : i+8])
		half := v.ShiftAllRight(1)
		signBit := v.And(one)
		negSign := zero.Sub(signBit)
		result := half.Xor(negSign)
		result.StoreSlice(dst[i : i+8])
	}
	for i := (len(src) / 8) * 8; i < len(src); i++ {
		dst[i] = uint32(zigzagDecode32(src[i]))
	}
}

// deltaEncodePerLaneAVX2 computes per-lane deltas using AVX2.
// Processes all 16 lanes in two register groups (lanes 0-7 and 8-15).
// Returns true if zigzag encoding was needed (negative deltas detected).
// Borrow detection is batched: all borrow masks are ORed into a single uint8
// and checked once after the loop, eliminating branches from the inner loop.
func deltaEncodePerLaneAVX2(dst, src []uint32) bool {
	var anyBorrow uint8

	for v := utlValuesPerLane - 1; v > 0; v-- {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		cur0 := archsimd.LoadUint32x8Slice(src[curBase : curBase+8])
		prev0 := archsimd.LoadUint32x8Slice(src[prevBase : prevBase+8])
		delta0 := cur0.Sub(prev0)

		cur1 := archsimd.LoadUint32x8Slice(src[curBase+8 : curBase+16])
		prev1 := archsimd.LoadUint32x8Slice(src[prevBase+8 : prevBase+16])
		delta1 := cur1.Sub(prev1)

		anyBorrow |= prev0.Greater(cur0).ToBits()
		anyBorrow |= prev1.Greater(cur1).ToBits()

		delta0.StoreSlice(dst[curBase : curBase+8])
		delta1.StoreSlice(dst[curBase+8 : curBase+16])
	}

	copy(dst[:utlLaneCount], src[:utlLaneCount])

	needZigZag := anyBorrow != 0
	if needZigZag {
		zigzagEncodeAVX2(dst, len(src))
	}
	return needZigZag
}

// deltaDecodePerLaneAVX2 performs per-lane prefix sums using AVX2.
func deltaDecodePerLaneAVX2(dst, deltas []uint32, useZigZag bool) {
	if useZigZag {
		zigzagDecodeAVX2(dst, deltas)
	} else {
		copy(dst, deltas)
	}

	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		prev0 := archsimd.LoadUint32x8Slice(dst[prevBase : prevBase+8])
		cur0 := archsimd.LoadUint32x8Slice(dst[curBase : curBase+8])
		sum0 := prev0.Add(cur0)
		sum0.StoreSlice(dst[curBase : curBase+8])

		prev1 := archsimd.LoadUint32x8Slice(dst[prevBase+8 : prevBase+16])
		cur1 := archsimd.LoadUint32x8Slice(dst[curBase+8 : curBase+16])
		sum1 := prev1.Add(cur1)
		sum1.StoreSlice(dst[curBase+8 : curBase+16])
	}
}

// deltaDecodePerLaneWithOverflowAVX2 performs prefix sum with overflow check.
// Returns the position of the first overflow (0 = no overflow).
func deltaDecodePerLaneWithOverflowAVX2(dst, deltas []uint32, useZigZag bool) int {
	if useZigZag {
		deltaDecodePerLaneAVX2(dst, deltas, true)
		return 0
	}

	copy(dst, deltas)
	var overflowPos int

	for v := 1; v < utlValuesPerLane; v++ {
		curBase := v * utlLaneCount
		prevBase := (v - 1) * utlLaneCount

		prev0 := archsimd.LoadUint32x8Slice(dst[prevBase : prevBase+8])
		cur0 := archsimd.LoadUint32x8Slice(dst[curBase : curBase+8])
		sum0 := prev0.Add(cur0)

		overflow0 := sum0.Less(prev0)
		if overflowPos == 0 && overflow0.ToBits() != 0 {
			lane := bits.TrailingZeros8(overflow0.ToBits())
			overflowPos = curBase + lane
		}

		sum0.StoreSlice(dst[curBase : curBase+8])

		prev1 := archsimd.LoadUint32x8Slice(dst[prevBase+8 : prevBase+16])
		cur1 := archsimd.LoadUint32x8Slice(dst[curBase+8 : curBase+16])
		sum1 := prev1.Add(cur1)

		overflow1 := sum1.Less(prev1)
		if overflowPos == 0 && overflow1.ToBits() != 0 {
			lane := bits.TrailingZeros8(overflow1.ToBits())
			overflowPos = curBase + 8 + lane
		}

		sum1.StoreSlice(dst[curBase+8 : curBase+16])
	}

	return overflowPos
}

// packUint32AVX2 is the full AVX2 packing pipeline.
// Uses AVX2 for bit-packing and delta/zigzag encoding; scalar for exceptions.
func packUint32AVX2(flag byte, dst []byte, values []uint32) ([]byte, error) {
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
	packInput := workValues
	if len(workValues) < blockSize {
		copy(padded[:], workValues)
		packInput = padded[:]
	}

	if !hasExceptions {
		totalLen := headerBytes + payloadBytes
		dst = ensureLen(dst, totalLen)
		bo.PutUint32(dst, encodeHeader(len(values), bitWidth, 0, headerFlags))
		packLanesUTLAVX2(dst[headerBytes:headerBytes+payloadBytes], packInput, bitWidth)
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

	packLanesUTLAVX2(dst[pOff:pOff+payloadBytes], packInput, bitWidth)
	archsimd.ClearAVXUpperBits()
	writeExceptionsDirect(dst[pOff+payloadBytes:], positions[:], bitmap[:], excCount, svbData)

	return dst[:totalLen], nil
}

// unpackUint32AVX2 is the full AVX2 unpacking pipeline.
// Uses AVX2 for bit-unpacking and delta/zigzag; scalar for exceptions.
func unpackUint32AVX2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
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
	unpackLanesUTLAVX2(dst, payload, blockSize, bitWidth)
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

// getUint32AVX2 delegates to scalar for random access.
func getUint32AVX2(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
