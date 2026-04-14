package utlpfor

// packLanesUTLScalar packs values into UTL lane-interleaved format using
// 64-bit pair-lane processing. Two adjacent lanes are packed simultaneously
// with a single uint64 store per super-word access, halving memory operations.
// This is using the pattern described in the FastLanes paper.
func packLanesUTLScalar(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	for lanePair := 0; lanePair < utlLaneCount; lanePair += 2 {
		packLanePairUTL64(dst, values, lanePair, bitWidth)
	}
}

// packLanePairUTL64 packs values for two adjacent UTL lanes using a pair
// of uint64 accumulators and uint64 stores.
// This is using the pattern described in the FastLanes paper.
func packLanePairUTL64(dst []byte, values []uint32, lane0, bitWidth int) {
	var mask uint64
	if bitWidth >= 32 {
		mask = 0xFFFFFFFF
	} else {
		mask = uint64((1 << bitWidth) - 1)
	}

	var acc0, acc1 uint64
	var bitsInAcc int
	outByteIdx := lane0 * 4

	for v := range utlValuesPerLane {
		seqIdx0 := lane0 + v*utlLaneCount
		seqIdx1 := lane0 + 1 + v*utlLaneCount
		var val0, val1 uint32
		if seqIdx0 < len(values) {
			val0 = values[seqIdx0]
		}
		if seqIdx1 < len(values) {
			val1 = values[seqIdx1]
		}

		acc0 |= (uint64(val0) & mask) << bitsInAcc
		acc1 |= (uint64(val1) & mask) << bitsInAcc
		bitsInAcc += bitWidth

		for bitsInAcc >= 32 {
			pair := uint64(uint32(acc0)) | (uint64(uint32(acc1)) << 32)
			bo.PutUint64(dst[outByteIdx:], pair)
			outByteIdx += utlSuperWordBytes
			acc0 >>= 32
			acc1 >>= 32
			bitsInAcc -= 32
		}
	}
	if bitsInAcc > 0 {
		pair := uint64(uint32(acc0)) | (uint64(uint32(acc1)) << 32)
		bo.PutUint64(dst[outByteIdx:], pair)
	}
}

// unpackLanesUTLScalar unpacks UTL lane-interleaved payload into sequential
// values using 64-bit pair-lane processing. Two adjacent lanes are unpacked
// simultaneously with a single uint64 load per super-word access.
// This is using the pattern described in the FastLanes paper.
func unpackLanesUTLScalar(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	for lanePair := 0; lanePair < utlLaneCount; lanePair += 2 {
		unpackLanePairUTL64(dst, payload, lanePair, bitWidth, count)
	}
}

// unpackLanePairUTL64 unpacks values for two adjacent UTL lanes using a pair
// of uint64 accumulators and uint64 loads.
// This is using the pattern described in the FastLanes paper.
func unpackLanePairUTL64(dst []uint32, payload []byte, lane0, bitWidth, count int) {
	var mask32 uint32
	if bitWidth >= 32 {
		mask32 = 0xFFFFFFFF
	} else {
		mask32 = (1 << bitWidth) - 1
	}

	var acc0, acc1 uint64
	var bitsInAcc int
	inByteIdx := lane0 * 4

	for v := range utlValuesPerLane {
		for bitsInAcc < bitWidth {
			if inByteIdx+8 > len(payload) {
				bitsInAcc = bitWidth
				break
			}
			pair := bo.Uint64(payload[inByteIdx:])
			acc0 |= uint64(uint32(pair)) << bitsInAcc
			acc1 |= (pair >> 32) << bitsInAcc
			inByteIdx += utlSuperWordBytes
			bitsInAcc += 32
		}

		seqIdx0 := lane0 + v*utlLaneCount
		seqIdx1 := lane0 + 1 + v*utlLaneCount
		if seqIdx0 < count {
			dst[seqIdx0] = uint32(acc0) & mask32
		}
		if seqIdx1 < count {
			dst[seqIdx1] = uint32(acc1) & mask32
		}

		acc0 >>= bitWidth
		acc1 >>= bitWidth
		bitsInAcc -= bitWidth
	}
}

// packUint32Scalar is the scalar implementation of PackUint32.
func packUint32Scalar(flag byte, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	headerFlags := headerTypeUint32Flag

	var useFOR bool
	var baseValue uint32
	var forW int
	if flag&NoFOR == 0 {
		useFOR, baseValue, forW = selectBitWidthWithFOR(values)
		if useFOR {
			forSubtractScalar(values, values, baseValue)
			headerFlags |= uint32(forW) << forWidthShift
		}
	}

	if flag&Delta != 0 {
		needZZ := deltaEncodePerLaneScalar(values)
		if needZZ {
			headerFlags |= headerZigZagFlag
		}
		headerFlags |= headerDeltaFlag
	}

	var bitWidth, excCount int
	if flag&NoPatch != 0 {
		bitWidth = selectBitWidthNoPatch(values)
	} else {
		bitWidth, excCount = selectBitWidth(values)
	}
	payloadBytes := utlPayloadBytes(bitWidth)
	hasExceptions := excCount > 0
	forBaseBytes := forBaseBytes(forW)

	if !hasExceptions {
		pOff := payloadOffset(forBaseBytes, false)
		totalLen := pOff + payloadBytes
		dst = ensureLen(dst, totalLen)
		bo.PutUint32(dst, encodeHeader(len(values), bitWidth, 0, headerFlags))
		if useFOR {
			writeFORBase(dst, baseValue, forW, false)
		}
		packLanesUTLScalar(dst[pOff:pOff+payloadBytes], values, bitWidth)
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

	packLanesUTLScalar(dst[pOff:pOff+payloadBytes], values, bitWidth)

	highBits := scratch[:blockSize]

	excOff := pOff + payloadBytes
	collectAndWriteExceptions(values, bitWidth, dst[excOff:], excCount, highBits)

	svbOffset := excOff + excIdxSize
	svbLen := encodeSVBIntoDst(dst[svbOffset:maxTotalLen], highBits[:excCount])

	bo.PutUint16(dst[headerBytes:], uint16(svbLen))
	return dst[:svbOffset+svbLen], nil
}

// unpackUint32Scalar is the scalar implementation of UnpackUint32.
func unpackUint32Scalar(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
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
	unpackLanesUTLScalar(dst, payload, blockSize, bitWidth)

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
			deltaDecodePerLaneScalar(dst, true)
		} else {
			overflowPos := deltaDecodePerLaneWithOverflowScalar(dst, false)
			if overflowPos > 0 {
				return nil, 0, &ErrOverflow{Position: overflowPos}
			}
		}
	}

	if hasFOR {
		forAddScalar(dst, count, forBase)
	}

	return dst, consumed, nil
}
