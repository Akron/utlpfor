package utlpfor

// packLanesUTLScalar packs values into UTL lane-interleaved format.
func packLanesUTLScalar(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	// TODO-PERF: consider unrolling the lane loop
	for lane := range utlLaneCount {
		packLaneUTL(dst, values, lane, bitWidth)
	}
}

// packLaneUTL packs values for a single UTL lane.
func packLaneUTL(dst []byte, values []uint32, lane, bitWidth int) {
	var mask uint64
	if bitWidth >= 32 {
		mask = 0xFFFFFFFF
	} else {
		mask = uint64((1 << bitWidth) - 1)
	}

	var acc uint64
	var bitsInAcc int
	outByteIdx := lane * 4

	for v := range utlValuesPerLane {
		seqIdx := lane + v*utlLaneCount
		var val uint32
		if seqIdx < len(values) {
			val = values[seqIdx]
		}
		acc |= (uint64(val) & mask) << bitsInAcc
		bitsInAcc += bitWidth
		for bitsInAcc >= 32 {
			bo.PutUint32(dst[outByteIdx:], uint32(acc))
			outByteIdx += utlSuperWordBytes
			acc >>= 32
			bitsInAcc -= 32
		}
	}
	if bitsInAcc > 0 {
		bo.PutUint32(dst[outByteIdx:], uint32(acc))
	}
}

// unpackLanesUTLScalar unpacks UTL lane-interleaved payload into sequential values.
func unpackLanesUTLScalar(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	// TODO-PERF: consider unrolling the lane loop
	for lane := range utlLaneCount {
		unpackLaneUTL(dst, payload, lane, bitWidth, count)
	}
}

// unpackLaneUTL unpacks values for a single UTL lane.
func unpackLaneUTL(dst []uint32, payload []byte, lane, bitWidth, count int) {
	var mask uint32
	if bitWidth >= 32 {
		mask = 0xFFFFFFFF
	} else {
		mask = (1 << bitWidth) - 1
	}

	var acc uint64
	var bitsInAcc int
	inByteIdx := lane * 4

	for v := range utlValuesPerLane {
		for bitsInAcc < bitWidth {
			if inByteIdx+4 > len(payload) {
				bitsInAcc = bitWidth
				break
			}
			acc |= uint64(bo.Uint32(payload[inByteIdx:])) << bitsInAcc
			inByteIdx += utlSuperWordBytes
			bitsInAcc += 32
		}
		value := uint32(acc) & mask
		acc >>= bitWidth
		bitsInAcc -= bitWidth
		seqIdx := lane + v*utlLaneCount
		if seqIdx < count {
			dst[seqIdx] = value
		}
	}
}

// packUint32Scalar is the scalar implementation of PackUint32.
func packUint32Scalar(flag byte, dst []byte, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	headerFlags := headerTypeUint32Flag
	workValues := values

	if flag&Delta != 0 {
		needZZ := deltaEncodePerLaneScalar(values, values)
		if needZZ {
			headerFlags |= headerZigZagFlag
		}
		headerFlags |= headerDeltaFlag
		workValues = values
	}

	bitWidth := maxBitWidth(workValues)
	payloadBytes := utlPayloadBytesLUT[bitWidth]
	totalLen := headerBytes + payloadBytes

	dst = ensureLen(dst, totalLen)
	bo.PutUint32(dst, encodeHeader(len(values), bitWidth, 0, headerFlags))

	payload := dst[headerBytes : headerBytes+payloadBytes]
	packLanesUTLScalar(payload, workValues, bitWidth)

	return dst[:totalLen], nil
}

// unpackUint32Scalar is the scalar implementation of UnpackUint32.
func unpackUint32Scalar(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	if len(buf) < headerBytes {
		return nil, 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, intType, _, _, hasDelta, hasZigZag, _ := decodeHeader(header)

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

	payloadBytes := utlPayloadBytesLUT[bitWidth]
	if len(buf) < headerBytes+payloadBytes {
		return nil, 0, ErrInvalidBuffer
	}

	if cap(dst) < count {
		dst = make([]uint32, count)
	}
	dst = dst[:count]

	payload := buf[headerBytes : headerBytes+payloadBytes]
	unpackLanesUTLScalar(dst, payload, count, bitWidth)

	if hasDelta {
		overflowPos := deltaDecodePerLaneWithOverflowScalar(dst, dst, hasZigZag)
		if overflowPos > 0 {
			return nil, 0, &ErrOverflow{Position: overflowPos}
		}
	}

	return dst, headerBytes + payloadBytes, nil
}
