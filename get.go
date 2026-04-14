package utlpfor

// extractPackedValueUTL extracts a single packed value from a UTL payload.
func extractPackedValueUTL(pos int, payload []byte, bitWidth int) uint32 {
	lane := pos % utlLaneCount
	posInLane := pos / utlLaneCount
	bitPos := posInLane * bitWidth
	wordInLane := bitPos / 32
	bitOffset := bitPos % 32

	byteOffset := wordInLane*utlSuperWordBytes + lane*4

	var acc uint64
	if byteOffset+4 <= len(payload) {
		acc = uint64(bo.Uint32(payload[byteOffset:]))
	}
	if bitWidth > 32-bitOffset {
		nextByteOffset := byteOffset + utlSuperWordBytes
		if nextByteOffset+4 <= len(payload) {
			acc |= uint64(bo.Uint32(payload[nextByteOffset:])) << 32
		}
	}

	acc >>= uint(bitOffset)
	var mask uint64
	if bitWidth >= 32 {
		mask = 0xFFFFFFFF
	} else {
		mask = uint64((1 << bitWidth) - 1)
	}
	return uint32(acc & mask)
}

// GetUint32 extracts a single value at the given position from the packed block.
func GetUint32(pos int, buf []byte) (uint32, error) {
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		return getUint32AVX512(pos, buf)
	case simdLevelAVX2:
		return getUint32AVX2(pos, buf)
	case simdLevelSSE2:
		return getUint32SSE2(pos, buf)
	default:
		return getUint32Scalar(pos, buf)
	}
}

// getUint32Scalar is the scalar implementation of GetUint32.
func getUint32Scalar(pos int, buf []byte) (uint32, error) {
	if len(buf) < headerBytes {
		return 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, intType, excCount, forWidth, hasExceptions, hasDelta, hasZigZag := decodeHeader(header)
	hasFOR := forWidth > 0

	if err := validateIntType(intType); err != nil {
		return 0, err
	}

	if pos < 0 || pos >= count {
		return 0, ErrPositionOutOfRange
	}

	if bitWidth > 32 {
		return 0, ErrInvalidBuffer
	}

	pOff := payloadOffset(0, hasExceptions)
	var forBase uint32
	if hasFOR {
		forBase = readFORBase(buf, pOff, forWidth)
		pOff += forBaseBytes(forWidth)
	}

	payloadBytes := utlPayloadBytes(bitWidth)
	if len(buf) < pOff+payloadBytes {
		return 0, ErrInvalidBuffer
	}

	payload := buf[pOff : pOff+payloadBytes]

	var value uint32
	var err error
	if !hasDelta {
		value, err = getValueDirect(pos, buf, payload, pOff+payloadBytes, bitWidth, count, excCount, hasExceptions)
	} else {
		value, err = getValueWithDelta(pos, buf, payload, pOff+payloadBytes, bitWidth, count, excCount, hasExceptions, hasZigZag)
	}
	if err != nil {
		return 0, err
	}
	return value + forBase, nil
}

// getValueDirect extracts a single value without delta decoding.
// The caller is responsible for adding the FOR base value.
// TODO-PERF: decodes ALL exception high bits just to return one value at excIdx.
// svbDecodeOne will enable targeted single-value decode.
func getValueDirect(pos int, buf []byte, payload []byte, excStart, bitWidth, count, excCount int, hasExceptions bool) (uint32, error) {
	var value uint32
	if bitWidth > 0 {
		value = extractPackedValueUTL(pos, payload, bitWidth)
	}

	if hasExceptions {
		excIdx := findExceptionIndex(buf, excStart, excCount, pos)
		if excIdx >= 0 {
			var decodeBuf [blockSize]uint32
			if _, err := decodeExceptionHighBitsInto(decodeBuf[:], buf, excStart, excCount); err != nil {
				return 0, err
			}
			value |= decodeBuf[excIdx] << bitWidth
		}
	}

	return value, nil
}

// getValueWithDelta extracts a value with per-lane delta decoding.
// Reconstructs the entire lane (up to posInLane) to compute the prefix sum.
// The caller is responsible for adding the FOR base value.
func getValueWithDelta(pos int, buf []byte, payload []byte, excStart, bitWidth, count, excCount int, hasExceptions, hasZigZag bool) (uint32, error) {
	lane := pos % utlLaneCount
	posInLane := pos / utlLaneCount

	var allHighBits [blockSize]uint32
	if hasExceptions {
		if _, err := decodeExceptionHighBitsInto(allHighBits[:], buf, excStart, excCount); err != nil {
			return 0, err
		}
	}

	var laneValues [utlValuesPerLane]uint32
	for v := 0; v <= posInLane; v++ {
		seqIdx := lane + v*utlLaneCount
		if seqIdx >= count {
			break
		}

		if bitWidth > 0 {
			laneValues[v] = extractPackedValueUTL(seqIdx, payload, bitWidth)
		}

		if hasExceptions {
			excIdx := findExceptionIndex(buf, excStart, excCount, seqIdx)
			if excIdx >= 0 {
				laneValues[v] |= allHighBits[excIdx] << bitWidth
			}
		}
	}

	if hasZigZag {
		laneValues[0] = uint32(zigzagDecode32(laneValues[0]))
	}
	for v := 1; v <= posInLane; v++ {
		seqIdx := lane + v*utlLaneCount
		if seqIdx >= count {
			break
		}
		if hasZigZag {
			laneValues[v] = laneValues[v-1] + uint32(zigzagDecode32(laneValues[v]))
		} else {
			laneValues[v] = laneValues[v-1] + laneValues[v]
		}
	}

	return laneValues[posInLane], nil
}
