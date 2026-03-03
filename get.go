package utlpfor

import "github.com/mhr3/streamvbyte"

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
	if len(buf) < headerBytes {
		return 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, intType, excCount, hasExceptions, hasDelta, hasZigZag, hasFOR := decodeHeader(header)

	if err := validateIntType(intType); err != nil {
		return 0, err
	}

	if count == 0 || pos < 0 || pos >= count {
		return 0, ErrPositionOutOfRange
	}

	if bitWidth > 32 {
		return 0, ErrInvalidBuffer
	}

	pOff := payloadOffset(hasFOR, hasExceptions)
	payloadBytes := utlPayloadBytesLUT[bitWidth]
	if len(buf) < pOff+payloadBytes {
		return 0, ErrInvalidBuffer
	}

	payload := buf[pOff : pOff+payloadBytes]

	if !hasDelta {
		return getValueDirect(pos, buf, payload, pOff+payloadBytes, bitWidth, count, excCount, hasExceptions, hasFOR)
	}
	return getValueWithDelta(pos, buf, payload, pOff+payloadBytes, bitWidth, count, excCount, hasExceptions, hasZigZag, hasFOR)
}

// getValueDirect extracts a single value without delta decoding.
func getValueDirect(pos int, buf []byte, payload []byte, excStart, bitWidth, count, excCount int, hasExceptions, hasFOR bool) (uint32, error) {
	var value uint32
	if bitWidth > 0 {
		value = extractPackedValueUTL(pos, payload, bitWidth)
	}

	if hasExceptions {
		excIdx := findExceptionIndex(buf, excStart, excCount, pos)
		if excIdx >= 0 {
			highBit, err := decodeExceptionHighBit(buf, excStart, excCount, excIdx, hasFOR)
			if err != nil {
				return 0, err
			}
			value |= highBit << bitWidth
		}
	}

	return value, nil
}

// getValueWithDelta extracts a value with per-lane delta decoding.
// Reconstructs the entire lane (up to posInLane) to compute the prefix sum.
func getValueWithDelta(pos int, buf []byte, payload []byte, excStart, bitWidth, count, excCount int, hasExceptions, hasZigZag, hasFOR bool) (uint32, error) {
	lane := pos % utlLaneCount
	posInLane := pos / utlLaneCount

	// Decode all exception high bits once (shared across lane positions).
	var allHighBits [blockSize]uint32
	if hasExceptions {
		if err := decodeAllExceptionHighBits(allHighBits[:], buf, excStart, excCount, hasFOR); err != nil {
			return 0, err
		}
	}

	// Extract and reconstruct work values for this lane.
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

	// Per-lane prefix sum (delta decode).
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

// decodeExceptionHighBit decodes a single exception's high bits by decoding
// all SVB values and returning the one at excIdx.
func decodeExceptionHighBit(buf []byte, excStart, excCount, excIdx int, hasFOR bool) (uint32, error) {
	svbLenOffset := headerBytes
	if hasFOR {
		svbLenOffset += headerFORBytes
	}
	if len(buf) < svbLenOffset+svbLenBytes {
		return 0, ErrInvalidBuffer
	}
	svbLen := int(bo.Uint16(buf[svbLenOffset:]))

	svbStart := excStart
	if excCount <= excBitmapThreshold {
		svbStart += excCount
	} else {
		svbStart += 16
	}

	if svbStart+svbLen > len(buf) {
		return 0, ErrInvalidBuffer
	}

	//stack buffer for small excCounts to avoid additional allocations
	var decodeBuf [blockSize]uint32
	highBits := streamvbyte.DecodeUint32(
		buf[svbStart:svbStart+svbLen], excCount,
		&streamvbyte.DecodeOptions[uint32]{Buffer: decodeBuf[:excCount]},
	)
	return highBits[excIdx], nil
}

// decodeAllExceptionHighBits decodes all SVB exception high bits into dst.
func decodeAllExceptionHighBits(dst []uint32, buf []byte, excStart, excCount int, hasFOR bool) error {
	svbLenOffset := headerBytes
	if hasFOR {
		svbLenOffset += headerFORBytes
	}
	if len(buf) < svbLenOffset+svbLenBytes {
		return ErrInvalidBuffer
	}
	svbLen := int(bo.Uint16(buf[svbLenOffset:]))

	svbStart := excStart
	if excCount <= excBitmapThreshold {
		svbStart += excCount
	} else {
		svbStart += 16
	}

	if svbStart+svbLen > len(buf) {
		return ErrInvalidBuffer
	}

	streamvbyte.DecodeUint32(
		buf[svbStart:svbStart+svbLen], excCount,
		&streamvbyte.DecodeOptions[uint32]{Buffer: dst[:excCount]},
	)
	return nil
}
