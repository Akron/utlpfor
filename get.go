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
	if len(buf) < headerBytes {
		return 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, intType, _, _, _, _, _ := decodeHeader(header)

	if err := validateIntType(intType); err != nil {
		return 0, err
	}

	if count == 0 || pos < 0 || pos >= count {
		return 0, ErrPositionOutOfRange
	}

	if bitWidth > 32 {
		return 0, ErrInvalidBuffer
	}

	if bitWidth == 0 {
		return 0, nil
	}

	payloadBytes := utlPayloadBytesLUT[bitWidth]
	if len(buf) < headerBytes+payloadBytes {
		return 0, ErrInvalidBuffer
	}

	payload := buf[headerBytes : headerBytes+payloadBytes]
	return extractPackedValueUTL(pos, payload, bitWidth), nil
}
