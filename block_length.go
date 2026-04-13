package utlpfor

// BlockLength returns the total byte length of the encoded block
// starting at the beginning of buf. Only the first 4-6 bytes of the
// buffer are read, enabling efficient block skipping in MMAP-backed files.
func BlockLength(buf []byte) (int, error) {
	if len(buf) < headerBytes {
		return 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, _, excCount, forWidth, hasExceptions, _, _ := decodeHeader(header)

	if count > blockSize {
		return 0, ErrInvalidBlockLength
	}
	if bitWidth > 32 {
		return 0, ErrInvalidBuffer
	}

	payloadBytes := bitWidth << 4 // step bitwidths: bw * 16
	forBaseBytes := (1 << forWidth) >> 1

	if !hasExceptions {
		return headerBytes + forBaseBytes + payloadBytes, nil
	}

	if len(buf) < headerBytes+svbLenBytes {
		return 0, ErrInvalidBuffer
	}
	svbLen := int(bo.Uint16(buf[headerBytes:]))

	excIdxSize := excCount
	if excCount > excBitmapThreshold {
		excIdxSize = 16
	}
	return headerBytes + svbLenBytes + forBaseBytes + payloadBytes + excIdxSize + svbLen, nil
}
