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

	forBaseBytes := forBaseBytesLUT[forWidth]
	base := payloadOffset(forBaseBytes, hasExceptions) + utlPayloadBytesLUT[bitWidth]

	if !hasExceptions {
		return base, nil
	}

	// svbLen is always at offset 4 (after header), before FOR base.
	if len(buf) < headerBytes+svbLenBytes {
		return 0, ErrInvalidBuffer
	}
	svbLen := int(bo.Uint16(buf[headerBytes:]))

	return base + excIndexSize(excCount) + svbLen, nil
}
