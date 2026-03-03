package utlpfor

// BlockLength returns the total byte length of the encoded block
// starting at the beginning of buf. Only the first 4-10 bytes of the
// buffer are read, enabling efficient block skipping in MMAP-backed files.
func BlockLength(buf []byte) (int, error) {
	if len(buf) < headerBytes {
		return 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, _, excCount, hasExceptions, _, _, hasFOR := decodeHeader(header)

	if count > blockSize {
		return 0, ErrInvalidBlockLength
	}
	if bitWidth > 32 {
		return 0, ErrInvalidBuffer
	}

	pOff := payloadOffset(hasFOR, hasExceptions)
	base := pOff + utlPayloadBytesLUT[bitWidth]

	if !hasExceptions {
		return base, nil
	}

	svbLen, err := readSVBLen(buf, hasFOR)
	if err != nil {
		return 0, err
	}

	return base + excIndexSize(excCount) + svbLen, nil
}
