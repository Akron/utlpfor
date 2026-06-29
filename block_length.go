package utlpfor

// BlockLength returns the total byte length of the encoded block
// starting at the beginning of buf. Only the first few bytes of the
// buffer are read, enabling efficient block skipping in MMAP-backed files.
// For uint64 blocks with combine-with-next set, the returned length
// includes both Block 1 and Block 2.
func BlockLength(src []byte) (int, error) {
	if len(src) < headerBytes {
		return 0, ErrInvalidBuffer
	}

	header := bo.Uint32(src)
	count, bitWidth, intType, excCount, forWidth, hasExceptions, _, _, _, hasCombine := decodeHeader(header)

	if count > blockSize {
		return 0, ErrInvalidBlockLength
	}
	if bitWidth > 32 {
		return 0, ErrInvalidBuffer
	}

	// The combine flag is only valid for uint64 blocks.
	hasCombine = hasCombine && intType == IntTypeUint64

	// Validate reserved/extension bits. Uint64 blocks allow the combine flag
	// (bit 19) but must still reject reserved (bit 18) and block-256 (bit 20).
	if intType == IntTypeUint64 {
		if header&(headerReservedBitsMask|headerBlock256Flag) != 0 {
			return 0, ErrInvalidFlags
		}
	} else if header&headerReservedMask != 0 {
		return 0, ErrInvalidFlags
	}

	// FOR base size is context-dependent: forWidth=3 with uint64 single-block
	// means 8-byte FOR64 base instead of the standard 4-byte uint32 base.
	forBBytes := forBaseBytesForBlock(forWidth, intType, hasCombine)

	pOff := payloadOffset(forBBytes, hasExceptions, hasCombine)
	block1Len := pOff + bitWidth<<4

	if hasExceptions {
		if len(src) < headerBytes+svbLenBytes {
			return 0, ErrInvalidBuffer
		}
		svbLen := int(bo.Uint16(src[headerBytes:]))
		excIdxSize := excCount
		if excCount > excBitmapThreshold {
			excIdxSize = 16
		}
		block1Len += excIdxSize + svbLen
	}

	if !hasCombine {
		return block1Len, nil
	}

	b2Len := int(readBlock2Len(src, forBBytes, hasExceptions))
	return block1Len + b2Len, nil
}
