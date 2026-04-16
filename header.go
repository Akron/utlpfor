package utlpfor

import "encoding/binary"

// bo is a convenience alias for little-endian byte order.
var bo = binary.LittleEndian

const (
	blockSize         = 128
	utlLaneCount      = 16
	utlValuesPerLane  = 8
	utlSuperWordBytes = 64

	headerBytes      = 4
	headerCountBits  = 8
	headerWidthBits  = 5
	headerCountMask  = (1 << headerCountBits) - 1
	headerWidthMask  = (1 << headerWidthBits) - 1
	headerWidthShift = headerCountBits

	headerTypeBits  = 2
	headerTypeMask  = (1 << headerTypeBits) - 1
	headerTypeShift = 13

	// Integer type constants for bits 13-14.
	IntTypeUint8  = 0
	IntTypeUint16 = 1
	IntTypeUint32 = 2
	IntTypeUint64 = 3

	headerTypeUint16Flag = uint32(IntTypeUint16) << headerTypeShift
	headerTypeUint32Flag = uint32(IntTypeUint32) << headerTypeShift

	// Bits 15-16: 2-bit FOR width field.
	forWidthShift = 15
	forWidthMask  = 0x3
	forWidthNone  = 0
	forWidthU8    = 1
	forWidthU16   = 2
	forWidthU32   = 3

	// Bit 17: SPECIAL flag. The current decoder ignores this bit.
	headerSpecialFlag = uint32(1 << 17)

	// Bit 18: reserved. Must be 0 in current implementation.
	headerReservedBitsMask = uint32(1 << 18)

	// Bit 19: combine-with-next (uint64 double-block extension).
	// Must be 0 in current implementation.
	headerCombineFlag = uint32(1 << 19)

	// Bit 20: block-length mode (0=128-value semantics, 1=256-value semantics).
	// Must be 0 in current implementation.
	headerBlock256Flag = uint32(1 << 20)

	// Bit 21: E1 block-length all-exception flag (reserved, silently ignored).
	headerBlock256AllExcFlag = uint32(1 << 21)

	// Bit 22: delta flag.
	headerDeltaFlag = uint32(1 << 22)

	// Bit 23: zigzag flag.
	headerZigZagFlag = uint32(1 << 23)

	headerExcCountBits  = 8
	headerExcCountMask  = (1 << headerExcCountBits) - 1
	headerExcCountShift = 24

	excBitmapThreshold = 16

	// svbLenBytes is the byte size of the StreamVByte length field.
	svbLenBytes = 2

	// headerReservedMask covers bit 18 and bits 19-20 (reserved + extension bits).
	// In the current implementation, all these must be zero.
	// Bit 17 (SPECIAL) and bit 21 (E1 all-exception) are intentionally excluded.
	// Both are currently ignored by the decoder.
	headerReservedMask = headerReservedBitsMask | headerCombineFlag | headerBlock256Flag
)


// utlPayloadBytes returns the UTL payload size in bytes for any bitwidth.
// Formula: ceil(bitWidth / 4) * 64. For step bitwidths (multiples of 4),
// this simplifies to bitWidth * 16 (one shift), but the general formula
// handles all bitwidths for test/validation use.
func utlPayloadBytes(bitWidth int) int { return ((bitWidth + 3) >> 2) << 6 }

// forBaseBytes returns the number of base value bytes for a FOR width code.
// Maps: 0->0, 1->1, 2->2, 3->4 (no FOR, uint8, uint16, uint32).
func forBaseBytes(forWidth int) int { return (1 << forWidth) >> 1 }

// encodeHeader packs count, bitWidth, excCount, and flags into a 32-bit header.
// The bitWidth must be a step bitwidth (0, 4, 8, 12, 16, 20, 24, 28, 32).
// It is stored as a 5-bit step index: encodedValue = bitWidth / 4.
func encodeHeader(count, bitWidth, excCount int, flags uint32) uint32 {
	encodedBW := bitWidth / 4
	h := uint32(count&headerCountMask) |
		(uint32(encodedBW&headerWidthMask) << headerWidthShift) |
		flags
	h |= uint32(excCount&headerExcCountMask) << headerExcCountShift
	return h
}

// decodeHeader extracts fields from a 32-bit header word.
// The 5-bit bitwidth field is decoded as: bitWidth = encodedValue * 4.
// forWidth is the 2-bit FOR width field (bits 15-16): 0=none, 1=u8, 2=u16, 3=u32.
func decodeHeader(header uint32) (count, bitWidth, intType, excCount, forWidth int,
	hasExceptions, hasDelta, hasZigZag, hasSpecial bool) {
	count = int(header & headerCountMask)
	encodedBW := int((header >> headerWidthShift) & headerWidthMask)
	bitWidth = encodedBW * 4
	intType = int((header >> headerTypeShift) & headerTypeMask)
	forWidth = int((header >> forWidthShift) & forWidthMask)
	excCount = int((header >> headerExcCountShift) & headerExcCountMask)
	hasExceptions = excCount > 0
	hasDelta = header&headerDeltaFlag != 0
	hasZigZag = header&headerZigZagFlag != 0
	hasSpecial = header&headerSpecialFlag != 0
	return
}

// Header reads the 4-byte block header from src and returns the decoded fields.
//
// EXPERIMENTAL: This function's return values may change in future versions.
func Header(src []byte) (count, bitWidth, excCount int,
	hasDelta, hasFOR, hasZigZag, hasSpecial bool, err error) {
	if len(src) < headerBytes {
		return 0, 0, 0, false, false, false, false, ErrInvalidBuffer
	}
	header := bo.Uint32(src)
	var intType, forWidth int
	var hasExceptions bool
	count, bitWidth, intType, excCount, forWidth, hasExceptions, hasDelta, hasZigZag, hasSpecial = decodeHeader(header)
	_ = hasExceptions
	if err = validateIntType(intType); err != nil {
		return 0, 0, 0, false, false, false, false, err
	}
	hasFOR = forWidth > 0
	return
}

// payloadOffset returns the byte offset where the UTL payload begins.
// forBaseBytes is the number of bytes for the FOR base value (0, 1, 2, or 4).
func payloadOffset(forBaseBytes int, hasExceptions bool) int {
	offset := headerBytes
	if hasExceptions {
		offset += svbLenBytes
	}
	offset += forBaseBytes
	return offset
}

// validateIntType checks that the integer type in the header is supported.
// IntTypeUint32 and IntTypeUint16 are accepted; others return ErrUnsupportedType.
func validateIntType(intType int) error {
	switch intType {
	case IntTypeUint32, IntTypeUint16:
		return nil
	default:
		return ErrUnsupportedType
	}
}
