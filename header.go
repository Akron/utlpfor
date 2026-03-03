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
	headerWidthBits  = 6
	headerCountMask  = (1 << headerCountBits) - 1
	headerWidthMask  = (1 << headerWidthBits) - 1
	headerWidthShift = headerCountBits

	headerTypeBits  = 2
	headerTypeMask  = (1 << headerTypeBits) - 1
	headerTypeShift = headerWidthShift + headerWidthBits

	// Integer type constants for bits 14-15.
	IntTypeUint8  = 0
	IntTypeUint16 = 1
	IntTypeUint32 = 2
	IntTypeUint64 = 3

	headerTypeUint16Flag = uint32(IntTypeUint16) << headerTypeShift
	headerTypeUint32Flag = uint32(IntTypeUint32) << headerTypeShift

	// Bit 18: SPECIAL flag -- modifies interpretation of other header fields
	// in specific combinations (e.g. full-block all-exceptions in 256-mode).
	// Silently ignored in current implementation.
	headerSpecialFlag = uint32(1 << 18)

	// Bit 19: combine-with-next (uint64 double-block extension).
	// Must be 0 in current implementation.
	headerCombineFlag = uint32(1 << 19)

	// Bit 20: block-length mode (0=128-value semantics, 1=256-value semantics).
	// Must be 0 in current implementation.
	headerBlock256Flag = uint32(1 << 20)

	// Bit 21: FOR flag (frame-of-reference).
	// Must be 0 in current implementation.
	headerFORFlag = uint32(1 << 21)

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

	// headerFORBytes is the byte size of the FOR base value.
	headerFORBytes = 4

	// headerReservedMask covers bits 19-21 (extension bits).
	// In the current implementation, all these must be zero.
	// Bit 18 (SPECIAL) is intentionally excluded -- it is silently ignored.
	headerReservedMask = headerCombineFlag | headerBlock256Flag | headerFORFlag
)

// utlPayloadBytesLUT maps bit width (0-32) to UTL payload size in bytes.
// Formula: ceil(8 * bitWidth / 32) * 64 = ceil(bitWidth/4) * 64.
var utlPayloadBytesLUT = [33]int{
	0,
	64, 64, 64, 64,
	128, 128, 128, 128,
	192, 192, 192, 192,
	256, 256, 256, 256,
	320, 320, 320, 320,
	384, 384, 384, 384,
	448, 448, 448, 448,
	512, 512, 512, 512,
}

// encodeHeader packs count, bitWidth, excCount, and flags into a 32-bit header.
func encodeHeader(count, bitWidth, excCount int, flags uint32) uint32 {
	h := uint32(count&headerCountMask) |
		(uint32(bitWidth&headerWidthMask) << headerWidthShift) |
		flags
	h |= uint32(excCount&headerExcCountMask) << headerExcCountShift
	return h
}

// decodeHeader extracts fields from a 32-bit header word.
func decodeHeader(header uint32) (count, bitWidth, intType, excCount int,
	hasExceptions, hasDelta, hasZigZag, hasFOR bool) {
	count = int(header & headerCountMask)
	bitWidth = int((header >> headerWidthShift) & headerWidthMask)
	intType = int((header >> headerTypeShift) & headerTypeMask)
	excCount = int((header >> headerExcCountShift) & headerExcCountMask)
	hasExceptions = excCount > 0
	hasDelta = header&headerDeltaFlag != 0
	hasZigZag = header&headerZigZagFlag != 0
	hasFOR = header&headerFORFlag != 0
	return
}

// payloadOffset returns the byte offset where the UTL payload begins.
func payloadOffset(hasFOR, hasExceptions bool) int {
	offset := headerBytes
	if hasFOR {
		offset += headerFORBytes
	}
	if hasExceptions {
		offset += svbLenBytes
	}
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

// blockBytesConsumed computes total bytes consumed by a block including exceptions.
func blockBytesConsumed(header uint32, svbLen int) int {
	_, bitWidth, _, excCount, hasExceptions, _, _, hasFOR := decodeHeader(header)
	base := payloadOffset(hasFOR, hasExceptions) + utlPayloadBytesLUT[bitWidth]
	if !hasExceptions {
		return base
	}
	excIndexSize := min(excCount, excBitmapThreshold)
	return base + excIndexSize + svbLen
}
