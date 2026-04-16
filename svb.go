package utlpfor

// svbControlByteCount returns the number of control bytes for count values.
func svbControlByteCount(count int) int { return (count + 3) >> 2 }

// svbControlBlockSizeLUT maps each control byte (256 entries) to its
// total data byte count. Entry i = sum of (code+1) for each of the
// 4 two-bit codes packed in byte i.
var svbControlBlockSizeLUT [256]byte

func init() {
	for i := range 256 {
		var size byte
		c := byte(i)
		size += (c & 0x03) + 1
		c >>= 2
		size += (c & 0x03) + 1
		c >>= 2
		size += (c & 0x03) + 1
		c >>= 2
		size += (c & 0x03) + 1
		svbControlBlockSizeLUT[i] = size
	}
}

// svbControlBlockSize returns the total data byte count for a single
// control byte (4 values).
func svbControlBlockSize(ctrl byte) int {
	return int(svbControlBlockSizeLUT[ctrl])
}

// svbDecodeOneInternal decodes a single StreamVByte value at the given
// index from encoded data. Scans control bytes to find the target
// value's data offset, then reads only the relevant 1-4 bytes.
func svbDecodeOneInternal(encoded []byte, count, index int) uint32 {
	cbCount := svbControlByteCount(count)
	dataOff := cbCount
	targetCB := index >> 2
	targetSlot := index & 3

	for cb := range targetCB {
		dataOff += svbControlBlockSize(encoded[cb])
	}

	ctrl := encoded[targetCB]
	for s := range targetSlot {
		dataOff += int((ctrl>>(s*2))&0x03) + 1
	}

	code := (ctrl >> (targetSlot * 2)) & 0x03
	byteLen := int(code) + 1
	return svbReadValue(encoded, dataOff, byteLen)
}

// svbReadValue reads 1-4 little-endian bytes from data[off:].
func svbReadValue(data []byte, off, byteLen int) uint32 {
	switch byteLen {
	case 1:
		return uint32(data[off])
	case 2:
		return uint32(bo.Uint16(data[off:]))
	case 3:
		return uint32(data[off]) | uint32(data[off+1])<<8 | uint32(data[off+2])<<16
	default:
		return bo.Uint32(data[off:])
	}
}

// svbDataSlice returns the SVB encoded data slice from the block buffer.
// The SVB data starts after the exception index (sorted positions or bitmap).
func svbDataSlice(buf []byte, excStart, excCount int) []byte {
	svbLen, _ := readSVBLen(buf)
	svbStart := excStart + excIndexSize(excCount)
	return buf[svbStart : svbStart+svbLen]
}
