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
// encoded must contain the control bytes plus the data bytes the control
// bytes describe; reads are clamped to len(encoded), so corrupt control
// bytes degrade to wrong values instead of panicking. Callers that
// require structural correctness (valid control-byte sums) must validate
// the region before calling; see decodeExceptionHighBitsInto.
func svbDecodeOneInternal(encoded []byte, count, index int) uint32 {
	if count <= 0 || index < 0 || index >= count {
		return 0
	}
	cbCount := svbControlByteCount(count)
	dataOff := cbCount
	targetCB := index >> 2
	targetSlot := index & 3

	for cb := range targetCB {
		if cb >= len(encoded) {
			return 0
		}
		dataOff += svbControlBlockSize(encoded[cb])
	}

	if targetCB >= len(encoded) {
		return 0
	}
	ctrl := encoded[targetCB]
	for s := range targetSlot {
		dataOff += int((ctrl>>(s*2))&0x03) + 1
	}

	code := (ctrl >> (targetSlot * 2)) & 0x03
	byteLen := int(code) + 1
	return svbReadValue(encoded, dataOff, byteLen)
}

// svbReadValue reads 1-4 little-endian bytes from data[off:]. Reads past
// the end of data are clamped to the available bytes, so callers working
// on corrupt input observe garbage values instead of a panic.
func svbReadValue(data []byte, off, byteLen int) uint32 {
	if off >= len(data) {
		return 0
	}
	end := min(off+byteLen, len(data))
	if end-off == byteLen {
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
	// Clamp: consume the remaining bytes, zero-fill the rest.
	var acc uint32
	for i := off; i < end; i++ {
		acc |= uint32(data[i]) << ((i - off) * 8)
	}
	return acc
}

// svbDataSlice returns the SVB encoded data slice from the block buffer.
// The SVB data starts after the exception index (sorted positions or bitmap).
// The returned slice is clamped to the buffer, so it is always safe to
// read; callers that require structural correctness must validate svbLen
// against the buffer length and reject with ErrInvalidBuffer beforehand.
func svbDataSlice(buf []byte, excStart, excCount int) []byte {
	svbLen, _ := readSVBLen(buf)
	svbStart := excStart + excIndexSize(excCount)
	if svbStart >= len(buf) {
		return nil
	}
	return buf[svbStart:min(svbStart+svbLen, len(buf))]
}
