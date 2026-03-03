package utlpfor

import (
	"math/bits"

	"github.com/mhr3/streamvbyte"
)

// excIndexSize returns the byte size of the exception index for a given count.
func excIndexSize(excCount int) int {
	if excCount <= excBitmapThreshold {
		return excCount
	}
	return 16
}

// readSVBLen reads the StreamVByte data length from the block buffer.
func readSVBLen(buf []byte, hasFOR bool) (int, error) {
	offset := headerBytes
	if hasFOR {
		offset += headerFORBytes
	}
	if len(buf) < offset+svbLenBytes {
		return 0, ErrInvalidBuffer
	}
	return int(bo.Uint16(buf[offset:])), nil
}

// decodeExceptionHighBitsInto decodes all SVB exception high bits into dst.
// excStart is the byte offset where the exception index begins (after payload).
// Returns the offset past the SVB data and any error.
func decodeExceptionHighBitsInto(dst []uint32, buf []byte, excStart, excCount int, hasFOR bool) (int, error) {
	svbLen, err := readSVBLen(buf, hasFOR)
	if err != nil {
		return 0, err
	}
	svbStart := excStart + excIndexSize(excCount)
	if svbStart+svbLen > len(buf) {
		return 0, ErrInvalidBuffer
	}
	streamvbyte.DecodeUint32(
		buf[svbStart:svbStart+svbLen], excCount,
		&streamvbyte.DecodeOptions[uint32]{Buffer: dst[:excCount]},
	)
	return svbStart + svbLen, nil
}

// collectExceptionsDirect finds values that exceed the chosen bit width.
// Returns the exception count. Fills positions (for <=16 exceptions)
// or bitmap (for >16 exceptions) and highBits.
func collectExceptionsDirect(values []uint32, bitWidth int,
	positions []byte, bitmap []byte, highBits []uint32) int {
	if bitWidth >= 32 {
		return 0
	}
	mask := uint32((1 << bitWidth) - 1)
	n := 0
	for i, v := range values {
		if v > mask {
			if n < len(positions) {
				positions[n] = byte(i)
			}
			highBits[n] = v >> bitWidth
			n++
		}
	}
	if n > excBitmapThreshold {
		clear(bitmap[:16])
		for i, v := range values {
			if v > mask {
				bitmap[i/8] |= 1 << (i % 8)
			}
		}
	}
	return n
}

// writeExceptionsDirect writes exception index and pre-encoded SVB data
// after the payload. Returns the number of bytes written.
func writeExceptionsDirect(dst []byte, positions []byte, bitmap []byte,
	excCount int, svbData []byte) int {
	offset := 0
	if excCount <= excBitmapThreshold {
		copy(dst[offset:], positions[:excCount])
		offset += excCount
	} else {
		copy(dst[offset:], bitmap[:16])
		offset += 16
	}
	copy(dst[offset:], svbData)
	offset += len(svbData)
	return offset
}

// encodeExceptionHighBits encodes the high bits of exception values
// using StreamVByte. Returns the encoded byte slice.
func encodeExceptionHighBits(highBits []uint32) []byte {
	return streamvbyte.EncodeUint32(highBits, nil)
}

// applyExceptions patches decoded values with exception high bits.
// excStart is the byte offset where the exception index begins (after payload).
// scratch is used as a decode buffer (must have capacity >= excCount).
// Returns the total number of bytes consumed and any error.
func applyExceptions(dst []uint32, buf []byte, excStart, count, bitWidth, excCount int, hasFOR bool, scratch []uint32) (int, error) {
	// TODO-PERF: scratch should always be large enough
	var decodeBuf []uint32
	if len(scratch) >= excCount {
		decodeBuf = scratch[:excCount]
	} else {
		decodeBuf = make([]uint32, excCount)
	}

	consumed, err := decodeExceptionHighBitsInto(decodeBuf, buf, excStart, excCount, hasFOR)
	if err != nil {
		return 0, err
	}

	if excCount <= excBitmapThreshold {
		for i := range excCount {
			pos := int(buf[excStart+i])
			if pos < count {
				dst[pos] |= decodeBuf[i] << bitWidth
			}
		}
	} else {
		bitmap := buf[excStart : excStart+16]
		excIdx := 0
		for pos := range count {
			if bitmap[pos/8]&(1<<(pos%8)) != 0 {
				dst[pos] |= decodeBuf[excIdx] << bitWidth
				excIdx++
			}
		}
	}
	return consumed, nil
}

// findExceptionIndex finds the index of a position in the exception list.
// For sorted positions (excCount <= 16): linear scan with early exit.
// For bitmap (excCount > 16): bit test + popcount.
// Returns -1 if the position is not an exception.
func findExceptionIndex(buf []byte, excStart, excCount, pos int) int {
	if excCount <= excBitmapThreshold {
		for i := range excCount {
			if buf[excStart+i] == byte(pos) {
				return i
			}
			if buf[excStart+i] > byte(pos) {
				return -1
			}
		}
		return -1
	}
	bitmap := buf[excStart : excStart+16]
	byteIdx := pos / 8
	bitIdx := pos % 8
	if bitmap[byteIdx]&(1<<bitIdx) == 0 {
		return -1
	}
	count := 0
	for b := range byteIdx {
		count += bits.OnesCount8(bitmap[b])
	}
	count += bits.OnesCount8(bitmap[byteIdx] & ((1 << bitIdx) - 1))
	return count
}
