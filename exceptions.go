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
// svbLen is always at offset 4 (immediately after the header), regardless of FOR.
func readSVBLen(buf []byte) (int, error) {
	return int(bo.Uint16(buf[headerBytes:])), nil
}

// decodeExceptionHighBitsInto decodes all SVB exception high bits into dst.
// excStart is the byte offset where the exception index begins (after payload).
// Returns the offset past the SVB data and any error.
func decodeExceptionHighBitsInto(dst []uint32, buf []byte, excStart, excCount int) (int, error) {
	svbLen, err := readSVBLen(buf)
	if err != nil {
		return 0, err
	}
	svbStart := excStart + excIndexSize(excCount)
	if svbStart+svbLen > len(buf) {
		return 0, ErrInvalidBuffer
	}
	// High bits are serialized immediately after exception index bytes.
	// Keeping this layout fixed avoids format branching in pack/unpack paths.
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

// encodeSVBIntoDst encodes exception high bits directly into a pre-allocated
// byte buffer. Returns the number of bytes written.
func encodeSVBIntoDst(dst []byte, highBits []uint32) int {
	svbData := streamvbyte.EncodeUint32(highBits,
		&streamvbyte.EncodeOptions[uint32]{Buffer: dst})
	return len(svbData)
}

// maxSVBEncodedLen returns the maximum encoded size for count SVB values.
func maxSVBEncodedLen(count int) int {
	return streamvbyte.MaxEncodedLen(count)
}

// collectAndWriteExceptions finds exception values and writes the exception
// index directly into excIdxDst, avoiding intermediate stack buffers and
// the separate writeExceptionIndex copy. For the bitmap path (excCount > 16),
// this also reduces from 2 passes to 1 pass over values.
// excCount must match the actual number of exceptions (from selectBitWidth).
func collectAndWriteExceptions(values []uint32, bitWidth int,
	excIdxDst []byte, excCount int, highBits []uint32) {
	if bitWidth >= 32 {
		return
	}
	mask := uint32((1 << bitWidth) - 1)
	if excCount <= excBitmapThreshold {
		n := 0
		for i, v := range values {
			if v > mask {
				// Small-exception mode stores sorted absolute positions.
				// This keeps Get() lookup cheap via early-exit linear scan.
				excIdxDst[n] = byte(i)
				highBits[n] = v >> bitWidth
				n++
			}
		}
	} else {
		// Build the 128-bit exception bitmap in registers first, then write once.
		// This reduces repeated byte stores in the hot loop.
		var word0, word1 uint64
		for i, v := range values {
			if v <= mask {
				continue
			}
			if i < 64 {
				word0 |= uint64(1) << i
			} else {
				word1 |= uint64(1) << (i - 64)
			}
		}
		bo.PutUint64(excIdxDst, word0)
		bo.PutUint64(excIdxDst[8:], word1)

		// Iterate set bits in ascending index order to match bitmap rank order.
		// SVB highBits must follow this exact order for correct reconstruction.
		n := 0
		for w := word0; w != 0; w &= w - 1 {
			i := bits.TrailingZeros64(w)
			highBits[n] = values[i] >> bitWidth
			n++
		}
		for w := word1; w != 0; w &= w - 1 {
			i := 64 + bits.TrailingZeros64(w)
			highBits[n] = values[i] >> bitWidth
			n++
		}
	}
}

// applyExceptions patches decoded values with exception high bits.
// excStart is the byte offset where the exception index begins (after payload).
// scratch is used as a decode buffer (must have capacity >= excCount).
// Returns the total number of bytes consumed and any error.
func applyExceptions(dst []uint32, buf []byte, excStart, count, bitWidth, excCount int, scratch []uint32) (int, error) {
	decodeBuf := scratch[:excCount]

	consumed, err := decodeExceptionHighBitsInto(decodeBuf, buf, excStart, excCount)
	if err != nil {
		return 0, err
	}
	shift := uint(bitWidth)

	if excCount <= excBitmapThreshold {
		excPos := buf[excStart : excStart+excCount]
		if count == blockSize {
			// Fast path: packed positions are guaranteed in-range for full blocks.
			for i := range excCount {
				dst[excPos[i]] |= decodeBuf[i] << shift
			}
		} else {
			for i := range excCount {
				pos := int(excPos[i])
				if pos < count {
					dst[pos] |= decodeBuf[i] << shift
				}
			}
		}
	} else {
		// Pre-shift once; bitmap apply can then OR without extra per-hit shifts.
		for i := range excCount {
			decodeBuf[i] <<= shift
		}
		bitmap := buf[excStart : excStart+16]
		applyBitmapExceptions(dst, decodeBuf, bitmap, count)
	}
	return consumed, nil
}

// applyBitmapExceptions patches decoded values using a 128-bit exception bitmap.
// It iterates only set bits, avoiding a full scan over all positions.
func applyBitmapExceptions(dst []uint32, decodeBuf []uint32, bitmap []byte, count int) {
	excIdx := 0

	word0 := bo.Uint64(bitmap)
	word1 := bo.Uint64(bitmap[8:])
	if count < blockSize {
		// Last partial block may carry stale high bits in trailing bitmap region.
		// Mask them so exception rank and decoded values stay aligned.
		if count < 64 {
			word1 = 0
			word0 &= (uint64(1) << count) - 1
		} else {
			word1 &= (uint64(1) << (count - 64)) - 1
		}
	}

	for word0 != 0 {
		// w &= w-1 drops one set bit, so cost scales with exceptions, not 128 slots.
		bitPos := bits.TrailingZeros64(word0)
		dst[bitPos] |= decodeBuf[excIdx]
		excIdx++
		word0 &= word0 - 1
	}

	for word1 != 0 {
		bitPos := bits.TrailingZeros64(word1)
		pos := 64 + bitPos
		dst[pos] |= decodeBuf[excIdx]
		excIdx++
		word1 &= word1 - 1
	}
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
	// Rank = popcount(bits before pos); this maps bitmap position to SVB index.
	for b := range byteIdx {
		count += bits.OnesCount8(bitmap[b])
	}
	count += bits.OnesCount8(bitmap[byteIdx] & ((1 << bitIdx) - 1))
	return count
}
