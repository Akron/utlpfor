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
// Returns ErrInvalidBuffer when the buffer is too short to hold the field.
func readSVBLen(buf []byte) (int, error) {
	if len(buf) < headerBytes+svbLenBytes {
		return 0, ErrInvalidBuffer
	}
	return int(bo.Uint16(buf[headerBytes:])), nil
}

// decodeExceptionHighBitsInto decodes all SVB exception high bits into dst.
// excStart is the byte offset where the exception index begins (after payload).
// The control bytes are validated to describe at most svbLen data bytes, so
// the decoders never read past the declared SVB region on corrupt input.
// Returns the offset past the SVB data and any error.
func decodeExceptionHighBitsInto(dst []uint32, buf []byte, excStart, excCount int) (int, error) {
	// readSVBLen and the svbStart+svbLen bound check are technically redundant
	// for all current callers (the unpack prologues already validate the full
	// exception region). They are kept as defense-in-depth: if a future caller
	// omits buffer validation, this function still won't panic. The cost is a
	// single always-predicted branch per block.
	svbLen, err := readSVBLen(buf)
	if err != nil {
		return 0, err
	}
	svbStart := excStart + excIndexSize(excCount)
	if svbStart+svbLen > len(buf) {
		return 0, ErrInvalidBuffer
	}
	// Validate that the control bytes don't describe more data bytes than
	// svbLen holds; otherwise the StreamVByte decoder would read past the
	// declared region and panic on a corrupt block.
	//
	// This per-control-byte scan is necessary - simpler bounds won't work:
	//   * Upper-bound check (ctrlCount + 4*excCount <= svbLen) is too strict:
	//     it rejects valid blocks whose exception high-bits encode in 1-2 bytes,
	//     because the actual needed bytes are far below the 4-bytes-per-value
	//     maximum.
	//   * Lower-bound check (ctrlCount + excCount <= svbLen) is too weak:
	//     it misses corrupt control bytes that individually claim more data
	//     bytes than the region holds.
	//
	// Only slots that decoders actually read are counted: all slots of
	// complete groups, plus slots 0..k-1 of the trailing partial group (the
	// encoder zero-pads the unused slots, so they claim no data bytes).
	// Cost: at most ceil(excCount/4) = 32 byte reads for a full block.
	ctrlCount := svbControlByteCount(excCount)
	if ctrlCount > svbLen {
		return 0, ErrInvalidBuffer
	}
	ctrlBytes := buf[svbStart : svbStart+ctrlCount]
	needed := ctrlCount
	for g := range excCount / 4 {
		needed += int(svbControlBlockSizeLUT[ctrlBytes[g]])
	}
	if k := excCount % 4; k > 0 {
		ctrl := ctrlBytes[excCount/4]
		for s := range k {
			needed += int((ctrl>>(s*2))&0x03) + 1
		}
	}
	if needed > svbLen {
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
// All index regions are validated before use; corrupt blocks yield
// ErrInvalidBuffer instead of out-of-range reads.
// Returns the total number of bytes consumed and any error.
func applyExceptions(dst []uint32, buf []byte, excStart, count, bitWidth, excCount int, scratch []uint32) (int, error) {
	if excCount > count {
		// NOT defense-in-depth: the unpack prologues do not validate
		// excCount vs count. A corrupt header with excCount > count
		// would panic on scratch[:excCount] below (scratch has capacity
		// blockSize = 128, excCount can be up to 255 from the header).
		return 0, ErrInvalidBuffer
	}
	decodeBuf := scratch[:excCount]

	consumed, err := decodeExceptionHighBitsInto(decodeBuf, buf, excStart, excCount)
	if err != nil {
		return 0, err
	}
	shift := uint(bitWidth)

	if excCount <= excBitmapThreshold {
		excPos := buf[excStart : excStart+excCount]
		if count == blockSize {
			// Fast path for full blocks (count == 128): valid positions are 0-127.
			// The &0x7F mask serves two purposes:
			//   1. Bounds check elimination (BCE): the compiler proves
			//      excPos[i]&0x7F < 128 == len(dst) at compile time, removing
			//      the runtime bounds check from this hot loop.
			//   2. Defense-in-depth: a corrupt position byte > 127 cannot index
			//      past dst without an explicit branch.
			for i := range excCount {
				dst[excPos[i]&0x7F] |= decodeBuf[i] << shift
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
// decodeBuf must cover every set bit of the masked bitmap; the caller
// guarantees count > excBitmapThreshold, which bounds the set-bit count.
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

	// Pre-check: the number of set bitmap bits must not exceed decodeBuf.
	// For valid blocks they are equal; for corrupt bitmaps this prevents
	// overrunning decodeBuf. Two OnesCount64 calls (single instruction
	// each on amd64) replace N per-iteration branch checks.
	if bits.OnesCount64(word0)+bits.OnesCount64(word1) > len(decodeBuf) {
		return
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
		dst[64+bitPos] |= decodeBuf[excIdx]
		excIdx++
		word1 &= word1 - 1
	}
}

// findExceptionIndex finds the index of a position in the exception list.
// For sorted positions (excCount <= 16): linear scan with early exit.
// For bitmap (excCount > 16): bit test + popcount.
// excIdx is the pre-sliced exception index region (positions or bitmap);
// reads are clamped to len(excIdx), so corrupt counts degrade to "not
// found" instead of panicking.
// Returns -1 if the position is not an exception.
func findExceptionIndex(excIdx []byte, excCount, pos int) int {
	if excCount <= excBitmapThreshold {
		// min: defense-in-depth — current caller (getValueDirect) validates
		// the region, so len(excIdx) >= excCount; the clamp prevents a panic
		// if a future caller passes an under-sized slice.
		n := min(excCount, len(excIdx))
		for i := range n {
			if excIdx[i] == byte(pos) {
				return i
			}
			if excIdx[i] > byte(pos) {
				return -1
			}
		}
		return -1
	}
	// min: defense-in-depth - see comment on the position-list path above.
	bitmap := excIdx[:min(16, len(excIdx))]
	if len(bitmap) == 0 {
		return -1
	}
	byteIdx := pos / 8
	bitIdx := pos % 8
	if byteIdx >= len(bitmap) {
		return -1
	}
	if bitmap[byteIdx]&(1<<bitIdx) == 0 {
		return -1
	}
	// Rank = popcount(bits before pos); this maps bitmap position to SVB
	// index. 2 POPCNTs replace the former byte-wise OnesCount8 loop.
	return bitmapRank128(bitmap, byteIdx, uint(bitIdx))
}
