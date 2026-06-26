package utlpfor

// ScratchLen64 is the minimum scratch buffer capacity (in uint32 elements)
// for zero-allocation uint64 Pack and Unpack operations.
// Two blocks of blockSize are needed: one for the lower/upper halves work
// area, and one for the per-block uint32 pipeline scratch.
const ScratchLen64 = 2 * blockSize

// splitUint64 decomposes each uint64 value into its lower and upper 32-bit halves.
// lower and upper must each have length >= len(values).
func splitUint64(values []uint64, lower, upper []uint32) {
	for i, v := range values {
		lower[i] = uint32(v)
		upper[i] = uint32(v >> 32)
	}
}

// combineUint64 reconstructs uint64 values from lower and upper 32-bit halves.
// dst must have length >= count; lower and upper must have length >= count.
func combineUint64(dst []uint64, lower, upper []uint32, count int) {
	for i := range count {
		dst[i] = uint64(upper[i])<<32 | uint64(lower[i])
	}
}

// allFitIn32Bits reports whether all values have zero upper 32 bits.
// Uses an OR-accumulator to avoid per-element branching.
func allFitIn32Bits(values []uint64) bool {
	var acc uint64
	for _, v := range values {
		acc |= v
	}
	return acc>>32 == 0
}

// narrowToUint32 copies uint64 values to uint32 by truncation.
// Used for the single-block path where allFitIn32Bits returned true.
// dst must have length >= count; values must have length >= count.
func narrowToUint32(dst []uint32, values []uint64, count int) {
	for i := range count {
		dst[i] = uint32(values[i])
	}
}

// PackUint64 encodes uint64 values into a packed block using the
// double-block strategy: values are split into lower and upper 32-bit
// halves, each encoded as a standard uint32 UTL block.
// If all values fit in 32 bits, a single block is produced.
// If dst has sufficient capacity, it is reused; otherwise a new slice
// is allocated. scratch with capacity >= ScratchLen64 (256) enables
// zero-allocation operation. Pass nil for scratch to use internal
// allocations.
// The values slice may be read but is not modified.
func PackUint64(flag Flag, values []uint64, dst []byte, scratch []uint32) ([]byte, error) {
	if len(scratch) < ScratchLen64 {
		scratch = make([]uint32, ScratchLen64)
	}
	return packUint64Scalar(flag, values, dst, scratch)
}

// packUint64Scalar is the scalar implementation of PackUint64.
//
// Scratch layout: scratch[0:blockSize] holds the current sub-block's values,
// scratch[blockSize:2*blockSize] is passed to packBlockScalar as its exception
// workspace. The two halves never overlap, so no copies are needed.
func packUint64Scalar(flag Flag, values []uint64, dst []byte, scratch []uint32) ([]byte, error) {
	count := len(values)
	if count == 0 || count > blockSize {
		return nil, ErrInvalidBuffer
	}

	off := 0
	if flag&Append != 0 {
		off = len(dst)
	}
	innerFlag := flag &^ Append

	// Fused pass: extract lower halves into scratch[0:count] and check if all
	// values fit in 32 bits via OR-accumulator. Extracting lower (not upper)
	// avoids a second pass in the common single-block case.
	var acc uint64
	for i, v := range values {
		scratch[i] = uint32(v)
		acc |= v
	}

	if acc>>32 == 0 {
		// All values fit in 32 bits: scratch already has the narrowed values.
		block, err := packBlockScalar(innerFlag, scratch[:count], nil, scratch[blockSize:], headerTypeUint64Flag, false)
		if err != nil {
			return nil, err
		}
		totalLen := len(block)
		if off > 0 {
			dst = ensureAppend(dst, off, totalLen)
			copy(dst[off:], block)
		} else {
			dst = block
		}
		return dst[:off+totalLen], nil
	}

	// Two-block path: lower halves are in scratch[0:count].
	// Encode Block 1 (lower) first with combine-with-next flag.
	block1, err := packBlockScalar(innerFlag, scratch[:count], nil, scratch[blockSize:], headerTypeUint64Flag|headerCombineFlag, true)
	if err != nil {
		return nil, err
	}

	// Extract upper halves into scratch[0:count] (lower consumed by packBlockScalar).
	for i, v := range values {
		scratch[i] = uint32(v >> 32)
	}

	// Encode Block 2 (upper).
	block2, err := packBlockScalar(innerFlag, scratch[:count], nil, scratch[blockSize:], headerTypeUint64Flag, false)
	if err != nil {
		return nil, err
	}

	// Write Block 2 length into Block 1 metadata.
	block1Header := bo.Uint32(block1)
	_, _, _, _, forWidth, hasExceptions, _, _, _, _ := decodeHeader(block1Header)
	writeBlock2Len(block1, forBaseBytes(forWidth), hasExceptions, uint16(len(block2)))

	totalLen := len(block1) + len(block2)
	if off > 0 {
		dst = ensureAppend(dst, off, totalLen)
	} else {
		dst = ensureLen(dst, totalLen)
	}
	copy(dst[off:], block1)
	copy(dst[off+len(block1):], block2)
	return dst[:off+totalLen], nil
}

// UnpackUint64 decodes a packed uint64 block into values, using scratch
// as workspace. Returns the populated values slice, the number of bytes
// consumed, and any error.
func UnpackUint64(dst []uint64, scratch []uint32, buf []byte) ([]uint64, int, error) {
	if len(scratch) < ScratchLen64 {
		scratch = make([]uint32, ScratchLen64)
	}
	return unpackUint64Scalar(dst, scratch, buf)
}

// unpackUint64Scalar is the scalar implementation of UnpackUint64.
//
// Scratch layout: scratch[0:blockSize] receives the unpacked uint32 values,
// scratch[blockSize:2*blockSize] is passed to unpackBlockScalarCore as its
// exception workspace. For two-block mode, lower halves are written into dst
// immediately after Block 1 decode, freeing scratch[0:blockSize] for Block 2.
func unpackUint64Scalar(dst []uint64, scratch []uint32, buf []byte) ([]uint64, int, error) {
	if len(buf) < headerBytes {
		return nil, 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	_, _, intType, _, _, _, _, _, _, hasCombine := decodeHeader(header)
	if err := validateIntType64(intType); err != nil {
		return nil, 0, err
	}

	if !hasCombine {
		// Single-block: all values fit in 32 bits, widen to uint64.
		vals, consumed, err := unpackUint32Scalar(scratch[:blockSize], scratch[blockSize:], buf, true)
		if err != nil {
			return nil, 0, err
		}
		count := len(vals)
		if cap(dst) < count {
			dst = make([]uint64, count)
		}
		dst = dst[:count]
		for i, v := range vals {
			dst[i] = uint64(v)
		}
		return dst, consumed, nil
	}

	// Two-block: decode Block 1 (lower), write to dst, then decode Block 2 (upper).
	lowerVals, block1Consumed, err := unpackUint32Scalar(scratch[:blockSize], scratch[blockSize:], buf, true)
	if err != nil {
		return nil, 0, err
	}
	count := len(lowerVals)

	// Write lower halves into dst immediately, freeing scratch for Block 2.
	if cap(dst) < count {
		dst = make([]uint64, count)
	}
	dst = dst[:count]
	for i, v := range lowerVals[:count] {
		dst[i] = uint64(v)
	}

	// Read block2Len from Block 1 metadata to locate Block 2.
	_, _, _, _, forWidth, hasExceptions, _, _, _, _ := decodeHeader(header)
	block2Len := int(readBlock2Len(buf, forBaseBytes(forWidth), hasExceptions))
	block2Start := block1Consumed
	if block2Start+block2Len > len(buf) {
		return nil, 0, ErrInvalidBuffer
	}

	// Decode Block 2 (upper halves), reusing scratch[0:blockSize].
	upperVals, _, err := unpackUint32Scalar(scratch[:blockSize], scratch[blockSize:], buf[block2Start:block2Start+block2Len], true)
	if err != nil {
		return nil, 0, err
	}
	if len(upperVals) != count {
		return nil, 0, ErrInvalidBuffer
	}

	// OR upper halves into dst.
	for i, v := range upperVals[:count] {
		dst[i] |= uint64(v) << 32
	}

	return dst, block1Consumed + block2Len, nil
}

// GetUint64 extracts a single uint64 value at the given position from
// the packed block. scratch with capacity >= ScratchLen64 enables
// zero-allocation operation; pass nil if zero-alloc is not required.
func GetUint64(pos int, buf []byte, scratch []uint32) (uint64, error) {
	return getUint64Scalar(pos, buf, scratch)
}

// getUint64Scalar is the scalar implementation of GetUint64.
func getUint64Scalar(pos int, buf []byte, scratch []uint32) (uint64, error) {
	if len(buf) < headerBytes {
		return 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	_, bitWidth, intType, excCount, forWidth, hasExceptions, _, _, _, hasCombine := decodeHeader(header)
	if err := validateIntType64(intType); err != nil {
		return 0, err
	}

	if !hasCombine {
		// Single-block: all values fit in 32 bits, widen to uint64.
		lower, err := getUint32Scalar(pos, buf, scratch, true)
		if err != nil {
			return 0, err
		}
		return uint64(lower), nil
	}

	// Two-block: extract lower from Block 1, upper from Block 2.
	lower, err := getUint32Scalar(pos, buf, scratch, true)
	if err != nil {
		return 0, err
	}

	// Compute Block 1 length inline from already-decoded header fields.
	forBBytes := forBaseBytes(forWidth)
	block2Len := int(readBlock2Len(buf, forBBytes, hasExceptions))
	block1Len := payloadOffset(forBBytes, hasExceptions, true) + bitWidth<<4
	if hasExceptions {
		if len(buf) < headerBytes+svbLenBytes {
			return 0, ErrInvalidBuffer
		}
		svbLen := int(bo.Uint16(buf[headerBytes:]))
		excIdxSize := excCount
		if excCount > excBitmapThreshold {
			excIdxSize = 16
		}
		block1Len += excIdxSize + svbLen
	}

	block2Start := block1Len
	if block2Start+block2Len > len(buf) {
		return 0, ErrInvalidBuffer
	}

	upper, err := getUint32Scalar(pos, buf[block2Start:block2Start+block2Len], scratch, true)
	if err != nil {
		return 0, err
	}

	return uint64(upper)<<32 | uint64(lower), nil
}

// MaxBlockLength64 returns the maximum byte length of a single packed
// uint64 block for the given encoding flags. This is useful for
// pre-allocating destination buffers to avoid allocations during PackUint64.
//
// The returned value is a conservative upper bound that accounts for the
// worst-case two-block encoding (Block 1 + Block 2).
//
// Flags that affect the worst-case length:
//   - NoPatch: disables exceptions, reducing the maximum length.
//   - NoFOR: disables frame-of-reference, removing FOR base bytes.
//
// Flags that do NOT affect the worst-case length (ignored):
//   - Delta, Special, Append.
func MaxBlockLength64(flag Flag) int {
	return 2*MaxBlockLength32(flag) + block2LenBytes
}
