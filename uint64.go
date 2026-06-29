package utlpfor

// ScratchLen64 is the minimum scratch buffer capacity (in uint32 elements)
// for zero-allocation uint64 Pack and Unpack operations.
//
// Three regions of blockSize (128) uint32 elements each:
//
//	scratch[0·blockSize : 1·blockSize]  – Block 1 values (lower halves / FOR64 work)
//	scratch[1·blockSize : 2·blockSize]  – exception workspace for current block
//	scratch[2·blockSize : 3·blockSize]  – Block 2 values (upper halves)
//
// Keeping lower and upper halves in separate, non-overlapping regions
// lets the two-block unpack path combine them in a single fused loop
// (dst[i] = upper[i]<<32 | lower[i]) instead of two passes.
const ScratchLen64 = 3 * blockSize

// for64BaseSize is the byte size of a FOR64 base value.
const for64BaseSize = 8

// ensureCapacity64 ensures dst has cap >= off+MaxBlockLength64(flag),
// preserving dst[:off]. Called once at the packUint64Scalar entry point
// so all inner pack paths can unconditionally trust the capacity.
// On the hot path (pre-allocated dst), this is a single comparison that
// the branch predictor always predicts correctly.
func ensureCapacity64(dst []byte, off int, flag Flag) []byte {
	needed := MaxBlockLength64(flag)
	if cap(dst)-off >= needed {
		return dst
	}
	grown := make([]byte, off, off+needed)
	copy(grown, dst[:off])
	return grown
}

// forSubtract64 subtracts base from each uint64 value and stores the
// result as uint32. Callers must ensure (values[i] - base) fits in 32 bits.
func forSubtract64(dst []uint32, values []uint64, base uint64) {
	for i, v := range values {
		dst[i] = uint32(v - base)
	}
}

// forAdd64 adds a uint64 base to each uint32 value and stores as uint64.
func forAdd64(dst []uint64, values []uint32, base uint64, count int) {
	for i := range count {
		dst[i] = uint64(values[i]) + base
	}
}

// isFor64SingleBlock reports whether the header describes a FOR64
// single-block (IntType=Uint64, combine=0, forWidth=3).
func isFor64SingleBlock(intType, forWidth int, hasCombine bool) bool {
	return intType == IntTypeUint64 && !hasCombine && forWidth == forWidthU32
}

// forBaseBytesForBlock returns the FOR base byte size, accounting for
// the FOR64 context: forWidth=3 with IntType=Uint64 in single-block mode
// yields an 8-byte FOR64 base instead of the standard 4-byte uint32 base.
func forBaseBytesForBlock(forWidth, intType int, hasCombine bool) int {
	if intType == IntTypeUint64 && !hasCombine && forWidth == forWidthU32 {
		return for64BaseSize
	}
	return forBaseBytes(forWidth)
}

// PackUint64 encodes uint64 values into a packed block using the
// double-block strategy: values are split into lower and upper 32-bit
// halves, each encoded as a standard uint32 UTL block.
// If all values fit in 32 bits, a single block is produced.
// If dst has sufficient capacity, it is reused; otherwise a new slice
// is allocated. scratch with capacity >= ScratchLen64 (384) enables
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
// Scratch layout (3·blockSize = 384 uint32 slots):
//
//	scratch[0 : blockSize]          – lower halves (block 1 values)
//	scratch[blockSize : 2·blockSize] – exception workspace (shared by both blocks)
//	scratch[2·blockSize : 3·blockSize] – upper halves (block 2 values)
//
// The fused extraction loop fills both lower and upper regions in a
// single pass, so the two-block path never needs a separate upper-half
// extraction loop.
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

	// Single capacity check: all encoding paths (single-block, FOR64,
	// two-block) fit within MaxBlockLength64. On the hot path with a
	// pre-allocated dst this is one always-true comparison.
	dst = ensureCapacity64(dst, off, innerFlag)

	// Fused pass: extract lower halves into scratch[0:count] and upper
	// halves into scratch[2·blockSize:], track min/max, and check if all
	// values fit in 32 bits via OR-accumulator.
	upper := scratch[2*blockSize:]
	var acc uint64
	min64, max64 := values[0], values[0]
	for i, v := range values {
		scratch[i] = uint32(v)
		upper[i] = uint32(v >> 32)
		acc |= v
		min64 = min(min64, v)
		max64 = max(max64, v)
	}

	// FOR64 gateway: if range fits in 32 bits and min > 0, encode as
	// single block with 8-byte FOR64 base.
	if innerFlag&NoFOR == 0 && min64 > 0 && (max64-min64) < (1<<32) {
		return packFor64SingleBlock(innerFlag, values, dst, scratch, off, count, min64)
	}

	if acc>>32 == 0 {
		// All values fit in 32 bits: scratch already has the narrowed values.
		// Capacity is guaranteed, so packBlockScalar writes directly into dst.
		block, err := packBlockScalar(innerFlag, scratch[:count], dst[off:off], scratch[blockSize:2*blockSize], headerTypeUint64Flag, false)
		if err != nil {
			return nil, err
		}
		return dst[:off+len(block)], nil
	}

	// Two-block path: capacity is guaranteed, both blocks pack directly
	// into contiguous dst memory without any allocations or copies.

	// Block 1 (lower halves, already in scratch[0:count]).
	block1, err := packBlockScalar(innerFlag, scratch[:count], dst[off:off], scratch[blockSize:2*blockSize], headerTypeUint64Flag|headerCombineFlag, true)
	if err != nil {
		return nil, err
	}
	block1Len := len(block1)

	// Block 2 (upper halves, already in scratch[2·blockSize:]).
	b2Off := off + block1Len
	dst = dst[:b2Off]
	block2, err := packBlockScalar(innerFlag, upper[:count], dst[b2Off:b2Off], scratch[blockSize:2*blockSize], headerTypeUint64Flag, false)
	if err != nil {
		return nil, err
	}
	block2Len := len(block2)

	// Write Block 2 length into Block 1 metadata.
	block1Header := bo.Uint32(block1)
	_, _, _, _, forWidth, hasExceptions, _, _, _, _ := decodeHeader(block1Header)
	writeBlock2Len(block1, forBaseBytes(forWidth), hasExceptions, uint16(block2Len))

	return dst[:off+block1Len+block2Len], nil
}

// packFor64SingleBlock encodes a uint64 block using the FOR64 single-block
// strategy: subtract min64 from all values to produce uint32 work values,
// pack via packBlockScalar (reusing all delta/exception/packing logic),
// then copy the small header metadata to the front and write the FOR64 base.
func packFor64SingleBlock(flag Flag, values []uint64, dst []byte, scratch []uint32,
	off, count int, min64 uint64) ([]byte, error) {

	work := scratch[:count]
	forSubtract64(work, values, min64)

	// Capacity is guaranteed by the caller (ensureCapacity64 at entry).
	// Pack inner block at off+for64BaseSize so the payload already sits
	// at its final position; only the header metadata needs relocating.
	innerOff := off + for64BaseSize
	dst = dst[:innerOff]
	inner, err := packBlockScalar(flag|NoFOR, work, dst[innerOff:innerOff], scratch[blockSize:], headerTypeUint64Flag, false)
	if err != nil {
		return nil, err
	}
	innerLen := len(inner)
	totalLen := innerLen + for64BaseSize

	// Extend dst to cover the full output area.
	dst = dst[:off+totalLen]

	// Determine header metadata size (4 bytes, or 6 with exceptions).
	header := bo.Uint32(inner)
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	insertPos := headerBytes
	if excCount > 0 {
		insertPos += svbLenBytes
	}

	// Copy header metadata (4 or 6 bytes) to the front—no overlap since
	// for64BaseSize (8) > insertPos (4 or 6).
	copy(dst[off:], inner[:insertPos])

	// Write the 8-byte FOR64 base right after the metadata.
	bo.PutUint64(dst[off+insertPos:], min64)

	// Set forWidth=3 in header.
	header |= uint32(forWidthU32) << forWidthShift
	bo.PutUint32(dst[off:], header)

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
// Scratch layout (3·blockSize = 384 uint32 slots):
//
//	scratch[0 : blockSize]            – Block 1 output (lower halves)
//	scratch[blockSize : 2·blockSize]  – Block 2 output (upper halves)
//	scratch[2·blockSize : 3·blockSize] – exception workspace (shared by both blocks)
//
// Lower and upper halves occupy non-overlapping regions, enabling a
// single fused combine loop instead of two separate passes.
func unpackUint64Scalar(dst []uint64, scratch []uint32, buf []byte) ([]uint64, int, error) {
	if len(buf) < headerBytes {
		return nil, 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	_, _, intType, _, forWidth, _, _, _, _, hasCombine := decodeHeader(header)
	if err := validateIntType64(intType); err != nil {
		return nil, 0, err
	}

	if !hasCombine {
		// Read FOR64 base if present (before inner unpack which skips these bytes).
		// for64Base remains 0 for non-FOR64 blocks, making forAdd64 equivalent
		// to a plain uint32→uint64 widening (branchless on the hot loop).
		var for64Base uint64
		if isFor64SingleBlock(intType, forWidth, hasCombine) {
			baseOff := headerBytes
			if excCount := int((header >> headerExcCountShift) & headerExcCountMask); excCount > 0 {
				baseOff += svbLenBytes
			}
			if len(buf) < baseOff+for64BaseSize {
				return nil, 0, ErrInvalidBuffer
			}
			for64Base = bo.Uint64(buf[baseOff:])
		}

		vals, consumed, err := unpackUint32Scalar(scratch[:blockSize], scratch[2*blockSize:], buf, true)
		if err != nil {
			return nil, 0, err
		}
		count := len(vals)
		if cap(dst) < count {
			dst = make([]uint64, count)
		}
		dst = dst[:count]

		// Unified path: forAdd64 with base=0 is equivalent to widening.
		forAdd64(dst, vals[:count], for64Base, count)
		return dst, consumed, nil
	}

	// Two-block path: decode both blocks into separate scratch regions,
	// then combine in a single fused loop.

	// Block 1 (lower halves) → scratch[0:blockSize], exceptions via scratch[2·blockSize:].
	lowerVals, block1Consumed, err := unpackUint32Scalar(scratch[:blockSize], scratch[2*blockSize:], buf, true)
	if err != nil {
		return nil, 0, err
	}
	count := len(lowerVals)

	// Read block2Len from Block 1 metadata to locate Block 2.
	_, _, _, _, fw, hasExceptions, _, _, _, _ := decodeHeader(header)
	block2Len := int(readBlock2Len(buf, forBaseBytes(fw), hasExceptions))
	block2Start := block1Consumed
	if block2Start+block2Len > len(buf) {
		return nil, 0, ErrInvalidBuffer
	}

	// Block 2 (upper halves) → scratch[blockSize:2·blockSize], exceptions via scratch[2·blockSize:].
	upperVals, _, err := unpackUint32Scalar(scratch[blockSize:2*blockSize], scratch[2*blockSize:], buf[block2Start:block2Start+block2Len], true)
	if err != nil {
		return nil, 0, err
	}
	if len(upperVals) != count {
		return nil, 0, ErrInvalidBuffer
	}

	if cap(dst) < count {
		dst = make([]uint64, count)
	}
	dst = dst[:count]

	// Fused single-pass combine: both halves are in non-overlapping scratch
	// regions, so we write each dst element exactly once.
	for i := range count {
		dst[i] = uint64(upperVals[i])<<32 | uint64(lowerVals[i])
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
		lower, err := getUint32Scalar(pos, buf, scratch, true)
		if err != nil {
			return 0, err
		}

		// for64Base is 0 for non-FOR64 blocks, making the add a no-op.
		var for64Base uint64
		if isFor64SingleBlock(intType, forWidth, hasCombine) {
			baseOff := headerBytes
			if excCount > 0 {
				baseOff += svbLenBytes
			}
			for64Base = bo.Uint64(buf[baseOff:])
		}
		return uint64(lower) + for64Base, nil
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
		excIdxSize := min(excCount, excBitmapThreshold)
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
