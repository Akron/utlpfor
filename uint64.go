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
// preserving dst[:off]. Called once at the packUint64 entry points
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

// combineUint64Scalar merges lower and upper uint32 halves into uint64 values.
func combineUint64Scalar(dst []uint64, lower, upper []uint32, count int) {
	for i := range count {
		dst[i] = uint64(upper[i])<<32 | uint64(lower[i])
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

// splitUint64Halves extracts lower and upper uint32 halves from uint64
// values, tracks min/max, and OR-accumulates all values in a single pass.
// Used by scalar, SSE2, and AVX2 pack paths. AVX512 uses dedicated SIMD.
func splitUint64Halves(lower, upper []uint32, values []uint64) (min64, max64, acc uint64) {
	min64, max64 = values[0], values[0]
	for i, v := range values {
		lower[i] = uint32(v)
		upper[i] = uint32(v >> 32)
		acc |= v
		min64 = min(min64, v)
		max64 = max(max64, v)
	}
	return
}

// wrapFor64SingleBlock wraps an already-packed inner block with FOR64 metadata.
// The inner block must reside at dst[off+for64BaseSize:]. This function
// relocates the header metadata to dst[off:] and inserts the 8-byte
// FOR64 base value between header metadata and payload.
func wrapFor64SingleBlock(dst []byte, off int, min64 uint64, inner []byte) []byte {
	innerLen := len(inner)
	totalLen := innerLen + for64BaseSize
	dst = dst[:off+totalLen]

	header := bo.Uint32(inner)
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	insertPos := headerBytes
	if excCount > 0 {
		insertPos += svbLenBytes
	}

	copy(dst[off:], inner[:insertPos])
	bo.PutUint64(dst[off+insertPos:], min64)
	header |= uint32(forWidthU32) << forWidthShift
	bo.PutUint32(dst[off:], header)

	return dst[:off+totalLen]
}

// packFor64 packs a FOR64 single-block from pre-subtracted uint32 work
// values in scratch[:count]. Callers must subtract min64 before calling.
// Packs via the provided packBlock function, then wraps with FOR64 metadata.
func packFor64(flag Flag, dst []byte, scratch []uint32, off, count int,
	min64 uint64, packBlock func(Flag, []uint32, []byte, []uint32, uint32, bool) ([]byte, error),
) ([]byte, error) {
	innerOff := off + for64BaseSize
	dst = dst[:innerOff]
	inner, err := packBlock(flag|NoFOR, scratch[:count], dst[innerOff:innerOff], scratch[blockSize:], headerTypeUint64Flag, false)
	if err != nil {
		return nil, err
	}
	return wrapFor64SingleBlock(dst, off, min64, inner), nil
}

// packUint64TwoBlock encodes a uint64 block as two sub-blocks (lower + upper
// halves) using the provided packBlock function.
func packUint64TwoBlock(flag Flag, dst []byte, scratch []uint32, off, count int,
	packBlock func(Flag, []uint32, []byte, []uint32, uint32, bool) ([]byte, error),
) ([]byte, error) {
	upper := scratch[2*blockSize:]

	block1, err := packBlock(flag, scratch[:count], dst[off:off], scratch[blockSize:2*blockSize], headerTypeUint64Flag|headerCombineFlag, true)
	if err != nil {
		return nil, err
	}
	block1Len := len(block1)

	b2Off := off + block1Len
	dst = dst[:b2Off]
	block2, err := packBlock(flag, upper[:count], dst[b2Off:b2Off], scratch[blockSize:2*blockSize], headerTypeUint64Flag, false)
	if err != nil {
		return nil, err
	}
	block2Len := len(block2)

	block1Header := bo.Uint32(block1)
	_, _, _, _, forWidth, hasExceptions, _, _, _, _ := decodeHeader(block1Header)
	writeBlock2Len(block1, forBaseBytes(forWidth), hasExceptions, uint16(block2Len))

	return dst[:off+block1Len+block2Len], nil
}

// unpackUint64Block handles the shared uint64 unpack orchestration:
// header decoding, single-block vs FOR64 vs two-block dispatch.
// The caller provides SIMD-level-specific functions for block unpacking,
// forAdd64, and combine.
func unpackUint64Block(dst []uint64, scratch []uint32, buf []byte,
	unpackBlock func([]uint32, []uint32, []byte, bool) ([]uint32, int, error),
	addBase func([]uint64, []uint32, uint64, int),
	combine func([]uint64, []uint32, []uint32, int),
) ([]uint64, int, error) {
	if len(buf) < headerBytes {
		return nil, 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	_, _, intType, _, forWidth, _, _, _, _, hasCombine := decodeHeader(header)
	if err := validateIntType64(intType); err != nil {
		return nil, 0, err
	}

	if !hasCombine {
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

		vals, consumed, err := unpackBlock(scratch[:blockSize], scratch[2*blockSize:], buf, true)
		if err != nil {
			return nil, 0, err
		}
		count := len(vals)
		if cap(dst) < count {
			dst = make([]uint64, count)
		}
		dst = dst[:count]

		addBase(dst, vals[:count], for64Base, count)
		return dst, consumed, nil
	}

	lowerVals, block1Consumed, err := unpackBlock(scratch[:blockSize], scratch[2*blockSize:], buf, true)
	if err != nil {
		return nil, 0, err
	}
	count := len(lowerVals)

	_, _, _, _, fw, hasExceptions, _, _, _, _ := decodeHeader(header)
	block2Len := int(readBlock2Len(buf, forBaseBytes(fw), hasExceptions))
	block2Start := block1Consumed
	if block2Start+block2Len > len(buf) {
		return nil, 0, ErrInvalidBuffer
	}

	upperVals, _, err := unpackBlock(scratch[blockSize:2*blockSize], scratch[2*blockSize:], buf[block2Start:block2Start+block2Len], true)
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

	combine(dst, lowerVals, upperVals, count)
	return dst, block1Consumed + block2Len, nil
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
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		return packUint64AVX512(flag, values, dst, scratch)
	case simdLevelAVX2:
		return packUint64AVX2(flag, values, dst, scratch)
	case simdLevelSSE2:
		return packUint64SSE2(flag, values, dst, scratch)
	default:
		return packUint64Scalar(flag, values, dst, scratch)
	}
}

// packUint64Scalar is the scalar implementation of PackUint64.
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

	dst = ensureCapacity64(dst, off, innerFlag)

	upper := scratch[2*blockSize:]
	min64, max64, acc := splitUint64Halves(scratch, upper, values)

	if innerFlag&NoFOR == 0 && min64 > 0 && (max64-min64) < (1<<32) {
		forSubtract64(scratch[:count], values, min64)
		return packFor64(innerFlag, dst, scratch, off, count, min64, packBlockScalar)
	}

	if acc>>32 == 0 {
		block, err := packBlockScalar(innerFlag, scratch[:count], dst[off:off], scratch[blockSize:2*blockSize], headerTypeUint64Flag, false)
		if err != nil {
			return nil, err
		}
		return dst[:off+len(block)], nil
	}

	return packUint64TwoBlock(innerFlag, dst, scratch, off, count, packBlockScalar)
}

// UnpackUint64 decodes a packed uint64 block into values, using scratch
// as workspace. Returns the populated values slice, the number of bytes
// consumed, and any error.
func UnpackUint64(dst []uint64, scratch []uint32, buf []byte) ([]uint64, int, error) {
	if len(scratch) < ScratchLen64 {
		scratch = make([]uint32, ScratchLen64)
	}
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		return unpackUint64AVX512(dst, scratch, buf)
	case simdLevelAVX2:
		return unpackUint64AVX2(dst, scratch, buf)
	case simdLevelSSE2:
		return unpackUint64SSE2(dst, scratch, buf)
	default:
		return unpackUint64Scalar(dst, scratch, buf)
	}
}

// unpackUint64Scalar is the scalar implementation of UnpackUint64.
func unpackUint64Scalar(dst []uint64, scratch []uint32, buf []byte) ([]uint64, int, error) {
	return unpackUint64Block(dst, scratch, buf, unpackUint32Scalar, forAdd64, combineUint64Scalar)
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

	lower, err := getUint32Scalar(pos, buf, scratch, true)
	if err != nil {
		return 0, err
	}

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
