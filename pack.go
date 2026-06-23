package utlpfor

// ScratchLen is the minimum scratch buffer capacity (in uint32 elements)
// for zero-allocation Pack and Unpack operations.
const ScratchLen = blockSize

// ensureLen returns a byte slice with at least n bytes, reusing dst if possible.
func ensureLen(dst []byte, n int) []byte {
	if cap(dst) >= n {
		return dst[:n]
	}
	return make([]byte, n)
}

// ensureAppend returns dst with len set to off+n, preserving dst[:off].
// If cap(dst) is sufficient, the slice is resliced; otherwise a new slice
// is allocated and the prefix is copied.
func ensureAppend(dst []byte, off, n int) []byte {
	needed := off + n
	if cap(dst) >= needed {
		return dst[:needed]
	}
	grown := make([]byte, needed)
	copy(grown, dst[:off])
	return grown
}

// PackUint32 encodes uint32 values into a packed block.
// If dst has sufficient capacity, it is reused; otherwise a new slice is allocated.
// If scratch has capacity >= ScratchLen (128), no heap allocations occur
// (assuming dst also has sufficient capacity). Pass nil for scratch to use
// internal allocations.
// The flag parameter controls encoding options (e.g. Delta for delta encoding).
// When Append is set, the packed block is written after the existing content
// of dst (starting at len(dst)) instead of overwriting from index 0.
// The returned slice includes the preserved prefix followed by the new block.
// The values slice may be modified in-place (FOR subtraction, delta encoding).
// Callers that need the original values must copy them before calling PackUint32.
func PackUint32(flag Flag, values []uint32, dst []byte, scratch []uint32) ([]byte, error) {
	if len(scratch) < blockSize {
		scratch = make([]uint32, blockSize)
	}
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		return packUint32AVX512(flag, dst, scratch, values)
	case simdLevelAVX2:
		return packUint32AVX2(flag, dst, scratch, values)
	case simdLevelSSE2:
		return packUint32SSE2(flag, dst, scratch, values)
	default:
		return packUint32Scalar(flag, dst, scratch, values)
	}
}
