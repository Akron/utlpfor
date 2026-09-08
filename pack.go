package utlpfor

// ScratchLen is the minimum scratch buffer capacity (in uint32 elements)
// for zero-allocation Pack and Unpack operations.
const ScratchLen = blockSize

// ScratchLenNoInPlace is the minimum scratch buffer capacity (in uint32
// elements) for zero-allocation Pack operations when NoInPlace or Append
// is set. The first blockSize elements hold the values work buffer and the
// second blockSize elements hold exception highBits.
const ScratchLenNoInPlace = 2 * blockSize

// ensureLen returns a byte slice with at least n bytes, reusing dst if possible.
func ensureLen(dst []byte, n int) []byte {
	if cap(dst) >= n {
		return dst[:n]
	}
	return make([]byte, n)
}

// maxBlockLen32 is the precomputed worst-case byte length of any single
// packed uint32 block (MaxBlockLength32(0)). Using the flag=0 maximum
// is safe for all flag combinations since it is the global upper bound.
var maxBlockLen32 = MaxBlockLength32(0)

// ensureCapacity32 ensures dst has cap >= off+maxBlockLen32,
// preserving dst[:off]. Called once in the Append path so all inner
// pack paths can unconditionally trust the capacity.
// On the hot path (pre-allocated dst), this is a single comparison that
// the branch predictor always predicts correctly; the grow path is kept
// out of line so the inlined fast path stays minimal.
// Growth is geometric (doubling, floored at maxBlockLen32) so that
// sequential Append calls into one buffer amortize to O(log n)
// reallocations and O(n) total copy work, mirroring the semantics of
// Go's builtin append. The fixed-quantum policy this replaces reallocated
// (and copied the whole prefix) on nearly every Append call, which made
// stream building O(n^2) in the accumulated payload size.
func ensureCapacity32(dst []byte, off int) []byte {
	if cap(dst)-off >= maxBlockLen32 {
		return dst
	}
	return growCapacity32(dst, off)
}

// growCapacity32 is the cold reallocation path of ensureCapacity32.
func growCapacity32(dst []byte, off int) []byte {
	newCap := max(cap(dst)*2, off+maxBlockLen32)
	grown := make([]byte, off, newCap)
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
//
// When NoInPlace or Append is set, the values slice is guaranteed unmodified
// after the call. The library uses scratch as a work buffer for FOR subtraction
// and delta encoding. For zero-allocation operation with NoInPlace/Append,
// provide scratch with capacity >= ScratchLenNoInPlace (256).
//
// Without NoInPlace or Append, the values slice may be modified in-place
// (FOR subtraction, delta encoding). Callers that need the original values
// must either set NoInPlace or copy them before calling PackUint32.
//
// To pre-allocate dst for zero-allocation packing, use MaxBlockLength32:
//
//	dst := make([]byte, 0, MaxBlockLength32(flag))
//	dst, err = PackUint32(flag|Append, values, dst, scratch)
func PackUint32(flag Flag, values []uint32, dst []byte, scratch []uint32) ([]byte, error) {
	// Append implies NoInPlace; this branchless form relies on the two bits
	// being adjacent (Append = 1<<4, NoInPlace = 1<<5).
	flag |= (flag & Append) << 1
	if flag&NoInPlace != 0 {
		if len(scratch) < ScratchLenNoInPlace {
			scratch = make([]uint32, ScratchLenNoInPlace)
		}
	} else if len(scratch) < ScratchLen {
		scratch = make([]uint32, ScratchLen)
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
