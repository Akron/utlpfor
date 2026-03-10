package utlpfor

// ensureLen returns a byte slice with at least n bytes, reusing dst if possible.
func ensureLen(dst []byte, n int) []byte {
	if cap(dst) >= n {
		return dst[:n]
	}
	return make([]byte, n)
}

// PackUint32 encodes uint32 values into a packed block.
// If dst has sufficient capacity, it is reused; otherwise a new slice is allocated.
// The flag byte controls encoding options (e.g. Delta for delta encoding).
// The values slice may be modified in-place (FOR subtraction, delta encoding).
// Callers that need the original values must copy them before calling PackUint32.
func PackUint32(flag byte, dst []byte, values []uint32) ([]byte, error) {
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		return packUint32AVX512(flag, dst, values)
	case simdLevelAVX2:
		return packUint32AVX2(flag, dst, values)
	case simdLevelSSE2:
		return packUint32SSE2(flag, dst, values)
	default:
		return packUint32Scalar(flag, dst, values)
	}
}
