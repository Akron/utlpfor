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
// When Delta is set, values are delta-encoded in place.
func PackUint32(flag byte, dst []byte, values []uint32) ([]byte, error) {
	switch simdLevel {
	default:
		return packUint32Scalar(flag, dst, values)
	}
}
