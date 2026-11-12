package utlpfor

// UnpackUint32 decodes a packed block into values, using scratch as workspace.
// Returns the populated values slice, the number of bytes consumed, and any error.
func UnpackUint32(src []byte, values []uint32, scratch []uint32) ([]uint32, int, error) {
	if len(scratch) < blockSize {
		scratch = make([]uint32, blockSize)
	}
	switch {
	case simdLevel.has(simdAVX512):
		return unpackUint32AVX512(values, scratch, src)
	case simdLevel.has(simdAVX2):
		return unpackUint32AVX2(values, scratch, src)
	case simdLevel.has(simdAVX):
		return unpackUint32SSE2(values, scratch, src)
	default:
		return unpackUint32Scalar(values, scratch, src, false)
	}
}
