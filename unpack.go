package utlpfor

// UnpackUint32 decodes a packed block into dst, using scratch as workspace.
// Returns the populated dst slice, the number of bytes consumed, and any error.
func UnpackUint32(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		return unpackUint32AVX512(dst, scratch, buf)
	case simdLevelAVX2:
		return unpackUint32AVX2(dst, scratch, buf)
	case simdLevelSSE2:
		return unpackUint32SSE2(dst, scratch, buf)
	default:
		return unpackUint32Scalar(dst, scratch, buf)
	}
}
