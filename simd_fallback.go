//go:build !amd64 || !goexperiment.simd

package utlpfor

// Fallback stubs for non-SIMD builds. These delegate to scalar and are
// never called at runtime (simdLevel is always simdLevelScalar), but
// must exist so that dispatch.go compiles on all platforms.

func packUint32AVX512(flag byte, dst []byte, values []uint32) ([]byte, error) {
	return packUint32Scalar(flag, dst, values)
}

func unpackUint32AVX512(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackUint32Scalar(dst, scratch, buf)
}

func getUint32AVX512(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}

func packUint32AVX2(flag byte, dst []byte, values []uint32) ([]byte, error) {
	return packUint32Scalar(flag, dst, values)
}

func unpackUint32AVX2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackUint32Scalar(dst, scratch, buf)
}

func getUint32AVX2(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}

func packUint32SSE2(flag byte, dst []byte, values []uint32) ([]byte, error) {
	return packUint32Scalar(flag, dst, values)
}

func unpackUint32SSE2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackUint32Scalar(dst, scratch, buf)
}

func getUint32SSE2(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
