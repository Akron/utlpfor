//go:build !amd64 || !goexperiment.simd

package utlpfor

// Fallback stubs for non-SIMD builds. These delegate to scalar and are
// never called at runtime (simdLevel is always simdLevelScalar), but
// must exist so that dispatch.go and uint64.go compile on all platforms.

func packUint32AVX512(flag Flag, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	return packUint32Scalar(flag, dst, scratch, values)
}

func unpackUint32AVX512(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackUint32Scalar(dst, scratch, buf, false)
}

func packUint32AVX2(flag Flag, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	return packUint32Scalar(flag, dst, scratch, values)
}

func unpackUint32AVX2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackUint32Scalar(dst, scratch, buf, false)
}

func packUint32SSE2(flag Flag, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	return packUint32Scalar(flag, dst, scratch, values)
}

func unpackUint32SSE2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackUint32Scalar(dst, scratch, buf, false)
}

// forSubtractSIMDtest delegates to scalar on non-SIMD builds.
func forSubtractSIMDtest(dst, src []uint32, baseValue uint32) {
	forSubtractScalar(dst, src, baseValue)
}

// forAddSIMDtest delegates to scalar on non-SIMD builds.
func forAddSIMDtest(output []uint32, count int, baseValue uint32) {
	forAddScalar(output, count, baseValue)
}

// findMinMaxSIMDtest delegates to scalar on non-SIMD builds.
func findMinMaxSIMDtest(values []uint32) (uint32, uint32) {
	return findMinMaxScalar(values)
}

// selectBitWidthSIMDtest delegates to scalar on non-SIMD builds.
func selectBitWidthSIMDtest(values []uint32) (int, int) {
	return selectBitWidth(values)
}

// selectBitWidthWithFORSIMDtest delegates to scalar on non-SIMD builds.
func selectBitWidthWithFORSIMDtest(values []uint32) (bool, uint32, int) {
	return selectBitWidthWithFOR(values)
}

func packUint64AVX512(flag Flag, values []uint64, dst []byte, scratch []uint32) ([]byte, error) {
	return packUint64Scalar(flag, values, dst, scratch)
}

func unpackUint64AVX512(dst []uint64, scratch []uint32, buf []byte) ([]uint64, int, error) {
	return unpackUint64Scalar(dst, scratch, buf)
}

func packUint64AVX2(flag Flag, values []uint64, dst []byte, scratch []uint32) ([]byte, error) {
	return packUint64Scalar(flag, values, dst, scratch)
}

func unpackUint64AVX2(dst []uint64, scratch []uint32, buf []byte) ([]uint64, int, error) {
	return unpackUint64Scalar(dst, scratch, buf)
}

func packUint64SSE2(flag Flag, values []uint64, dst []byte, scratch []uint32) ([]byte, error) {
	return packUint64Scalar(flag, values, dst, scratch)
}

func unpackUint64SSE2(dst []uint64, scratch []uint32, buf []byte) ([]uint64, int, error) {
	return unpackUint64Scalar(dst, scratch, buf)
}
