//go:build !amd64 || !goexperiment.simd

package utlpfor

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
