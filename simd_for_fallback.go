//go:build !amd64 || !goexperiment.simd

package utlpfor

// forSubtractSIMD delegates to scalar on non-SIMD builds.
func forSubtractSIMD(dst, src []uint32, baseValue uint32) {
	forSubtractScalar(dst, src, baseValue)
}

// forAddSIMD delegates to scalar on non-SIMD builds.
func forAddSIMD(output []uint32, count int, baseValue uint32) {
	forAddScalar(output, count, baseValue)
}

// findMinMaxSIMD delegates to scalar on non-SIMD builds.
func findMinMaxSIMD(values []uint32) (uint32, uint32) {
	return findMinMaxScalar(values)
}
