//go:build !amd64 || !goexperiment.simd

package utlpfor

// selectBitWidthSIMDtest delegates to scalar on non-SIMD builds.
func selectBitWidthSIMDtest(values []uint32) (int, int) {
	return selectBitWidth(values)
}

// selectBitWidthWithFORSIMDtest delegates to scalar on non-SIMD builds.
func selectBitWidthWithFORSIMDtest(values []uint32) (bool, uint32, int) {
	return selectBitWidthWithFOR(values)
}
