//go:build !amd64 || !goexperiment.simd

package utlpfor

// detectSIMDLevel returns scalar level when SIMD is not available.
func detectSIMDLevel() int {
	return simdLevelScalar
}
