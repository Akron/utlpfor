//go:build !amd64 || !goexperiment.simd

package utlpfor

// detectSIMDLevel returns the empty feature set (scalar level) when SIMD is
// not available.
func detectSIMDLevel() simdFeatures {
	return simdLevelScalar
}
