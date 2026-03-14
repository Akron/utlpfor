//go:build goexperiment.simd && amd64

package utlpfor

import "simd/archsimd"

// detectSIMDLevel probes CPU features and returns the best SIMD level.
func detectSIMDLevel() int {
	return selectSimdLevel(
		archsimd.X86.AVX512VBMI(),
		archsimd.X86.AVX512(),
		archsimd.X86.AVX2(),
		true, // amd64 baseline includes SSE2
	)
}
