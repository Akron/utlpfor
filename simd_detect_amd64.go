//go:build goexperiment.simd && amd64

package utlpfor

import "simd/archsimd"

// detectSIMDLevel probes CPU features and returns the best SIMD level.
func detectSIMDLevel() int {
	return selectSimdLevel(
		archsimd.X86.AVX512VBMI(),
		archsimd.X86.AVX512(),
		archsimd.X86.AVX2(),
		// The 128-bit kernels are archsimd code, i.e. VEX encoded: they need
		// AVX. CPUs with SSE only (no AVX) run the scalar code.
		archsimd.X86.AVX(),
	)
}
