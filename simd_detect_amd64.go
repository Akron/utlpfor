//go:build goexperiment.simd && amd64

package utlpfor

import "simd/archsimd"

// detectSIMDLevel probes CPU features and returns the best SIMD level.
func detectSIMDLevel() int {
	if archsimd.X86.AVX512VBMI() {
		return simdLevelAVX512VBMI
	}
	if archsimd.X86.AVX512() {
		return simdLevelAVX512
	}
	if archsimd.X86.AVX2() {
		return simdLevelAVX2
	}
	return simdLevelSSE2
}
