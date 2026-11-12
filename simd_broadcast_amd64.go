//go:build goexperiment.simd && amd64

package utlpfor

import "simd/archsimd"

// The 128-bit kernels (simdLevelSSE2, which needs AVX) must run on CPUs that have AVX but no
// AVX2 (Sandy Bridge, Ivy Bridge, Bulldozer ... Piledriver).
//
// archsimd.BroadcastUint32x4 and BroadcastUint64x2 are documented as
// "CPU Feature: AVX2" and compile to VPBROADCASTD/VPBROADCASTQ, which raises
// SIGILL on such a CPU. The helpers below build the same vector from a
// four-/two-element array with a plain (AVX) vector load instead.
// scripts/check_avx128_isa.sh verifies that no AVX2-only instruction is left.

// broadcastUint32x4 fills a 128-bit vector with x using AVX instructions only.
func broadcastUint32x4(x uint32) archsimd.Uint32x4 {
	v := [4]uint32{x, x, x, x}
	return archsimd.LoadUint32x4Array(&v)
}

// broadcastUint64x2 fills a 128-bit vector with x using AVX instructions only.
func broadcastUint64x2(x uint64) archsimd.Uint64x2 {
	v := [2]uint64{x, x}
	return archsimd.LoadUint64x2Array(&v)
}
