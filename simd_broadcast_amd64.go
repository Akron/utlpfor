//go:build goexperiment.simd && amd64

package utlpfor

import "simd/archsimd"

// The 128-bit kernels (simdLevelSSE2, which needs AVX) must run on CPUs that have AVX but no
// AVX2 (Sandy Bridge, Ivy Bridge, Bulldozer ... Piledriver).
//
// Before the Go 1.28 development tree, archsimd.BroadcastUint32x4 and
// BroadcastUint64x2 were documented as "CPU Feature: AVX2" and compile to
// VPBROADCASTD/VPBROADCASTQ, which raises SIGILL on such a CPU. The helpers
// below build the same vector with AVX instructions only:
//
//	uint32: VMOVD  r32, X0 ; VPSHUFD  $0x00, X0, X0
//	uint64: VMOVQ  r64, X0 ; VPUNPCKLQDQ X0, X0, X0
//
// Do NOT build the vector in a [4]uint32{x, x, x, x} array and load it with
// LoadUint32x4Array: that compiles to four 4-byte stores followed by one 16-byte
// load, which cannot be forwarded from the store buffer (a "store-forwarding
// stall", about 12-15 cycles on Sandy Bridge, paid at each call; it hurts small
// inputs most).
//
// scripts/check_avx128_isa.sh verifies that no AVX2-only instruction is left;
// TestBroadcastHelpers checks that every lane holds x.

// broadcastUint32x4 fills a 128-bit vector with x using AVX instructions only.
func broadcastUint32x4(x uint32) archsimd.Uint32x4 {
	var z archsimd.Uint32x4
	return z.SetElem(0, x).PermuteScalars(0, 0, 0, 0)
}

// broadcastUint64x2 fills a 128-bit vector with x using AVX instructions only.
func broadcastUint64x2(x uint64) archsimd.Uint64x2 {
	var z archsimd.Uint64x2
	v := z.SetElem(0, x)
	return v.InterleaveLo(v)
}
