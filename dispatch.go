package utlpfor

// simdFeatures is a bit set of the SIMD features that the runtime dispatch
// may use. One bit per supported feature, so a check is a single mask test:
//
//	if simdLevel.has(simdAVX2) { ... }
//
// The set is normalized by selectSimdLevel: a bit is only set if the bits of
// all lower features are set as well (VBMI needs AVX-512, AVX-512 needs AVX2,
// AVX2 needs AVX). The zero value is the scalar level (no feature).
type simdFeatures uint8

const (
	// simdAVX is the 128-bit level: archsimd Uint32x4/Uint64x2 kernels.
	//
	// NOTE: the kernels carry the label "SSE2" in their names (e.g.
	// unpackSSE2BW4, simd_sse2_amd64.go, simdLevelSSE2), which is not perfect
	// (kept to keep the code change small). simd/archsimd emits only
	// VEX-encoded instructions, so these kernels need AVX (not AVX2, not SSE2
	// alone). A CPU with SSE only (no AVX) cannot run any archsimd code and
	// gets the scalar level (no bit set), silently, without any error.
	simdAVX simdFeatures = 1 << iota
	simdAVX2
	simdAVX512
	simdAVX512VBMI
)

// The detected levels as (normalized) feature sets. The "SSE2" name of the
// 128-bit level is historical, see simdAVX.
const (
	simdLevelScalar     simdFeatures = 0
	simdLevelSSE2                    = simdAVX
	simdLevelAVX2                    = simdLevelSSE2 | simdAVX2
	simdLevelAVX512                  = simdLevelAVX2 | simdAVX512
	simdLevelAVX512VBMI              = simdLevelAVX512 | simdAVX512VBMI
)

// has reports whether the feature bit f is set. Because the set is
// normalized, has(simdAVX2) also means that AVX is available, and so on.
func (s simdFeatures) has(f simdFeatures) bool {
	return s&f != 0
}

// simdLevel holds the detected SIMD features, set once at init.
var simdLevel = simdLevelScalar

func init() {
	simdLevel = detectSIMDLevel()
}

// selectSimdLevel returns the set of usable SIMD features for given CPU
// feature flags. Exported for testability; the actual dispatch uses
// detectSIMDLevel.
// A feature bit is only set if its own flag AND the flags of all lower
// features are set (VBMI needs AVX-512, AVX-512 needs AVX2, AVX2 needs AVX): a
// CPU without AVX gets the empty set (scalar), whatever else is claimed. Real
// CPUs never report an inconsistent set, but a simulated one can:
// GODEBUG=cpu.avx512f=off leaves the VBMI flag set (internal/cpu has no option
// for it), and GODEBUG=cpu.avx=off leaves AVX2 and AVX-512 set.
func selectSimdLevel(hasAVX512VBMI, hasAVX512, hasAVX2, hasAVX bool) simdFeatures {
	var s simdFeatures
	if !hasAVX {
		return s // scalar (see simdAVX)
	}
	s |= simdAVX
	if !hasAVX2 {
		return s
	}
	s |= simdAVX2
	if !hasAVX512 {
		return s
	}
	s |= simdAVX512
	if !hasAVX512VBMI {
		return s
	}
	return s | simdAVX512VBMI
}
