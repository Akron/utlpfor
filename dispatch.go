package utlpfor

// SIMD level constants for runtime dispatch.
//
// simdLevelSSE2 is the 128-bit level: archsimd Uint32x4/Uint64x2 kernels.
//
// NOTE: the label "SSE2" is not perfect (kept to keep the code change small).
// simd/archsimd emits only VEX-encoded instructions, so these kernels need
// AVX (not AVX2, not SSE2 alone). A CPU with SSE only (no AVX) cannot run any
// archsimd code and gets simdLevelScalar, silently, without any error.
// The kernels also keep the historical "SSE2" in their names
// (e.g. unpackSSE2BW4).
const (
	simdLevelScalar     = 0
	simdLevelSSE2       = 1
	simdLevelAVX2       = 2
	simdLevelAVX512     = 3
	simdLevelAVX512VBMI = 4
)

// simdLevel holds the detected SIMD level, set once at init.
var simdLevel = simdLevelScalar

func init() {
	simdLevel = detectSIMDLevel()
}

// selectSimdLevel returns the SIMD level for given CPU feature flags.
// Exported for testability; the actual dispatch uses detectSIMDLevel.
// A level is only selected if its own flag is set: a CPU without AVX gets
// the scalar level, whatever else is claimed.
func selectSimdLevel(hasAVX512VBMI, hasAVX512, hasAVX2, hasAVX bool) int {
	if hasAVX512VBMI {
		return simdLevelAVX512VBMI
	}
	if hasAVX512 {
		return simdLevelAVX512
	}
	if hasAVX2 {
		return simdLevelAVX2
	}
	if hasAVX {
		return simdLevelSSE2 // 128-bit kernels, needs AVX (see above)
	}
	return simdLevelScalar
}
