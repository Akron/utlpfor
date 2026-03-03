package utlpfor

// SIMD level constants for runtime dispatch.
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
func selectSimdLevel(hasAVX512VBMI, hasAVX512, hasAVX2, hasSSE2 bool) int {
	if hasAVX512VBMI {
		return simdLevelAVX512VBMI
	}
	if hasAVX512 {
		return simdLevelAVX512
	}
	if hasAVX2 {
		return simdLevelAVX2
	}
	if hasSSE2 {
		return simdLevelSSE2
	}
	return simdLevelScalar
}
