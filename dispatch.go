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

// detectSIMDLevel returns scalar until it's wired up with archsimd probes.
func detectSIMDLevel() int {
	return simdLevelScalar
}
