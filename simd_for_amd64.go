//go:build goexperiment.simd && amd64

package utlpfor

// findMinMaxSIMDtest computes the minimum and maximum of a uint32 slice using
// the best available SIMD level.
func findMinMaxSIMDtest(values []uint32) (uint32, uint32) {
	switch {
	case simdLevel.has(simdAVX512):
		return findMinMaxAVX512(values)
	case simdLevel.has(simdAVX2):
		return findMinMaxAVX2(values)
	case simdLevel.has(simdAVX):
		return findMinMaxSSE2(values)
	default:
		return findMinMaxScalar(values)
	}
}

// forSubtractSIMDtest subtracts baseValue from each element using SIMD:
// dst[i] = src[i] - baseValue.
func forSubtractSIMDtest(dst, src []uint32, baseValue uint32) {
	switch {
	case simdLevel.has(simdAVX512):
		forSubtractAVX512(dst, src, baseValue)
	case simdLevel.has(simdAVX2):
		forSubtractAVX2(dst, src, baseValue)
	case simdLevel.has(simdAVX):
		forSubtractSSE2(dst, src, baseValue)
	default:
		forSubtractScalar(dst, src, baseValue)
	}
}

// forAddSIMDtest adds baseValue to each of the first count elements using SIMD:
// output[i] += baseValue.
func forAddSIMDtest(output []uint32, count int, baseValue uint32) {
	switch {
	case simdLevel.has(simdAVX512):
		forAddAVX512(output, count, baseValue)
	case simdLevel.has(simdAVX2):
		forAddAVX2(output, count, baseValue)
	case simdLevel.has(simdAVX):
		forAddSSE2(output, count, baseValue)
	default:
		forAddScalar(output, count, baseValue)
	}
}
