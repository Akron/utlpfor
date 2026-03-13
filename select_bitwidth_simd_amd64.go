//go:build goexperiment.simd && amd64

package utlpfor

import "math/bits"

// gtCountU32 returns 1 if v > threshold, else 0, without a branch.
// We subtract (threshold+1) in uint64; unsigned underflow sets the top bit for
// v <= threshold, then we invert that bit to get a 0/1 increment.
func gtCountU32(v, threshold uint32) int {
	return int(((uint64(v) - (uint64(threshold) + 1)) >> 63) ^ 1)
}

// selectBitWidthSIMDtest computes the optimal step bitwidth using SIMD threshold
// comparisons. Instead of computing bits.Len32 per value and incrementing a
// histogram bin (scatter-add), this approach compares all values against each
// step threshold simultaneously. The 8 thresholds are split into two groups
// of 4 to avoid register pressure (5 SIMD registers per pass: 4 thresholds
// + 1 value vector). AVX-512 uses a single pass (9 of 32 ZMM registers).
// Only for testing purposes.
func selectBitWidthSIMDtest(values []uint32) (width int, excCount int) {
	var excCounts [9]int
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		excCounts = buildExcCountsAVX512(values)
	case simdLevelAVX2:
		excCounts = buildExcCountsAVX2(values)
	case simdLevelSSE2:
		excCounts = buildExcCountsSSE2(values)
	default:
		return selectBitWidth(values)
	}
	width, excCount, _ = chooseBestFromExcCounts(excCounts)
	return
}

// selectBitWidthWithFORSIMDtest uses SIMD-accelerated findMinMax and SIMD
// threshold comparisons for the standard histogram. The two-level rejection
// strategy (min==0, forMaxStep >= rawMaxStep/stdWidth) is preserved.
// Only for testing purposes.
func selectBitWidthWithFORSIMDtest(values []uint32) (useFOR bool, baseValue uint32, forWidth int) {
	minVal, maxVal := findMinMaxSIMDtest(values)
	if minVal == 0 {
		return false, 0, forWidthNone
	}
	forMaxStep := roundUpToStep(bits.Len32(maxVal - minVal))
	rawMaxStep := roundUpToStep(bits.Len32(maxVal))
	if forMaxStep >= rawMaxStep {
		return false, 0, forWidthNone
	}

	var excCounts [9]int
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		excCounts = buildExcCountsAVX512(values)
	case simdLevelAVX2:
		excCounts = buildExcCountsAVX2(values)
	case simdLevelSSE2:
		excCounts = buildExcCountsSSE2(values)
	default:
		return decideFORFull(values, minVal, maxVal)
	}
	stdWidth, _, _ := chooseBestFromExcCounts(excCounts)
	if forMaxStep >= stdWidth {
		return false, 0, forWidthNone
	}
	return true, minVal, selectFORWidth(minVal)
}
