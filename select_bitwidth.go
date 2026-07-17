package utlpfor

import "math/bits"

// stepWidth converts a step index (0-8) to the corresponding bitwidth.
// Will be inlined by the compiler.
func stepWidth(si int) int { return si << 2 }

// roundUpToStep rounds a raw bitwidth up to the nearest step bitwidth.
func roundUpToStep(bw int) int {
	return ((bw + 3) / 4) * 4
}

// stepIndex converts a step bitwidth (0,4,8,...,32) to the step index (0-8).
func stepIndex(bw int) int { return bw >> 2 }

// selectBitWidthNoPatch computes the minimum step bitwidth that fits all
// values with zero exceptions. Uses a single OR-reduction pass over the
// data followed by bits.Len32 and step rounding.
func selectBitWidthNoPatch(values []uint32) int {
	var orAll uint32
	for _, v := range values {
		orAll |= v
	}
	return roundUpToStep(bits.Len32(orAll))
}

// excCostBytesPerValue maps step-delta index (0-8) to SVB bytes-per-value.
// Index is (maxStepIdx - candidateStepIdx), i.e. the number of 4-bit steps
// between the candidate and the maximum. Pre-computed from:
//
//	min(max(((delta*4)+7)>>3, 1), 4)
//
// where delta is the step index difference.
var excCostBytesPerValue = [9]int{1, 1, 1, 2, 2, 3, 3, 4, 4}

// selectBitWidth finds the optimal step bitwidth that minimizes total block size.
// Uses a single pass over values to build a 9-bin step histogram and track the
// OR of all values (for branchless maxStepIdx computation). Exception counts at
// each candidate are computed via running suffix sums over the 9 bins.
func selectBitWidth(values []uint32) (width int, excCount int) {
	var hist [9]int
	var orAll uint32
	for _, v := range values {
		orAll |= v
		hist[(bits.Len32(v)+3)>>2]++
	}
	maxStepIdx := int((bits.Len32(orAll) + 3) >> 2)
	width, excCount, _ = chooseBestFromHist(hist, maxStepIdx)
	return
}

// chooseBestFromHist evaluates step candidates using a 9-bin histogram
// and returns the optimal bitwidth, exception count, and estimated block cost.
// maxStepIdx is the highest histogram bin with nonzero count.
func chooseBestFromHist(hist [9]int, maxStepIdx int) (bestWidth, bestExcCount, bestCost int) {
	maxStep := stepWidth(maxStepIdx)
	bestWidth = maxStep
	bestCost = headerBytes + (maxStepIdx << 6)

	excAbove := 0
	for si := maxStepIdx - 1; si >= 0; si-- {
		excAbove += hist[si+1]
		cost := headerBytes + (si << 6) + estimateExceptionCost(excAbove, maxStepIdx-si)
		if cost < bestCost {
			bestCost = cost
			bestWidth = stepWidth(si)
			bestExcCount = excAbove
		}
	}
	return
}

// chooseBestFromExcCounts evaluates step candidates using cumulative exception
// counts (from SIMD threshold comparisons). excCounts[si] = number of values
// exceeding the threshold for step width stepBitWidths[si]. This is equivalent
// to chooseBestFromHist but bypasses the histogram entirely.
func chooseBestFromExcCounts(excCounts [9]int) (bestWidth, bestExcCount, bestCost int) {
	// excCounts is monotonic, so the first zero threshold determines the
	// highest step worth considering. Build a mask of zero-count positions and
	// use TrailingZeros16 instead of an early-exit branch.
	var zeroMask uint16
	for si, c := range excCounts {
		zeroMask |= uint16(1-((bits.Len64(uint64(c))+63)>>6)) << si
	}
	maxStepIdx := bits.TrailingZeros16(zeroMask | (1 << 8))
	/*
			maxStepIdx := 8
		for si := range 9 {
			if excCounts[si] == 0 {
				maxStepIdx = si
				break
			}
		}
	*/
	maxStep := stepWidth(maxStepIdx)
	bestWidth = maxStep
	bestCost = headerBytes + (maxStepIdx << 6)

	for si := maxStepIdx - 1; si >= 0; si-- {
		ec := excCounts[si]
		cost := headerBytes + (si << 6) + estimateExceptionCost(ec, maxStepIdx-si)
		if cost < bestCost {
			bestCost = cost
			bestWidth = stepWidth(si)
			bestExcCount = ec
		}
	}
	return
}

// estimateExceptionCost estimates the byte overhead of storing exceptions.
// Branchless: uses min() for the position-vs-bitmap index cost decision.
// stepDelta is the number of 4-bit steps between candidate and maximum
// (i.e., maxStepIdx - candidateStepIdx). Uses pre-computed LUT for
// bytes-per-value to avoid the min/max chain.
func estimateExceptionCost(excCount, stepDelta int) int {
	bpv := excCostBytesPerValue[stepDelta]
	controlBytes := (excCount + 3) >> 2
	return svbLenBytes + min(excCount, excBitmapThreshold) + controlBytes + excCount*bpv
}
