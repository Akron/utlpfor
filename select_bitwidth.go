package utlpfor

import "math/bits"

// stepBitWidths lists the 9 valid step bitwidths for 128-value blocks.
// UTL payload sizes quantize in groups of 4 (ceil(bw/4) * 64), so
// non-step bitwidths produce the same payload as the next step up
// while representing fewer values (more exceptions, no savings).
var stepBitWidths = [9]int{0, 4, 8, 12, 16, 20, 24, 28, 32}

// roundUpToStep rounds a raw bitwidth up to the nearest step bitwidth.
func roundUpToStep(bw int) int {
	if bw <= 0 {
		return 0
	}
	return ((bw + 3) / 4) * 4
}

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
	maxStep := stepBitWidths[maxStepIdx]
	bestWidth = maxStep
	bestCost = headerBytes + utlPayloadBytesLUT[maxStep]

	excAbove := 0
	for si := maxStepIdx - 1; si >= 0; si-- {
		excAbove += hist[si+1]
		w := stepBitWidths[si]
		cost := headerBytes + utlPayloadBytesLUT[w] + estimateExceptionCost(excAbove, maxStep-w)
		if cost < bestCost {
			bestCost = cost
			bestWidth = w
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
	maxStepIdx := 8
	for si := range 9 {
		if excCounts[si] == 0 {
			maxStepIdx = si
			break
		}
	}
	maxStep := stepBitWidths[maxStepIdx]
	bestWidth = maxStep
	bestCost = headerBytes + utlPayloadBytesLUT[maxStep]

	for si := maxStepIdx - 1; si >= 0; si-- {
		ec := excCounts[si]
		w := stepBitWidths[si]
		cost := headerBytes + utlPayloadBytesLUT[w] + estimateExceptionCost(ec, maxStep-w)
		if cost < bestCost {
			bestCost = cost
			bestWidth = w
			bestExcCount = ec
		}
	}
	return
}

// estimateExceptionCost estimates the byte overhead of storing exceptions.
// Branchless: uses min() for the position-vs-bitmap index cost decision.
// Precondition: excCount > 0 (guaranteed by chooseBestFromHist caller).
func estimateExceptionCost(excCount, maxHighBitWidth int) int {
	bytesPerValue := min(max((maxHighBitWidth+7)>>3, 1), 4)
	controlBytes := (excCount + 3) >> 2
	return svbLenBytes + min(excCount, excBitmapThreshold) + controlBytes + excCount*bytesPerValue
}
