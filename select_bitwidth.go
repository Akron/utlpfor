package utlpfor

import "math/bits"

// stepBitWidths lists the 9 valid step bitwidths for 128-value blocks.
// UTL payload sizes quantize in groups of 4 (ceil(bw/4) * 64), so
// non-step bitwidths produce the same payload as the next step up
// while representing fewer values (more exceptions, no savings).
var stepBitWidths = [9]int{0, 4, 8, 12, 16, 20, 24, 28, 32}

// maxBitWidth returns the minimum number of bits needed to represent
// the largest value in the slice.
func maxBitWidth(values []uint32) int {
	var ored uint32
	for _, v := range values {
		ored |= v
	}
	return bits.Len32(ored)
}

// roundUpToStep rounds a raw bitwidth up to the nearest step bitwidth.
func roundUpToStep(bw int) int {
	if bw <= 0 {
		return 0
	}
	return ((bw + 3) / 4) * 4
}

// selectBitWidth finds the optimal step bitwidth that minimizes total block size.
// Only the 9 step bitwidths (0, 4, 8, ..., 32) are evaluated because
// non-step bitwidths produce identical payload sizes as the next step up.
// Returns the chosen width and the number of exceptions at that width.
func selectBitWidth(values []uint32) (width int, excCount int) {
	var orAll uint32
	for _, v := range values {
		orAll |= v
	}
	maxWidth := bits.Len32(orAll)

	maxStepIdx := min((maxWidth+3)/4, 8)
	maxStep := stepBitWidths[maxStepIdx]

	bestWidth := maxStep
	bestSize := headerBytes + utlPayloadBytesLUT[maxStep]
	bestExcCount := 0

	var freqs [33]int
	for _, v := range values {
		freqs[bits.Len32(v)]++
	}

	for si := maxStepIdx - 1; si >= 0; si-- {
		w := stepBitWidths[si]
		excCountCum := 0
		for bw := w + 1; bw <= maxStep; bw++ {
			excCountCum += freqs[bw]
		}
		if excCountCum == 0 {
			continue
		}

		payloadCost := utlPayloadBytesLUT[w]
		excOverhead := estimateExceptionCost(excCountCum, maxStep-w)
		totalCost := headerBytes + payloadCost + excOverhead
		if totalCost < bestSize {
			bestSize = totalCost
			bestWidth = w
			bestExcCount = excCountCum
		}
	}
	return bestWidth, bestExcCount
}

// estimateExceptionCost estimates the byte overhead of storing exceptions.
// The cost includes svbLen (2 bytes), exception index (positions or bitmap),
// and StreamVByte data size estimated from the maximum high-bit width.
func estimateExceptionCost(excCount, maxHighBitWidth int) int {
	if excCount == 0 {
		return 0
	}
	var indexCost int
	if excCount <= excBitmapThreshold {
		indexCost = excCount
	} else {
		indexCost = 16
	}
	bytesPerValue := min(max((maxHighBitWidth+7)/8, 1), 4)
	controlBytes := (excCount + 3) / 4
	svbEstimate := controlBytes + excCount*bytesPerValue
	return svbLenBytes + indexCost + svbEstimate
}
