package utlpfor

import "math/bits"

// maxBitWidth returns the minimum number of bits needed to represent
// the largest value in the slice.
func maxBitWidth(values []uint32) int {
	var ored uint32
	for _, v := range values {
		ored |= v
	}
	return bits.Len32(ored)
}

// selectBitWidth finds the optimal bit width that minimizes total block size.
// It sweeps from the maximum needed width downward, accumulating exception
// counts and comparing payload savings against exception overhead.
// Returns the chosen width and the number of exceptions at that width.
func selectBitWidth(values []uint32) (width int, excCount int) {
	var freqs [33]int
	var orAll uint32
	for _, v := range values {
		freqs[bits.Len32(v)]++
		orAll |= v
	}
	maxWidth := bits.Len32(orAll)

	bestWidth := maxWidth
	bestSize := headerBytes + utlPayloadBytesLUT[maxWidth]
	bestExcCount := 0

	excCountCum := 0
	for w := maxWidth - 1; w >= 0; w-- {
		excCountCum += freqs[w+1]
		payloadCost := utlPayloadBytesLUT[w]
		excOverhead := estimateExceptionCost(excCountCum, maxWidth-w)
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
