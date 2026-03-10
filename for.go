package utlpfor

import "math/bits"

// forWidthByByteCount maps byte-count (0-4) to FOR width code.
// 0 bytes -> none, 1 -> u8, 2 -> u16, 3 or 4 -> u32.
var forWidthByByteCount = [5]int{forWidthNone, forWidthU8, forWidthU16, forWidthU32, forWidthU32}

// selectFORWidth returns the smallest FOR width code that can represent minVal.
// Branchless: uses bits.Len32 (compiles to BSR/LZCNT) + LUT.
func selectFORWidth(minVal uint32) int {
	return forWidthByByteCount[(bits.Len32(minVal)+7)>>3]
}

// readFORBase reads the variable-width FOR base value from the block buffer.
// The base is located after the header (and after svbLen if exceptions are present).
func readFORBase(buf []byte, forWidth int, hasExceptions bool) uint32 {
	offset := headerBytes
	if hasExceptions {
		offset += svbLenBytes
	}
	switch forWidth {
	case forWidthU8:
		return uint32(buf[offset])
	case forWidthU16:
		return uint32(bo.Uint16(buf[offset:]))
	case forWidthU32:
		return bo.Uint32(buf[offset:])
	default:
		return 0
	}
}

// writeFORBase writes the variable-width FOR base value into the block buffer.
// The base is placed after the header (and after svbLen if exceptions are present).
func writeFORBase(buf []byte, forBase uint32, forWidth int, hasExceptions bool) {
	offset := headerBytes
	if hasExceptions {
		offset += svbLenBytes
	}
	switch forWidth {
	case forWidthU8:
		buf[offset] = byte(forBase)
	case forWidthU16:
		bo.PutUint16(buf[offset:], uint16(forBase))
	case forWidthU32:
		bo.PutUint32(buf[offset:], forBase)
	}
}

// totalBlockCost estimates the byte size of an encoded block given packing parameters.
// Used by selectBitWidthWithFOR to compare FOR vs non-FOR costs.
func totalBlockCost(bitWidth, excCount, forBaseBytes int) int {
	cost := headerBytes + utlPayloadBytesLUT[bitWidth] + forBaseBytes
	if excCount > 0 {
		cost += svbLenBytes
		if excCount > excBitmapThreshold {
			cost += 16
		} else {
			cost += excCount
		}
		// Approximate SVB data: 2 bytes per exception value is a rough estimate.
		cost += excCount * 2
	}
	return cost
}

// findMinMaxScalar computes the minimum and maximum of a uint32 slice.
func findMinMaxScalar(values []uint32) (min, max uint32) {
	min, max = values[0], values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return
}

// findMinScalar returns the minimum value in a uint32 slice.
func findMinScalar(values []uint32) uint32 {
	minVal := values[0]
	for _, v := range values[1:] {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

// findMaxScalar returns the maximum value in a uint32 slice.
func findMaxScalar(values []uint32) uint32 {
	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

// selectBitWidthWithFOR decides whether FOR compression is beneficial and returns
// the optimal packing parameters.
// It computes min/max first, then builds standard and FOR-reduced histograms in
// a single follow-up pass and compares both cost models.
func selectBitWidthWithFOR(values []uint32) (width, excCount int, useFOR bool, baseValue uint32, forWidth int) {
	minVal, maxVal := findMinMaxScalar(values)
	maxWidth := bits.Len32(maxVal)
	maxStepIdx := min((maxWidth+3)/4, 8)
	var freqs [33]int

	if minVal == 0 {
		for _, v := range values {
			freqs[bits.Len32(v)]++
		}
		stdWidth, stdExcCount, _ := chooseBestStepWidth(freqs, maxStepIdx)
		return stdWidth, stdExcCount, false, 0, forWidthNone
	}

	forMaxBW := bits.Len32(maxVal - minVal)
	forMaxStepIdx := min((forMaxBW+3)/4, 8)

	var forFreqs [33]int
	for _, v := range values {
		freqs[bits.Len32(v)]++
		forFreqs[bits.Len32(v-minVal)]++
	}

	stdWidth, stdExcCount, stdCost := chooseBestStepWidth(freqs, maxStepIdx)

	forW := selectFORWidth(minVal)
	forBaseBytes := forBaseBytesLUT[forW]

	// Quick reject: if FOR-subtract doesn't cross a step boundary,
	// it cannot reduce payload size, so the FOR overhead is pure loss.
	if roundUpToStep(forMaxBW) >= stdWidth {
		return stdWidth, stdExcCount, false, 0, forWidthNone
	}

	forBestWidth, forBestExcCount, forBestCost := chooseBestStepWidth(forFreqs, forMaxStepIdx)
	forBestCost += forBaseBytes

	if forBestCost < stdCost {
		return forBestWidth, forBestExcCount, true, minVal, forW
	}
	return stdWidth, stdExcCount, false, 0, forWidthNone
}

// chooseBestStepWidth computes the best step bitwidth for one histogram.
func chooseBestStepWidth(freqs [33]int, maxStepIdx int) (bestWidth, bestExcCount, bestCost int) {
	maxStep := stepBitWidths[maxStepIdx]
	bestWidth = maxStep
	bestCost = headerBytes + utlPayloadBytesLUT[maxStep]

	var suffix [34]int
	for bw := 32; bw >= 0; bw-- {
		suffix[bw] = suffix[bw+1] + freqs[bw]
	}

	for si := maxStepIdx - 1; si >= 0; si-- {
		w := stepBitWidths[si]
		excCount := suffix[w+1]
		if excCount == 0 {
			continue
		}
		cost := headerBytes + utlPayloadBytesLUT[w] + estimateExceptionCost(excCount, maxStep-w)
		if cost < bestCost {
			bestCost = cost
			bestWidth = w
			bestExcCount = excCount
		}
	}

	return bestWidth, bestExcCount, bestCost
}

// forSubtractScalar subtracts baseValue from each element: dst[i] = src[i] - baseValue.
func forSubtractScalar(dst, src []uint32, baseValue uint32) {
	// TODO-PERF: Maybe partially unrolled
	for i, v := range src {
		dst[i] = v - baseValue
	}
}

// forAddScalar adds baseValue to each of the first count elements: output[i] += baseValue.
func forAddScalar(output []uint32, count int, baseValue uint32) {
	for i := range count {
		output[i] += baseValue
	}
}
