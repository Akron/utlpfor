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

// readFORBase reads the variable-width FOR base value from the block buffer
// at the provided offset.
func readFORBase(buf []byte, offset, forWidth int) uint32 {
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

// svbAvgBytesPerExcNumer and svbAvgBytesPerExcDenom express the average
// StreamVByte bytes per exception as a rational number (23/10 = 2.3).
// This accounts for ~0.25 bytes control overhead plus ~2 bytes average
// data per value. Tune these constants based on profiling real workloads.
const (
	svbAvgBytesPerExcNumer = 23
	svbAvgBytesPerExcDenom = 10
)

// totalBlockCost estimates the byte size of an encoded block given packing parameters.
// Branchless: uses min() for the position-vs-bitmap decision and a hasExc flag
// to avoid branching on exception presence.
func totalBlockCost(bitWidth, excCount, forBaseBytes int) int {
	hasExc := min(excCount, 1)
	svbDataEstimate := (excCount*svbAvgBytesPerExcNumer + svbAvgBytesPerExcDenom/2) / svbAvgBytesPerExcDenom
	return headerBytes + utlPayloadBytesLUT[bitWidth] + forBaseBytes +
		hasExc*svbLenBytes + min(excCount, excBitmapThreshold) + svbDataEstimate
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

// decideFORFull determines whether FOR compression is beneficial given
// pre-computed min/max values. Uses a two-level rejection strategy:
//
//  1. min==0 -> immediate reject (no FOR base to subtract)
//  2. rawMaxStep comparison -> reject if FOR can't reduce the raw max step
//  3. stdWidth comparison -> reject if FOR can't beat the histogram-optimal step
//
// When both checks pass, FOR is guaranteed to save >=60 bytes (one step = 64
// bytes payload savings minus <=4 bytes FOR base overhead). The FOR histogram
// (second data pass) is eliminated entirely -- only the standard histogram is
// built, and only when the rawMaxStep pre-check passes.
//
// NOTE: This scalar implementation uses a separate findMinMax pass followed by
// a scatter-add histogram. The SIMD path (selectBitWidthWithFORSIMD) uses
// SIMD threshold comparisons instead, avoiding the scatter-add pattern.
func decideFORFull(values []uint32, minVal, maxVal uint32) (useFOR bool, baseValue uint32, forWidth int) {
	if minVal == 0 {
		return false, 0, forWidthNone
	}
	forMaxStep := roundUpToStep(bits.Len32(maxVal - minVal))
	rawMaxStep := roundUpToStep(bits.Len32(maxVal))
	if forMaxStep >= rawMaxStep {
		return false, 0, forWidthNone
	}
	// rawMaxStep check passed but may be inflated by outliers.
	// Build histogram to get the actual optimal step (stdWidth).
	var hist [9]int
	var orAll uint32
	for _, v := range values {
		orAll |= v
		hist[(bits.Len32(v)+3)>>2]++
	}
	maxStepIdx := int((bits.Len32(orAll) + 3) >> 2)
	stdWidth, _, _ := chooseBestFromHist(hist, maxStepIdx)
	if forMaxStep >= stdWidth {
		return false, 0, forWidthNone
	}
	return true, minVal, selectFORWidth(minVal)
}

// selectBitWidthWithFOR decides whether FOR compression is beneficial.
// Uses findMinMaxScalar for min/max, then the two-level rejection heuristic.
func selectBitWidthWithFOR(values []uint32) (useFOR bool, baseValue uint32, forWidth int) {
	minVal, maxVal := findMinMaxScalar(values)
	return decideFORFull(values, minVal, maxVal)
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
