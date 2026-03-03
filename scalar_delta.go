package utlpfor

// scalar_delta.go contains scalar per-lane delta and zigzag encode/decode.
//
// The per-lane delta approach follows the FastLanes paper (Algorithm 4):
//   Afroozeh, A. & Muehlbauer, P. (2023). "FastLanes: An SIMD-Friendly
//   Layout for Analytic Databases." VLDB 2023.
//   https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf
//   Reference: https://github.com/cwida/FastLanes

// zigzagEncode32 encodes a signed int32 as an unsigned uint32.
func zigzagEncode32(v int32) uint32 {
	return uint32(v<<1) ^ uint32(v>>31)
}

// zigzagDecode32 decodes a zigzag-encoded uint32 back to signed int32.
func zigzagDecode32(v uint32) int32 {
	return int32(v>>1) ^ -int32(v&1)
}

// deltaEncodePerLaneScalar computes per-lane deltas in UTL lane order.
// Returns true if zigzag encoding was needed (decreasing values detected).
func deltaEncodePerLaneScalar(dst, src []uint32) bool {
	needZigZag := false
	count := len(src)

	for lane := range utlLaneCount {
		// Process values from last to first within the lane to avoid
		// overwriting values we still need (when dst == src).
		for v := utlValuesPerLane - 1; v > 0; v-- {
			cur := lane + v*utlLaneCount
			prev := lane + (v-1)*utlLaneCount
			if cur >= count || prev >= count {
				continue
			}
			if src[cur] < src[prev] {
				needZigZag = true
			}
			dst[cur] = src[cur] - src[prev]
		}
		// Lane base value is preserved as-is.
		if lane < count {
			dst[lane] = src[lane]
		}
	}

	if needZigZag {
		for i := range dst[:count] {
			dst[i] = zigzagEncode32(int32(dst[i]))
		}
	}
	return needZigZag
}

// deltaDecodePerLaneScalar computes per-lane prefix sums in UTL lane order.
func deltaDecodePerLaneScalar(dst, deltas []uint32, useZigZag bool) {
	count := len(deltas)

	for lane := range utlLaneCount {
		if lane >= count {
			break
		}
		if useZigZag {
			dst[lane] = uint32(zigzagDecode32(deltas[lane]))
		} else {
			dst[lane] = deltas[lane]
		}
		for v := 1; v < utlValuesPerLane; v++ {
			cur := lane + v*utlLaneCount
			if cur >= count {
				break
			}
			if useZigZag {
				dst[cur] = dst[cur-utlLaneCount] + uint32(zigzagDecode32(deltas[cur]))
			} else {
				dst[cur] = dst[cur-utlLaneCount] + deltas[cur]
			}
		}
	}
}

// deltaDecodePerLaneWithOverflowScalar performs prefix sum with overflow detection.
// Returns the lane-order index of the first overflow, or 0 if no overflow.
func deltaDecodePerLaneWithOverflowScalar(dst, deltas []uint32, useZigZag bool) int {
	if useZigZag {
		deltaDecodePerLaneScalar(dst, deltas, true)
		return 0
	}

	count := len(deltas)
	var overflowPos int

	for lane := range utlLaneCount {
		if lane >= count {
			break
		}
		dst[lane] = deltas[lane]
		for v := 1; v < utlValuesPerLane; v++ {
			cur := lane + v*utlLaneCount
			prev := lane + (v-1)*utlLaneCount
			if cur >= count {
				break
			}
			next := dst[prev] + deltas[cur]
			if overflowPos == 0 && next < dst[prev] {
				overflowPos = cur
			}
			dst[cur] = next
		}
	}
	return overflowPos
}
