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

// deltaEncodePerLaneScalar computes per-lane deltas in-place in UTL lane order.
// Returns true if zigzag encoding was needed (decreasing values detected).
func deltaEncodePerLaneScalar(values []uint32) bool {
	needZigZag := false
	count := len(values)

	for lane := range utlLaneCount {
		for v := utlValuesPerLane - 1; v > 0; v-- {
			cur := lane + v*utlLaneCount
			prev := lane + (v-1)*utlLaneCount
			if cur >= count || prev >= count {
				continue
			}
			if values[cur] < values[prev] {
				needZigZag = true
			}
			values[cur] = values[cur] - values[prev]
		}
	}

	if needZigZag {
		for i := range values[:count] {
			values[i] = zigzagEncode32(int32(values[i]))
		}
	}
	return needZigZag
}

// deltaDecodePerLaneScalar computes per-lane prefix sums in-place in UTL lane order.
func deltaDecodePerLaneScalar(values []uint32, useZigZag bool) {
	count := len(values)

	for lane := range utlLaneCount {
		if lane >= count {
			break
		}
		if useZigZag {
			values[lane] = uint32(zigzagDecode32(values[lane]))
		}
		for v := 1; v < utlValuesPerLane; v++ {
			cur := lane + v*utlLaneCount
			if cur >= count {
				break
			}
			if useZigZag {
				values[cur] = values[cur-utlLaneCount] + uint32(zigzagDecode32(values[cur]))
			} else {
				values[cur] = values[cur-utlLaneCount] + values[cur]
			}
		}
	}
}

// deltaDecodePerLaneWithOverflowScalar performs in-place prefix sum with overflow detection.
// Returns the lane-order index of the first overflow, or 0 if no overflow.
func deltaDecodePerLaneWithOverflowScalar(values []uint32, useZigZag bool) int {
	if useZigZag {
		deltaDecodePerLaneScalar(values, true)
		return 0
	}

	count := len(values)
	var overflowPos int

	for lane := range utlLaneCount {
		if lane >= count {
			break
		}
		for v := 1; v < utlValuesPerLane; v++ {
			cur := lane + v*utlLaneCount
			prev := lane + (v-1)*utlLaneCount
			if cur >= count {
				break
			}
			next := values[prev] + values[cur]
			if overflowPos == 0 && next < values[prev] {
				overflowPos = cur
			}
			values[cur] = next
		}
	}
	return overflowPos
}
