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
