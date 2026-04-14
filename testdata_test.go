package utlpfor

import "math/rand"

// stepBitWidths lists all valid step bitwidths for test iteration.
var stepBitWidths = [9]int{0, 4, 8, 12, 16, 20, 24, 28, 32}

// genSequential returns n sequentially increasing uint32 values [0, 1, 2, ...].
// Matches fastpfor-go's genSequential for cross-repo benchmark comparison.
func genSequential(n int) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = uint32(i)
	}
	return out
}

// genMonotonic returns n monotonically increasing uint32 values with small steps.
// Matches fastpfor-go's genMonotonic for cross-repo benchmark comparison.
func genMonotonic(n int) []uint32 {
	out := make([]uint32, n)
	var acc uint32
	for i := range out {
		acc += uint32(i%7 + 1)
		out[i] = acc
	}
	return out
}

// genMixed returns n uint32 values with random fluctuations around a baseline.
// Matches fastpfor-go's genMixed for cross-repo benchmark comparison.
func genMixed(n int) []uint32 {
	out := make([]uint32, n)
	rng := rand.New(rand.NewSource(1234))
	acc := int64(1 << 20)
	for i := range out {
		gain := rng.Intn(4096)
		loss := rng.Intn(4096)
		acc += int64(gain - loss)
		if acc < 0 {
			acc = int64(rng.Intn(1 << 16))
		}
		out[i] = uint32(acc)
	}
	return out
}

// genDataWithSmallExceptions returns 128 values with ~10 small exceptions.
// Matches fastpfor-go's genDataWithSmallExceptions for cross-repo comparison.
func genDataWithSmallExceptions() []uint32 {
	out := make([]uint32, blockSize)
	for i := range out {
		out[i] = uint32(i % 256)
	}
	for i := range 10 {
		out[i*12] = 256 + uint32(i*10)
	}
	return out
}

// genDataWithLargeExceptions returns 128 values with ~20 large exceptions.
// Matches fastpfor-go's genDataWithLargeExceptions for cross-repo comparison.
func genDataWithLargeExceptions() []uint32 {
	out := make([]uint32, blockSize)
	for i := range 20 {
		out[i*6] = 0xFFFFFFFF - uint32(i*1000)
	}
	return out
}
