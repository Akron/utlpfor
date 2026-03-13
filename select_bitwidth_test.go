package utlpfor

import (
	"math/bits"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- selectBitWidth SIMD-matches-scalar Tests ---

func TestSelectBitWidth_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(77, 0))
	for trial := range 500 {
		values := make([]uint32, 128)
		for i := range values {
			values[i] = rng.Uint32()
		}

		scalarWidth, scalarExc := selectBitWidth(values)
		simdWidth, simdExc := selectBitWidthSIMDtest(values)

		assert.Equal(t, scalarWidth, simdWidth,
			"width mismatch trial %d", trial)
		assert.Equal(t, scalarExc, simdExc,
			"excCount mismatch trial %d", trial)
	}
}

func TestSelectBitWidth_SIMDMatchesScalar_SmallValues(t *testing.T) {
	for bw := 0; bw <= 32; bw++ {
		values := make([]uint32, 128)
		mask := uint32((1 << bw) - 1)
		if bw == 32 {
			mask = 0xFFFFFFFF
		}
		for i := range values {
			values[i] = uint32(i) & mask
		}
		scalarWidth, scalarExc := selectBitWidth(values)
		simdWidth, simdExc := selectBitWidthSIMDtest(values)
		assert.Equal(t, scalarWidth, simdWidth, "bw=%d", bw)
		assert.Equal(t, scalarExc, simdExc, "bw=%d", bw)
	}
}

func TestSelectBitWidth_SIMDMatchesScalar_WithOutliers(t *testing.T) {
	rng := rand.New(rand.NewPCG(55, 0))
	for trial := range 200 {
		values := make([]uint32, 128)
		baseBW := rng.IntN(28) + 1
		for i := range values {
			values[i] = rng.Uint32() & ((1 << baseBW) - 1)
		}
		outliers := rng.IntN(32) + 1
		for range outliers {
			values[rng.IntN(128)] = rng.Uint32()
		}

		scalarWidth, scalarExc := selectBitWidth(values)
		simdWidth, simdExc := selectBitWidthSIMDtest(values)
		assert.Equal(t, scalarWidth, simdWidth,
			"trial %d", trial)
		assert.Equal(t, scalarExc, simdExc,
			"trial %d", trial)
	}
}

// --- selectBitWidthWithFOR SIMD-matches-scalar Tests ---

func TestSelectBitWidthWithFOR_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(33, 0))
	for trial := range 500 {
		base := rng.Uint32() >> 8
		spread := uint32(rng.IntN(4096))
		values := make([]uint32, 128)
		for i := range values {
			values[i] = base + uint32(rng.IntN(int(spread)+1))
		}

		sf, sb, sfw := selectBitWidthWithFOR(values)
		ifr, ib, ifw := selectBitWidthWithFORSIMDtest(values)

		assert.Equal(t, sf, ifr, "useFOR trial %d", trial)
		assert.Equal(t, sb, ib, "baseValue trial %d", trial)
		assert.Equal(t, sfw, ifw, "forWidth trial %d", trial)
	}
}

func TestSelectBitWidthWithFOR_SIMDMatchesScalar_MinZero(t *testing.T) {
	rng := rand.New(rand.NewPCG(88, 0))
	for trial := range 200 {
		values := make([]uint32, 128)
		for i := range values {
			values[i] = rng.Uint32()
		}
		values[rng.IntN(128)] = 0

		sf, _, _ := selectBitWidthWithFOR(values)
		ifr, _, _ := selectBitWidthWithFORSIMDtest(values)

		assert.Equal(t, sf, ifr, "useFOR trial %d", trial)
	}
}

func TestSelectBitWidthWithFOR_SIMDMatchesScalar_Clustered(t *testing.T) {
	rng := rand.New(rand.NewPCG(44, 0))
	for trial := range 200 {
		base := uint32(rng.IntN(10000000)) + 1000
		spread := uint32(rng.IntN(200))
		values := make([]uint32, 128)
		for i := range values {
			values[i] = base + uint32(rng.IntN(int(spread)+1))
		}

		sf, sb, sfw := selectBitWidthWithFOR(values)
		ifr, ib, ifw := selectBitWidthWithFORSIMDtest(values)

		assert.Equal(t, sf, ifr, "useFOR trial %d", trial)
		assert.Equal(t, sb, ib, "baseValue trial %d", trial)
		assert.Equal(t, sfw, ifw, "forWidth trial %d", trial)
	}
}

// --- Benchmarks ---

func BenchmarkSelectBitWidth_Scalar(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 7)
	}
	for b.Loop() {
		selectBitWidth(values)
	}
}

func BenchmarkSelectBitWidth_SIMD(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 7)
	}
	for b.Loop() {
		selectBitWidthSIMDtest(values)
	}
}

func BenchmarkSelectBitWidthWithFOR_Scalar(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	var useFOR bool
	for b.Loop() {
		useFOR, _, _ = selectBitWidthWithFOR(values)
	}
	_ = useFOR
}

func BenchmarkSelectBitWidthWithFOR_SIMD(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	var useFOR bool
	for b.Loop() {
		useFOR, _, _ = selectBitWidthWithFORSIMDtest(values)
	}
	_ = useFOR
}

// --- Unit Tests ---

func TestSelectBitWidth_AllSameWidth(t *testing.T) {
	for bw := 0; bw <= 32; bw++ {
		values := make([]uint32, 128)
		mask := uint32((1 << bw) - 1)
		if bw == 32 {
			mask = 0xFFFFFFFF
		}
		for i := range values {
			values[i] = mask
		}
		width, excCount := selectBitWidth(values)
		wantStep := roundUpToStep(bw)
		assert.Equal(t, wantStep, width, "bw=%d", bw)
		assert.Equal(t, 0, excCount, "bw=%d", bw)
	}
}

func TestSelectBitWidth_AllZeros(t *testing.T) {
	values := make([]uint32, 128)
	width, excCount := selectBitWidth(values)
	assert.Equal(t, 0, width)
	assert.Equal(t, 0, excCount)
}

func TestSelectBitWidth_FewExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[0] = 0xFFFFFFFF
	width, excCount := selectBitWidth(values)
	assert.Less(t, width, 32, "should choose lower width with exceptions")
	assert.Greater(t, excCount, 0)
}

func TestSelectBitWidth_ExceptionsVsFullWidth(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 0x10000000 + uint32(i)
	}
	width, _ := selectBitWidth(values)
	assert.True(t, width >= 28, "all-large values should use high bit width")
}

func TestSelectBitWidth_ZeroPlusExceptions(t *testing.T) {
	values := make([]uint32, 128)
	values[10] = 0xFFFF
	values[50] = 0xFFFFF
	width, excCount := selectBitWidth(values)
	assert.Less(t, width, 20,
		"mostly zeros with few exceptions should choose low width")
	assert.Greater(t, excCount, 0)
}

func TestSelectBitWidth_MatchesStepBitWidthForUniform(t *testing.T) {
	for bw := 0; bw <= 32; bw++ {
		values := make([]uint32, 128)
		mask := uint32((1 << bw) - 1)
		if bw == 32 {
			mask = 0xFFFFFFFF
		}
		for i := range values {
			values[i] = uint32(i) & mask
		}
		width, excCount := selectBitWidth(values)
		var orAll uint32
		for _, v := range values {
			orAll |= v
		}
		maxBW := bits.Len32(orAll)
		wantStep := roundUpToStep(maxBW)
		assert.Equal(t, wantStep, width,
			"uniform data should match step-rounded maxBitWidth (bw=%d)", bw)
		assert.Equal(t, 0, excCount, "bw=%d", bw)
	}
}

func TestSelectBitWidth_SingleException(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[42] = 0x100000
	width, excCount := selectBitWidth(values)
	assert.Less(t, width, 24, "should pick narrower width with single outlier")
	assert.Greater(t, excCount, 0)
}

func TestSelectBitWidth_AlwaysReturnsStep(t *testing.T) {
	stepSet := map[int]bool{0: true, 4: true, 8: true, 12: true,
		16: true, 20: true, 24: true, 28: true, 32: true}
	rng := rand.New(rand.NewPCG(42, 0))
	for trial := range 1000 {
		values := make([]uint32, 128)
		for i := range values {
			values[i] = rng.Uint32()
		}
		width, _ := selectBitWidth(values)
		require.True(t, stepSet[width],
			"selectBitWidth returned non-step width %d (trial %d)", width, trial)
	}
}

func TestSelectBitWidth_RoundUpToStep(t *testing.T) {
	tests := []struct {
		bw   int
		want int
	}{
		{0, 0}, {1, 4}, {2, 4}, {3, 4}, {4, 4},
		{5, 8}, {6, 8}, {7, 8}, {8, 8},
		{9, 12}, {12, 12}, {13, 16}, {16, 16},
		{17, 20}, {20, 20}, {21, 24}, {24, 24},
		{25, 28}, {28, 28}, {29, 32}, {32, 32},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, roundUpToStep(tt.bw), "roundUpToStep(%d)", tt.bw)
	}
}
