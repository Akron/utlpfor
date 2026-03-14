//go:build goexperiment.simd && amd64

package utlpfor

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
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

// --- SIMD Benchmarks ---

func BenchmarkSelectBitWidth_SIMD(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 7)
	}
	for b.Loop() {
		selectBitWidthSIMDtest(values)
	}
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

func BenchmarkFindMinMaxSSE2(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	for b.Loop() {
		findMinMaxSSE2(values)
	}
}

func BenchmarkFindMinMaxAVX2(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	for b.Loop() {
		findMinMaxAVX2(values)
	}
}

func BenchmarkFindMinMaxAVX512(b *testing.B) {
	if simdLevel < simdLevelAVX512 {
		b.Skip("AVX-512 not available")
	}
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	for b.Loop() {
		findMinMaxAVX512(values)
	}
}

func BenchmarkBuildExcCountsSSE2(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 7)
	}
	for b.Loop() {
		buildExcCountsSSE2(values)
	}
}

func BenchmarkSelectBitWidthWithFOR_SSE2(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	var useFOR bool
	var bv uint32
	var fw int
	for b.Loop() {
		useFOR, bv, fw = selectBitWidthWithFORSSE2(values)
	}
	_ = useFOR
	_ = bv
	_ = fw
}

func BenchmarkSelectBitWidthWithFOR_AVX2(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	var useFOR bool
	var bv uint32
	var fw int
	for b.Loop() {
		useFOR, bv, fw = selectBitWidthWithFORAVX2(values)
	}
	_ = useFOR
	_ = bv
	_ = fw
}

func BenchmarkBuildExcCountsAVX2(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 7)
	}
	for b.Loop() {
		buildExcCountsAVX2(values)
	}
}
