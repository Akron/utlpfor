//go:build goexperiment.simd && amd64

package utlpfor

import (
	"math/bits"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// bit-exactness harness
// These tests pin the SIMD helper loops to pointer-based
// loads + hoisted broadcasts: every helper is compared against its scalar
// reference across all bitwidths/lengths so the mechanical rewrite cannot
// silently change results.

// gtCountU32Scalar is the scalar reference for gtCountU32.
func gtCountU32Scalar(v, threshold uint32) int {
	if v > threshold {
		return 1
	}
	return 0
}

// buildExcCountsScalar computes cumulative exception counts with a plain
// per-value pass; the SIMD variants must return exactly this array.
func buildExcCountsScalar(values []uint32) (exc [9]int) {
	for _, v := range values {
		exc[0] += gtCountU32Scalar(v, 0)
		exc[1] += gtCountU32Scalar(v, 0xF)
		exc[2] += gtCountU32Scalar(v, 0xFF)
		exc[3] += gtCountU32Scalar(v, 0xFFF)
		exc[4] += gtCountU32Scalar(v, 0xFFFF)
		exc[5] += gtCountU32Scalar(v, 0xFFFFF)
		exc[6] += gtCountU32Scalar(v, 0xFFFFFF)
		exc[7] += gtCountU32Scalar(v, 0xFFFFFFF)
	}
	return
}

// TestBuildExcCounts_SIMDMatchesScalar compares each SIMD buildExcCounts
// variant against the scalar reference over values with outliers at every
// possible threshold (all step widths are represented).
func TestBuildExcCounts_SIMDMatchesScalar(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	rng := rand.New(rand.NewSource(2024))
	for trial := range 50 {
		// Mix small values (fit in low thresholds) with outliers, so every
		// threshold bin sees both counted and uncounted values.
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = rng.Uint32() & 0xFF
		}
		for range 20 {
			values[rng.Intn(blockSize)] = rng.Uint32()
		}

		want := buildExcCountsScalar(values)

		if simdLevel >= simdLevelSSE2 {
			assert.Equal(t, want, buildExcCountsSSE2(values), "SSE2 trial %d", trial)
		}
		if simdLevel >= simdLevelAVX2 {
			assert.Equal(t, want, buildExcCountsAVX2(values), "AVX2 trial %d", trial)
		}
		if simdLevel >= simdLevelAVX512 {
			assert.Equal(t, want, buildExcCountsAVX512(values), "AVX512 trial %d", trial)
		}
	}
}

// TestBuildExcCounts_SIMDMatchesScalar_Lengths exercises short lengths,
// exercising the scalar tails of each SIMD variant (n%step != 0 and n<step).
func TestBuildExcCounts_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	rng := rand.New(rand.NewSource(99))
	for n := range 20 {
		values := make([]uint32, n)
		for i := range values {
			// Borderline values around the step thresholds.
			values[i] = uint32(0xFF0 + rng.Intn(0x20))
		}
		want := buildExcCountsScalar(values)

		if simdLevel >= simdLevelSSE2 {
			assert.Equal(t, want, buildExcCountsSSE2(values), "SSE2 len=%d", n)
		}
		if simdLevel >= simdLevelAVX2 {
			assert.Equal(t, want, buildExcCountsAVX2(values), "AVX2 len=%d", n)
		}
		if simdLevel >= simdLevelAVX512 {
			assert.Equal(t, want, buildExcCountsAVX512(values), "AVX512 len=%d", n)
		}
	}
}

// TestSelectBitWidthNoPatch_SIMDMatchesScalar pins the OR-reduce helper used
// by the NoPatch pack path against the scalar selectBitWidthNoPatch.
func TestSelectBitWidthNoPatch_SIMDMatchesScalar(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	rng := rand.New(rand.NewSource(3))
	for trial := range 100 {
		// Random values with a shared low-bit mask: the OR result lands in
		// a range of step widths rather than always 32.
		baseBW := rng.Intn(30) + 1
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = rng.Uint32() & ((1 << baseBW) - 1)
		}

		want := selectBitWidthNoPatch(values)
		if simdLevel >= simdLevelSSE2 {
			assert.Equal(t, want, selectBitWidthNoPatchSSE2(values), "SSE2 trial %d", trial)
		}
		if simdLevel >= simdLevelAVX2 {
			assert.Equal(t, want, selectBitWidthNoPatchAVX2(values), "AVX2 trial %d", trial)
		}
		if simdLevel >= simdLevelAVX512 {
			assert.Equal(t, want, selectBitWidthNoPatchAVX512(values), "AVX512 trial %d", trial)
		}
	}
}

// TestSelectBitWidthNoPatch_SIMDMatchesScalar_Lengths exercises short
// lengths to cover the scalar tails of the OR-reduce loops.
func TestSelectBitWidthNoPatch_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	for n := range 20 {
		values := make([]uint32, n)
		for i := range values {
			values[i] = uint32(1)<<bits.Len32(uint32(max(n, 1)-1)) - 1 | uint32(i)
		}
		want := selectBitWidthNoPatch(values)
		if simdLevel >= simdLevelSSE2 {
			assert.Equal(t, want, selectBitWidthNoPatchSSE2(values), "SSE2 len=%d", n)
		}
		if simdLevel >= simdLevelAVX2 {
			assert.Equal(t, want, selectBitWidthNoPatchAVX2(values), "AVX2 len=%d", n)
		}
		if simdLevel >= simdLevelAVX512 {
			assert.Equal(t, want, selectBitWidthNoPatchAVX512(values), "AVX512 len=%d", n)
		}
	}
}

// TestZigzag_SIMDMatchesScalar_Lengths covers zigzag encode/decode for
// lengths 0..blockSize+7 (both SIMD body and scalar tail plus beyond-block
// lengths used when callers pass unaligned sizes).
func TestZigzag_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	rng := rand.New(rand.NewSource(7))
	for n := range blockSize + 8 {
		orig := make([]uint32, n)
		for i := range orig {
			orig[i] = rng.Uint32()
		}

		encWant := make([]uint32, n)
		copy(encWant, orig)
		zigzagEncodeSlice(encWant, n)

		decWant := make([]uint32, n)
		zigzagDecodeSlice(decWant, encWant)

		if simdLevel >= simdLevelSSE2 {
			buf := make([]uint32, n)
			copy(buf, orig)
			zigzagEncodeSSE2(buf, n)
			assert.Equal(t, encWant, buf, "SSE2 encode len=%d", n)
			zigzagDecodeSSE2(buf)
			assert.Equal(t, decWant, buf, "SSE2 decode len=%d", n)
		}
		if simdLevel >= simdLevelAVX2 {
			buf := make([]uint32, n)
			copy(buf, orig)
			zigzagEncodeAVX2(buf, n)
			assert.Equal(t, encWant, buf, "AVX2 encode len=%d", n)
			zigzagDecodeAVX2(buf)
			assert.Equal(t, decWant, buf, "AVX2 decode len=%d", n)
		}
	}
}

// TestAllFitIn32Bits_SIMDMatchesScalar pins the uint64 OR-accumulate
// helpers against the scalar reference (upper-half decision).
func TestAllFitIn32Bits_SIMDMatchesScalar(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	cases := []struct {
		name string
		gen  func(i int) uint64
		last uint64 // value for the final slot (mixed tail)
	}{
		{"all32", func(i int) uint64 { return uint64(i * 7) }, 0xFFFFFFFF},
		{"above32", func(i int) uint64 { return 0x1_0000_0000 + uint64(i) }, 0x1_0000_0000},
		{"mixed", func(i int) uint64 { return uint64(i) }, 0x1_0000_0000},
		{"zeros", func(i int) uint64 { return 0 }, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for n := range blockSize + 1 {
				values := make([]uint64, n)
				for i := range values {
					values[i] = tc.gen(i)
				}
				if n > 0 {
					// Force a mixed tail so the final accumulator reduction matters.
					values[n-1] = tc.last
				}

				// Scalar reference: OR-accumulate all values, test upper half.
				var acc uint64
				for _, v := range values {
					acc |= v
				}
				want := acc>>32 == 0

				if simdLevel >= simdLevelSSE2 {
					assert.Equal(t, want, allFitIn32BitsSSE2(values), "SSE2 len=%d", n)
				}
				if simdLevel >= simdLevelAVX2 {
					assert.Equal(t, want, allFitIn32BitsAVX2(values), "AVX2 len=%d", n)
				}
				if simdLevel >= simdLevelAVX512 {
					assert.Equal(t, want, allFitIn32BitsAVX512(values), "AVX512 len=%d", n)
				}
			}
		})
	}
}

// TestForSubtractAdd_SIMDMatchesScalar_Lengths covers the scalar tails of
// the FOR subtract/add helpers at lengths that leave remainders for
// SSE2 (4-wide), AVX2 (8-wide), and AVX-512 (16-wide) chunks.
func TestForSubtractAdd_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	rng := rand.New(rand.NewSource(77))
	for n := range blockSize + 17 {
		base := rng.Uint32() >> 4
		values := make([]uint32, n)
		for i := range values {
			values[i] = base + rng.Uint32()&0xFFF
		}

		subWant := make([]uint32, n)
		forSubtractScalar(subWant, values, base)
		addWant := make([]uint32, n)
		copy(addWant, values)
		forAddScalar(addWant, n, base)

		if simdLevel >= simdLevelSSE2 {
			sub := make([]uint32, n)
			forSubtractSSE2(sub, values, base)
			assert.Equal(t, subWant, sub, "SSE2 subtract len=%d", n)
			add := make([]uint32, n)
			copy(add, values)
			forAddSSE2(add, n, base)
			assert.Equal(t, addWant, add, "SSE2 add len=%d", n)
		}
		if simdLevel >= simdLevelAVX2 {
			sub := make([]uint32, n)
			forSubtractAVX2(sub, values, base)
			assert.Equal(t, subWant, sub, "AVX2 subtract len=%d", n)
			add := make([]uint32, n)
			copy(add, values)
			forAddAVX2(add, n, base)
			assert.Equal(t, addWant, add, "AVX2 add len=%d", n)
		}
		if simdLevel >= simdLevelAVX512 {
			sub := make([]uint32, n)
			forSubtractAVX512(sub, values, base)
			assert.Equal(t, subWant, sub, "AVX512 subtract len=%d", n)
			add := make([]uint32, n)
			copy(add, values)
			forAddAVX512(add, n, base)
			assert.Equal(t, addWant, add, "AVX512 add len=%d", n)
		}
	}
}

// TestNarrowToUint32_SIMDMatchesScalar_Lengths covers narrowToUint32
// conversions across lengths including the scalar tails.
func TestNarrowToUint32_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelAVX2 {
		t.Skip("SIMD not available")
	}

	for n := range blockSize + 9 {
		values := make([]uint64, n)
		for i := range values {
			values[i] = uint64(i*13 + 7)
		}
		want := make([]uint32, n)
		narrowToUint32(want, values, n)

		dst := make([]uint32, n)
		narrowToUint32AVX2(dst, values, n)
		assert.Equal(t, want, dst, "AVX2 len=%d", n)

		if simdLevel >= simdLevelAVX512 {
			dst512 := make([]uint32, n)
			narrowToUint32AVX512(dst512, values, n)
			assert.Equal(t, want, dst512, "AVX512 len=%d", n)
		}
	}
}

// TestAnalyzeUint64_SIMDMatchesScalar_Lengths extends the existing
// analyzeUint64 test with short lengths covering the scalar tails.
func TestAnalyzeUint64_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	rng := rand.New(rand.NewSource(321))
	// n starts at 1: the scalar reference requires at least one value
	// (analyzeUint64 indexes values[0] unconditionally).
	for n := 1; n <= blockSize+4; n++ {
		values := make([]uint64, n)
		for i := range values {
			values[i] = uint64(rng.Int63()) | 1<<40 // exercise upper halves too
		}
		wantMin, wantMax, wantAcc := analyzeUint64(values)

		if simdLevel >= simdLevelSSE2 {
			gotMin, gotMax, gotAcc := analyzeUint64SSE2(values)
			assert.Equal(t, wantMin, gotMin, "SSE2 min len=%d", n)
			assert.Equal(t, wantMax, gotMax, "SSE2 max len=%d", n)
			assert.Equal(t, wantAcc, gotAcc, "SSE2 acc len=%d", n)
		}
		if simdLevel >= simdLevelAVX2 {
			gotMin, gotMax, gotAcc := analyzeUint64AVX2(values)
			assert.Equal(t, wantMin, gotMin, "AVX2 min len=%d", n)
			assert.Equal(t, wantMax, gotMax, "AVX2 max len=%d", n)
			assert.Equal(t, wantAcc, gotAcc, "AVX2 acc len=%d", n)
		}
		if simdLevel >= simdLevelAVX512 {
			gotMin, gotMax, gotAcc := analyzeUint64AVX512(values)
			assert.Equal(t, wantMin, gotMin, "AVX512 min len=%d", n)
			assert.Equal(t, wantMax, gotMax, "AVX512 max len=%d", n)
			assert.Equal(t, wantAcc, gotAcc, "AVX512 acc len=%d", n)
		}
	}
}

// TestCombineUint64_SIMDMatchesScalar_Lengths covers the combine helpers
// (used on the uint64 two-block unpack path) across lengths and tails.
func TestCombineUint64_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	for n := range blockSize + 9 {
		lower := make([]uint32, n)
		upper := make([]uint32, n)
		for i := range lower {
			lower[i] = uint32(i * 3)
			upper[i] = uint32(i*5 + 1)
		}
		want := make([]uint64, n)
		combineUint64Scalar(want, lower, upper, n)

		if simdLevel >= simdLevelSSE2 {
			dst := make([]uint64, n)
			combineUint64SSE2(dst, lower, upper, n)
			assert.Equal(t, want, dst, "SSE2 len=%d", n)
		}
		if simdLevel >= simdLevelAVX2 {
			dst := make([]uint64, n)
			combineUint64AVX2(dst, lower, upper, n)
			assert.Equal(t, want, dst, "AVX2 len=%d", n)
		}
		if simdLevel >= simdLevelAVX512 {
			dst := make([]uint64, n)
			combineUint64AVX512(dst, lower, upper, n)
			assert.Equal(t, want, dst, "AVX512 len=%d", n)
		}
	}
}

// TestForSubtract64Add64_SIMDMatchesScalar_Lengths covers the uint64
// FOR subtract/add helpers across lengths, verifying the pointer-based
// loads + off/2 offset arithmetic produce results matching the scalar reference.
func TestForSubtract64Add64_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	rng := rand.New(rand.NewSource(64))
	for n := range blockSize + 9 {
		base := uint64(rng.Int63() >> 16)
		values := make([]uint64, n)
		for i := range values {
			values[i] = base + uint64(rng.Intn(0xFFFF))
		}

		// Scalar reference for forSubtract64.
		subWant := make([]uint32, n)
		for i, v := range values {
			subWant[i] = uint32(v - base)
		}

		if simdLevel >= simdLevelSSE2 {
			sub := make([]uint32, n)
			forSubtract64SSE2(sub, values, base)
			assert.Equal(t, subWant, sub, "SSE2 subtract64 len=%d", n)
		}
		if simdLevel >= simdLevelAVX2 {
			sub := make([]uint32, n)
			forSubtract64AVX2(sub, values, base)
			assert.Equal(t, subWant, sub, "AVX2 subtract64 len=%d", n)
		}
		if simdLevel >= simdLevelAVX512 {
			sub := make([]uint32, n)
			forSubtract64AVX512(sub, values, base)
			assert.Equal(t, subWant, sub, "AVX512 subtract64 len=%d", n)
		}

		// Scalar reference for forAdd64.
		src32 := make([]uint32, n)
		for i := range src32 {
			src32[i] = uint32(rng.Int31())
		}
		addWant := make([]uint64, n)
		for i := range n {
			addWant[i] = uint64(src32[i]) + base
		}

		if simdLevel >= simdLevelSSE2 {
			dst := make([]uint64, n)
			forAdd64SSE2(dst, src32, base, n)
			assert.Equal(t, addWant, dst, "SSE2 add64 len=%d", n)
		}
		if simdLevel >= simdLevelAVX2 {
			dst := make([]uint64, n)
			forAdd64AVX2(dst, src32, base, n)
			assert.Equal(t, addWant, dst, "AVX2 add64 len=%d", n)
		}
		if simdLevel >= simdLevelAVX512 {
			dst := make([]uint64, n)
			forAdd64AVX512(dst, src32, base, n)
			assert.Equal(t, addWant, dst, "AVX512 add64 len=%d", n)
		}
	}
}

// TestSplitUint64_SIMDMatchesScalar_Lengths covers splitUint64AVX512
// (the AVX2 variant has a separate test in simd_pack_test.go).
func TestSplitUint64_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelAVX512 {
		t.Skip("AVX-512 not available")
	}

	for n := range blockSize + 9 {
		values := make([]uint64, n)
		for i := range values {
			values[i] = uint64(i)*0x100000003 + 42
		}
		wantLo := make([]uint32, n)
		wantHi := make([]uint32, n)
		for i, v := range values {
			wantLo[i] = uint32(v)
			wantHi[i] = uint32(v >> 32)
		}

		gotLo := make([]uint32, n)
		gotHi := make([]uint32, n)
		splitUint64AVX512(gotLo, gotHi, values, n)
		assert.Equal(t, wantLo, gotLo, "AVX512 splitLo len=%d", n)
		assert.Equal(t, wantHi, gotHi, "AVX512 splitHi len=%d", n)
	}
}

// TestFindMinMax_SIMDMatchesScalar_Lengths covers the SIMD findMinMax
// helpers across lengths, exercising the scalar tails and dual-accumulator
// reduction paths.
func TestFindMinMax_SIMDMatchesScalar_Lengths(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	rng := rand.New(rand.NewSource(111))
	for n := 1; n <= blockSize+8; n++ {
		values := make([]uint32, n)
		for i := range values {
			values[i] = rng.Uint32()
		}
		wantMin, wantMax := findMinMaxScalar(values)

		if simdLevel >= simdLevelSSE2 {
			gotMin, gotMax := findMinMaxSSE2(values)
			assert.Equal(t, wantMin, gotMin, "SSE2 min len=%d", n)
			assert.Equal(t, wantMax, gotMax, "SSE2 max len=%d", n)
		}
		if simdLevel >= simdLevelAVX2 {
			gotMin, gotMax := findMinMaxAVX2(values)
			assert.Equal(t, wantMin, gotMin, "AVX2 min len=%d", n)
			assert.Equal(t, wantMax, gotMax, "AVX2 max len=%d", n)
		}
		if simdLevel >= simdLevelAVX512 {
			gotMin, gotMax := findMinMaxAVX512(values)
			assert.Equal(t, wantMin, gotMin, "AVX512 min len=%d", n)
			assert.Equal(t, wantMax, gotMax, "AVX512 max len=%d", n)
		}
	}
}

// TestSelectBitWidthWithFOR_SIMDMatchesScalar_AllLevels extends the
// existing FOR selection test to cover all four SIMD dispatch levels
// (the previous test only exercised the runtime-level dispatch).
func TestSelectBitWidthWithFOR_SIMDMatchesScalar_AllLevels(t *testing.T) {
	if simdLevel < simdLevelSSE2 {
		t.Skip("SIMD not available")
	}

	rng := rand.New(rand.NewSource(33))
	for trial := range 200 {
		base := rng.Uint32() >> 8
		spread := uint32(rng.Intn(4096))
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = base + uint32(rng.Intn(int(spread)+1))
		}

		useFORWant, baseWant, widthWant := selectBitWidthWithFOR(values)
		if simdLevel >= simdLevelSSE2 {
			g, b, w := selectBitWidthWithFORSSE2(values)
			assert.Equal(t, useFORWant, g, "SSE2 useFOR trial %d", trial)
			assert.Equal(t, baseWant, b, "SSE2 base trial %d", trial)
			assert.Equal(t, widthWant, w, "SSE2 width trial %d", trial)
		}
		if simdLevel >= simdLevelAVX2 {
			g, b, w := selectBitWidthWithFORAVX2(values)
			assert.Equal(t, useFORWant, g, "AVX2 useFOR trial %d", trial)
			assert.Equal(t, baseWant, b, "AVX2 base trial %d", trial)
			assert.Equal(t, widthWant, w, "AVX2 width trial %d", trial)
		}
		if simdLevel >= simdLevelAVX512 {
			g, b, w := selectBitWidthWithFORAVX512(values)
			assert.Equal(t, useFORWant, g, "AVX512 useFOR trial %d", trial)
			assert.Equal(t, baseWant, b, "AVX512 base trial %d", trial)
			assert.Equal(t, widthWant, w, "AVX512 width trial %d", trial)
		}
	}
}
