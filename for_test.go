package utlpfor

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHeaderFORWidth_Encode(t *testing.T) {
	tests := []struct {
		name     string
		forWidth int
	}{
		{"none", forWidthNone},
		{"u8", forWidthU8},
		{"u16", forWidthU16},
		{"u32", forWidthU32},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags := headerTypeUint32Flag | uint32(tt.forWidth<<forWidthShift)
			header := encodeHeader(128, 8, 0, flags)
			_, _, _, _, gotWidth, _, _, _ := decodeHeader(header)
			assert.Equal(t, tt.forWidth, gotWidth)
		})
	}
}

func TestHeaderFORWidth_NoFOR(t *testing.T) {
	header := encodeHeader(128, 8, 0, headerTypeUint32Flag)
	_, _, _, _, forWidth, _, _, _ := decodeHeader(header)
	assert.Equal(t, forWidthNone, forWidth)
}

func TestPayloadOffset_FORWidths(t *testing.T) {
	tests := []struct {
		name          string
		forBaseBytes  int
		hasExceptions bool
		want          int
	}{
		{"no_for_no_exc", 0, false, 4},
		{"u8_for_no_exc", 1, false, 5},
		{"u16_for_no_exc", 2, false, 6},
		{"u32_for_no_exc", 4, false, 8},
		{"no_for_exc", 0, true, 6},
		{"u8_for_exc", 1, true, 7},
		{"u16_for_exc", 2, true, 8},
		{"u32_for_exc", 4, true, 10},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := payloadOffset(tt.forBaseBytes, tt.hasExceptions)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestTotalBlockCost(t *testing.T) {
	tests := []struct {
		name         string
		bitWidth     int
		excCount     int
		forBaseBytes int
		want         int
	}{
		{"bw8_no_exc_no_for", 8, 0, 0, 4 + 128},
		{"bw8_no_exc_u8for", 8, 0, 1, 4 + 1 + 128},
		{"bw8_no_exc_u16for", 8, 0, 2, 4 + 2 + 128},
		{"bw8_no_exc_u32for", 8, 0, 4, 4 + 4 + 128},
		{"bw8_5exc_no_for", 8, 5, 0, 4 + 2 + 128 + 5 + 5*2},
		{"bw8_5exc_u32for", 8, 5, 4, 4 + 2 + 4 + 128 + 5 + 5*2},
		{"bw8_20exc_bitmap", 8, 20, 0, 4 + 2 + 128 + 16 + 20*2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := totalBlockCost(tt.bitWidth, tt.excCount, tt.forBaseBytes)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestFindMinMaxScalar(t *testing.T) {
	tests := []struct {
		name    string
		values  []uint32
		wantMin uint32
		wantMax uint32
	}{
		{"ascending", []uint32{1, 2, 3, 4, 5}, 1, 5},
		{"descending", []uint32{100, 50, 25, 10}, 10, 100},
		{"all_same", []uint32{42, 42, 42, 42}, 42, 42},
		{"single", []uint32{99}, 99, 99},
		{"with_zero", []uint32{0, 100, 50}, 0, 100},
		{"max_uint32", []uint32{0, 0xFFFFFFFF}, 0, 0xFFFFFFFF},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMin, gotMax := findMinMaxScalar(tt.values)
			assert.Equal(t, tt.wantMin, gotMin)
			assert.Equal(t, tt.wantMax, gotMax)
		})
	}
}

func TestSelectBitWidthWithFOR_ClusteredData(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	width, excCount, useFOR, baseValue, forWidth := selectBitWidthWithFOR(values)
	assert.True(t, useFOR, "FOR should be selected for clustered data")
	assert.Equal(t, uint32(1000000), baseValue)
	assert.Less(t, width, 20, "bit width should be small after FOR-subtract")
	assert.Equal(t, forWidthU32, forWidth, "base >= 65536 -> uint32 FOR")
	_ = excCount
}

func TestSelectBitWidthWithFOR_ClusteredSmallBase(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 200 + uint32(i%10)
	}
	_, _, useFOR, baseValue, forWidth := selectBitWidthWithFOR(values)
	if useFOR {
		assert.Equal(t, uint32(200), baseValue)
		assert.Equal(t, forWidthU8, forWidth, "base < 256 -> uint8 FOR")
	}
}

func TestSelectBitWidthWithFOR_ClusteredMediumBase(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 50000 + uint32(i%10)
	}
	_, _, useFOR, baseValue, forWidth := selectBitWidthWithFOR(values)
	if useFOR {
		assert.Equal(t, uint32(50000), baseValue)
		assert.Equal(t, forWidthU16, forWidth, "256 <= base < 65536 -> uint16 FOR")
	}
}

func TestSelectBitWidthWithFOR_MinIsZero(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	_, _, useFOR, baseValue, _ := selectBitWidthWithFOR(values)
	assert.False(t, useFOR, "FOR should not be used when min=0")
	assert.Equal(t, uint32(0), baseValue)
}

func TestSelectBitWidthWithFOR_NotBeneficial(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 100000)
	}
	_, _, useFOR, _, _ := selectBitWidthWithFOR(values)
	assert.False(t, useFOR,
		"FOR should not be used when range is wide")
}

func TestSelectBitWidthWithFOR_AllSameValue(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 999999
	}
	width, excCount, useFOR, baseValue, forWidth := selectBitWidthWithFOR(values)
	assert.True(t, useFOR, "FOR should be selected for constant data with min>0")
	assert.Equal(t, uint32(999999), baseValue)
	assert.Equal(t, 0, width, "bit width should be 0 after FOR-subtract")
	assert.Equal(t, 0, excCount)
	assert.Equal(t, forWidthU32, forWidth)
}

func TestPackUint32_FORRoundTrip(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i)
	}
	original := slices.Clone(values)
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_FORPlusDeltaRoundTrip(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i*10)
	}
	original := slices.Clone(values)
	packed, err := PackUint32(Delta, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_FORWithExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%10)
	}
	values[50] = 0xFFFFFFFF
	original := slices.Clone(values)
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_FORWidthIsSet(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	header := bo.Uint32(packed)
	forWidth := int((header >> forWidthShift) & forWidthMask)
	assert.NotEqual(t, forWidthNone, forWidth,
		"FOR width should be set for clustered data with high base")
	assert.Equal(t, forWidthU32, forWidth,
		"base >= 65536 should use uint32 FOR")
}

func TestPackUint32_FORNotUsedWhenNotBeneficial(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	header := bo.Uint32(packed)
	forWidth := int((header >> forWidthShift) & forWidthMask)
	assert.Equal(t, forWidthNone, forWidth,
		"FOR should not be used when min=0")
}

func TestPackUint32_FORWidthU8(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 200 + uint32(i%5)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	header := bo.Uint32(packed)
	forWidth := int((header >> forWidthShift) & forWidthMask)
	if forWidth != forWidthNone {
		assert.Equal(t, forWidthU8, forWidth,
			"base < 256 should use uint8 FOR")
	}
}

func TestPackUint32_FORWidthU16(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 50000 + uint32(i%5)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	header := bo.Uint32(packed)
	forWidth := int((header >> forWidthShift) & forWidthMask)
	if forWidth != forWidthNone {
		assert.Equal(t, forWidthU16, forWidth,
			"256 <= base < 65536 should use uint16 FOR")
	}
}

func TestPackUint32_FORCompressesClusteredData(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	forWidth := int((header >> forWidthShift) & forWidthMask)
	assert.NotEqual(t, forWidthNone, forWidth,
		"FOR width should be set")
	assert.Less(t, len(packed), 128*4,
		"FOR-compressed block should be smaller than uncompressed input")
}

func TestBlockLength_WithFOR(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%50)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	blockLen, err := BlockLength(packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), blockLen,
		"BlockLength must match actual packed length")
}

func TestBlockLength_WithFORAndExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%10)
	}
	values[50] = 0xFFFFFFFF
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	blockLen, err := BlockLength(packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), blockLen)
}

func TestGetUint32_WithFOR(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i*3)
	}
	original := slices.Clone(values)
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	for pos := range original {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, original[pos], got, "pos=%d", pos)
	}
}

func TestGetUint32_WithFORAndDelta(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i*10)
	}
	packed, err := PackUint32(Delta, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	for pos := range unpacked {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

func TestGetUint32_WithFORAndExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i%10)
	}
	values[42] = 0x80000000
	original := slices.Clone(values)
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	for pos := range original {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, original[pos], got, "pos=%d", pos)
	}
}

func TestPackUint32_FORRandomVectors(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := range 100 {
		base := rng.Uint32() >> 8
		spread := uint32(rng.Intn(256))
		values := make([]uint32, 128)
		for i := range values {
			values[i] = base + uint32(rng.Intn(int(spread)+1))
		}
		original := slices.Clone(values)

		packed, err := PackUint32(0, nil, values)
		require.NoError(t, err, "trial %d", trial)
		unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
		require.NoError(t, err, "trial %d", trial)
		assert.Equal(t, original, unpacked, "trial %d", trial)
	}
}

func TestPackUint32_FORRandomVectorsWithDelta(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for trial := range 100 {
		base := rng.Uint32() >> 8
		values := make([]uint32, 128)
		for i := range values {
			values[i] = base + uint32(i) + uint32(rng.Intn(10))
		}
		original := slices.Clone(values)

		packed, err := PackUint32(Delta, nil, values)
		require.NoError(t, err, "trial %d", trial)
		unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
		require.NoError(t, err, "trial %d", trial)
		assert.Equal(t, original, unpacked, "trial %d", trial)
	}
}

func TestPackUint32_FORAllMaxValues(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 0xFFFFFFFF
	}
	original := slices.Clone(values)
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_FORSingleException(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 500000
	}
	values[64] = 500000 + 0x10000
	original := slices.Clone(values)
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_FORSmallBlock(t *testing.T) {
	values := []uint32{1000000, 1000001, 1000002, 1000003}
	original := slices.Clone(values)
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestUnpackUint32_ZeroAllocs_WithFOR(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)

	allocs := testing.AllocsPerRun(100, func() {
		UnpackUint32(dst, scratch, packed)
	})
	assert.Equal(t, float64(0), allocs,
		"expected zero allocations with FOR")
}

func TestSelectFORWidth(t *testing.T) {
	tests := []struct {
		name   string
		minVal uint32
		want   int
	}{
		{"zero", 0, forWidthNone},
		{"one", 1, forWidthU8},
		{"255", 255, forWidthU8},
		{"256", 256, forWidthU16},
		{"65535", 65535, forWidthU16},
		{"65536", 65536, forWidthU32},
		{"max", 0xFFFFFFFF, forWidthU32},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := selectFORWidth(tt.minVal)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestFORBaseRoundTrip(t *testing.T) {
	tests := []struct {
		name          string
		base          uint32
		forWidth      int
		hasExceptions bool
	}{
		{"u8_no_exc", 42, forWidthU8, false},
		{"u8_exc", 255, forWidthU8, true},
		{"u16_no_exc", 50000, forWidthU16, false},
		{"u16_exc", 65535, forWidthU16, true},
		{"u32_no_exc", 1000000, forWidthU32, false},
		{"u32_exc", 0xFFFFFFFF, forWidthU32, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := make([]byte, 16)
			writeFORBase(buf, tt.base, tt.forWidth, tt.hasExceptions)
			got := readFORBase(buf, tt.forWidth, tt.hasExceptions)
			assert.Equal(t, tt.base, got)
		})
	}
}

func TestFindMinMax_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	for trial := range 50 {
		values := make([]uint32, 128)
		for i := range values {
			values[i] = rng.Uint32()
		}

		scalarMin, scalarMax := findMinMaxScalar(values)
		simdMin, simdMax := findMinMaxSIMD(values)

		assert.Equal(t, scalarMin, simdMin, "min mismatch, trial %d", trial)
		assert.Equal(t, scalarMax, simdMax, "max mismatch, trial %d", trial)
	}
}

func TestFindMinMax_SIMDSmallSlice(t *testing.T) {
	values := []uint32{100, 50, 200, 25}
	min, max := findMinMaxSIMD(values)
	assert.Equal(t, uint32(25), min)
	assert.Equal(t, uint32(200), max)
}

func TestFORSubtract_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(55))
	for trial := range 50 {
		base := rng.Uint32() >> 4
		values := make([]uint32, 128)
		for i := range values {
			values[i] = base + uint32(rng.Intn(1000))
		}

		scalarResult := make([]uint32, 128)
		forSubtractScalar(scalarResult, values, base)

		simdResult := make([]uint32, 128)
		forSubtractSIMD(simdResult, values, base)

		assert.Equal(t, scalarResult, simdResult, "trial %d", trial)
	}
}

func TestFORAdd_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(66))
	for trial := range 50 {
		base := rng.Uint32() >> 4
		values := make([]uint32, 128)
		for i := range values {
			values[i] = uint32(rng.Intn(1000))
		}

		scalarResult := make([]uint32, 128)
		copy(scalarResult, values)
		forAddScalar(scalarResult, 128, base)

		simdResult := make([]uint32, 128)
		copy(simdResult, values)
		forAddSIMD(simdResult, 128, base)

		assert.Equal(t, scalarResult, simdResult, "trial %d", trial)
	}
}

func TestForSubtractAddScalar(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 1000000 + uint32(i)
	}
	subtracted := make([]uint32, 128)
	forSubtractScalar(subtracted, values, 1000000)
	for i := range subtracted {
		assert.Equal(t, uint32(i), subtracted[i])
	}
	forAddScalar(subtracted, 128, 1000000)
	assert.Equal(t, values, subtracted)
}
