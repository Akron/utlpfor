package utlpfor

import (
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMaxBlockLength64_GeneralCase(t *testing.T) {
	// Worst case: two-block, exceptions possible.
	// Block 1: header(4) + svbLen(2) + forBase(4) + block2Len(2) + payload(28*16=448) + excIdx(16) + svbData(544) = 1020
	// Block 2: header(4) + svbLen(2) + forBase(4) + payload(28*16=448) + excIdx(16) + svbData(544) = 1018
	// Total: 2038
	result := MaxBlockLength64(0)
	assert.Equal(t, 2038, result)
}

func TestMaxBlockLength64_NoPatch(t *testing.T) {
	// Two-block, no exceptions.
	// Block 1: header(4) + forBase(4) + block2Len(2) + payload(32*16=512) = 522
	// Block 2: header(4) + forBase(4) + payload(32*16=512) = 520
	// Total: 1042
	result := MaxBlockLength64(NoPatch)
	assert.Equal(t, 1042, result)
}

func TestMaxBlockLength64_NoPatchNoFOR(t *testing.T) {
	// Two-block, no exceptions, no FOR base.
	// Block 1: header(4) + block2Len(2) + payload(32*16=512) = 518
	// Block 2: header(4) + payload(32*16=512) = 516
	// Total: 1034
	result := MaxBlockLength64(NoPatch | NoFOR)
	assert.Equal(t, 1034, result)
}

func TestMaxBlockLength64_NoFOR(t *testing.T) {
	// Two-block, exceptions possible, no FOR base.
	// Block 1: header(4) + svbLen(2) + block2Len(2) + payload(28*16=448) + excIdx(16) + svbData(544) = 1016
	// Block 2: header(4) + svbLen(2) + payload(28*16=448) + excIdx(16) + svbData(544) = 1014
	// Total: 2030
	result := MaxBlockLength64(NoFOR)
	assert.Equal(t, 2030, result)
}

func TestMaxBlockLength64_GreaterOrEqualMaxBlockLength32(t *testing.T) {
	for _, flag := range []Flag{0, NoPatch, NoFOR, NoPatch | NoFOR} {
		assert.GreaterOrEqual(t, MaxBlockLength64(flag), MaxBlockLength32(flag),
			"MaxBlockLength64 should be >= MaxBlockLength32 for flag %d", flag)
	}
}

func TestMaxBlockLength64_IgnoresNonSizeFlags(t *testing.T) {
	base := MaxBlockLength64(0)
	assert.Equal(t, base, MaxBlockLength64(Delta))
	assert.Equal(t, base, MaxBlockLength64(Special))
	assert.Equal(t, base, MaxBlockLength64(Append))
}

func packUnpackUint64RoundTrip(t *testing.T, flag Flag, values []uint64) {
	t.Helper()
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(flag, values, nil, scratch)
	require.NoError(t, err)
	require.NotNil(t, packed)

	dst := make([]uint64, blockSize)
	unpacked, consumed, err := UnpackUint64(dst, scratch, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed, "consumed bytes should match packed length")
	require.Equal(t, len(values), len(unpacked))
	assert.Equal(t, values, unpacked)
}

func TestPackUnpackUint64_AllZero(t *testing.T) {
	values := make([]uint64, blockSize)
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_AllFitIn32Bits(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i * 100)
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_SequentialFromZero(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i)
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_Crossing32BitBoundary(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFC0 + uint64(i)
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_AboveBoundaryUniformUpper(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)*100
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_TimestampLike(t *testing.T) {
	values := make([]uint64, blockSize)
	baseTimestamp := uint64(1_719_300_000_000)
	for i := range values {
		values[i] = baseTimestamp + uint64(i)
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_RandomValues(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = rng.Uint64()
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_SingleValue(t *testing.T) {
	values := []uint64{0x1234567890ABCDEF}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_PartialBlock(t *testing.T) {
	for _, count := range []int{1, 2, 15, 16, 17, 63, 64, 65, 100, 127} {
		t.Run("", func(t *testing.T) {
			values := make([]uint64, count)
			for i := range values {
				values[i] = 0x200000000 + uint64(i)*7
			}
			packUnpackUint64RoundTrip(t, 0, values)
		})
	}
}

func TestPackUnpackUint64_FullBlock(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i) * 1000
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_WithDelta(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i) * 100
	}
	packUnpackUint64RoundTrip(t, Delta, values)
}

func TestPackUnpackUint64_WithDelta_BoundaryCrossing(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFC0 + uint64(i)
	}
	packUnpackUint64RoundTrip(t, Delta, values)
}

func TestPackUnpackUint64_WithNoPatch(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}
	packUnpackUint64RoundTrip(t, NoPatch, values)
}

func TestPackUnpackUint64_WithNoFOR(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)*100
	}
	packUnpackUint64RoundTrip(t, NoFOR, values)
}

func TestPackUnpackUint64_WithNoPatchNoFOR(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}
	packUnpackUint64RoundTrip(t, NoPatch|NoFOR, values)
}

func TestPackUint64_EmptyInput(t *testing.T) {
	scratch := make([]uint32, ScratchLen64)
	_, err := PackUint64(0, nil, nil, scratch)
	assert.Error(t, err)
}

func TestPackUint64_EmptySlice(t *testing.T) {
	scratch := make([]uint32, ScratchLen64)
	_, err := PackUint64(0, []uint64{}, nil, scratch)
	assert.Error(t, err)
}

func TestPackUint64_TooManyValues(t *testing.T) {
	scratch := make([]uint32, ScratchLen64)
	values := make([]uint64, blockSize+1)
	_, err := PackUint64(0, values, nil, scratch)
	assert.Error(t, err)
}

func TestPackUnpackUint64_MaxUint64Values(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = math.MaxUint64
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_MixedMagnitudes(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		if i%4 == 0 {
			values[i] = uint64(i)
		} else if i%4 == 1 {
			values[i] = 0xFFFFFFFF
		} else if i%4 == 2 {
			values[i] = 0x100000000
		} else {
			values[i] = math.MaxUint64 - uint64(i)
		}
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestGetUint64_SingleBlockAllFitIn32(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i * 100)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	for i, want := range values {
		got, err := GetUint64(i, packed, scratch)
		require.NoError(t, err, "pos=%d", i)
		assert.Equal(t, want, got, "pos=%d", i)
	}
}

func TestGetUint64_TwoBlockMode(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)*100
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	for i, want := range values {
		got, err := GetUint64(i, packed, scratch)
		require.NoError(t, err, "pos=%d", i)
		assert.Equal(t, want, got, "pos=%d", i)
	}
}

func TestGetUint64_VerifyAgainstUnpack(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = rng.Uint64()
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, scratch, packed)
	require.NoError(t, err)

	for i, want := range unpacked {
		got, err := GetUint64(i, packed, scratch)
		require.NoError(t, err, "pos=%d", i)
		assert.Equal(t, want, got, "pos=%d: GetUint64 != UnpackUint64", i)
	}
}

func TestGetUint64_FirstAndLastPosition(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFC0 + uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	got0, err := GetUint64(0, packed, scratch)
	require.NoError(t, err)
	assert.Equal(t, values[0], got0)

	gotLast, err := GetUint64(len(values)-1, packed, scratch)
	require.NoError(t, err)
	assert.Equal(t, values[len(values)-1], gotLast)
}

func TestGetUint64_OutOfRange(t *testing.T) {
	values := make([]uint64, 10)
	for i := range values {
		values[i] = uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	_, err = GetUint64(10, packed, scratch)
	assert.Error(t, err)

	_, err = GetUint64(-1, packed, scratch)
	assert.Error(t, err)
}

func TestGetUint64_WithDelta(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)*7
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(Delta, values, nil, scratch)
	require.NoError(t, err)

	for i, want := range values {
		got, err := GetUint64(i, packed, scratch)
		require.NoError(t, err, "pos=%d", i)
		assert.Equal(t, want, got, "pos=%d", i)
	}
}

func TestGetUint64_OnUint32BlockReturnsError(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint32(0, values, nil, scratch)
	require.NoError(t, err)

	_, err = GetUint64(0, packed, scratch)
	assert.Error(t, err)
	assert.True(t, errors.Is(err, ErrUnsupportedType), "expected ErrUnsupportedType, got %v", err)
}

func TestBlockLength_Uint64SingleBlock(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	bl, err := BlockLength(packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), bl)
}

func TestBlockLength_Uint64TwoBlock(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	bl, err := BlockLength(packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), bl)
}

func TestBlockLength_Uint64MatchesConsumedBytes(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	scratch := make([]uint32, ScratchLen64)

	for trial := range 100 {
		count := rng.Intn(blockSize) + 1
		values := make([]uint64, count)
		for i := range values {
			values[i] = rng.Uint64()
		}
		packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
		require.NoError(t, err, "trial %d", trial)

		bl, err := BlockLength(packed)
		require.NoError(t, err, "trial %d", trial)

		dst := make([]uint64, blockSize)
		_, consumed, err := UnpackUint64(dst, scratch, packed)
		require.NoError(t, err, "trial %d", trial)
		assert.Equal(t, consumed, bl, "trial %d: BlockLength != consumed", trial)
	}
}

func TestUnpackUint32_OnUint64BlockReturnsError(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	dst := make([]uint32, blockSize)
	_, _, err = UnpackUint32(packed, dst, scratch)
	assert.Error(t, err)
	assert.True(t, errors.Is(err, ErrUnsupportedType), "expected ErrUnsupportedType, got %v", err)
}

func TestGetUint32_OnUint64BlockReturnsError(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	_, err = GetUint32(0, packed, scratch)
	assert.Error(t, err)
	assert.True(t, errors.Is(err, ErrUnsupportedType), "expected ErrUnsupportedType, got %v", err)
}

func TestPackUint64_NilScratchWorks(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i)
	}
	packed, err := PackUint64(0, values, nil, nil)
	require.NoError(t, err)
	require.NotNil(t, packed)

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, nil, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestUnpackUint64_NilScratchWorks(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, nil, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUint64_UndersizedScratchWorks(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i)
	}
	smallScratch := make([]uint32, 10)
	packed, err := PackUint64(0, values, nil, smallScratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, nil, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUint64_PreAllocatedScratch(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i) * 100
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, scratch, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestScratchLen64_Value(t *testing.T) {
	assert.Equal(t, 2*blockSize, ScratchLen64)
	assert.Equal(t, 256, ScratchLen64)
}

func TestPackUnpackUint64_BoundaryValue_0xFFFFFFFF(t *testing.T) {
	values := []uint64{0xFFFFFFFF}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_BoundaryValue_0x100000000(t *testing.T) {
	values := []uint64{0x100000000}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_BoundaryValue_0x100000001(t *testing.T) {
	values := []uint64{0x100000001}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_DeltaTimestamps(t *testing.T) {
	values := make([]uint64, blockSize)
	baseTimestamp := uint64(1_719_300_000_000)
	for i := range values {
		values[i] = baseTimestamp + uint64(i)
	}
	packUnpackUint64RoundTrip(t, Delta, values)
}

func TestPackUnpackUint64_AllSameAbove32Bits(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xDEADBEEF12345678
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_WithAppend(t *testing.T) {
	values1 := make([]uint64, blockSize)
	values2 := make([]uint64, blockSize)
	for i := range values1 {
		values1[i] = uint64(i)
		values2[i] = 0x100000000 + uint64(i)
	}

	scratch := make([]uint32, ScratchLen64)
	maxLen := MaxBlockLength64(0)
	if maxLen <= 0 {
		maxLen = 4096
	}
	dst := make([]byte, 0, 2*maxLen)
	var err error
	dst, err = PackUint64(Append, values1, dst, scratch)
	require.NoError(t, err)
	block1Len := len(dst)

	dst, err = PackUint64(Append, values2, dst, scratch)
	require.NoError(t, err)

	// Unpack first block
	out := make([]uint64, blockSize)
	unpacked1, consumed1, err := UnpackUint64(out, scratch, dst[:block1Len])
	require.NoError(t, err)
	assert.Equal(t, block1Len, consumed1)
	assert.Equal(t, values1, unpacked1)

	// Unpack second block
	unpacked2, _, err := UnpackUint64(out, scratch, dst[block1Len:])
	require.NoError(t, err)
	assert.Equal(t, values2, unpacked2)
}

func TestSplitUint64_Identity(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xDEADBEEF00000000 | uint64(i*100)
	}
	lower := make([]uint32, blockSize)
	upper := make([]uint32, blockSize)
	splitUint64(values, lower, upper)

	for i, v := range values {
		assert.Equal(t, uint32(v), lower[i], "lower[%d]", i)
		assert.Equal(t, uint32(v>>32), upper[i], "upper[%d]", i)
	}
}

func TestSplitCombine_RoundTrip(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFEDCBA9876543210 - uint64(i)*0x0101010101010101
	}
	lower := make([]uint32, blockSize)
	upper := make([]uint32, blockSize)
	splitUint64(values, lower, upper)

	result := make([]uint64, blockSize)
	combineUint64(result, lower, upper, blockSize)
	assert.Equal(t, values, result)
}

func TestSplitCombine_ZeroValues(t *testing.T) {
	values := make([]uint64, blockSize)
	lower := make([]uint32, blockSize)
	upper := make([]uint32, blockSize)
	splitUint64(values, lower, upper)

	for i := range blockSize {
		assert.Equal(t, uint32(0), lower[i])
		assert.Equal(t, uint32(0), upper[i])
	}

	result := make([]uint64, blockSize)
	combineUint64(result, lower, upper, blockSize)
	assert.Equal(t, values, result)
}

func TestSplitCombine_MaxUint64(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = math.MaxUint64
	}
	lower := make([]uint32, blockSize)
	upper := make([]uint32, blockSize)
	splitUint64(values, lower, upper)

	for i := range blockSize {
		assert.Equal(t, uint32(0xFFFFFFFF), lower[i])
		assert.Equal(t, uint32(0xFFFFFFFF), upper[i])
	}

	result := make([]uint64, blockSize)
	combineUint64(result, lower, upper, blockSize)
	assert.Equal(t, values, result)
}

func TestSplitCombine_BoundaryValues(t *testing.T) {
	values := []uint64{0xFFFFFFFF, 0x100000000, 0x100000001, 0}
	lower := make([]uint32, len(values))
	upper := make([]uint32, len(values))
	splitUint64(values, lower, upper)

	assert.Equal(t, uint32(0xFFFFFFFF), lower[0])
	assert.Equal(t, uint32(0), upper[0])

	assert.Equal(t, uint32(0), lower[1])
	assert.Equal(t, uint32(1), upper[1])

	assert.Equal(t, uint32(1), lower[2])
	assert.Equal(t, uint32(1), upper[2])

	assert.Equal(t, uint32(0), lower[3])
	assert.Equal(t, uint32(0), upper[3])

	result := make([]uint64, len(values))
	combineUint64(result, lower, upper, len(values))
	assert.Equal(t, values, result)
}

func TestSplitCombine_PartialCount(t *testing.T) {
	values := []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210, 0x1111111122222222}
	lower := make([]uint32, len(values))
	upper := make([]uint32, len(values))
	splitUint64(values, lower, upper)

	result := make([]uint64, len(values))
	combineUint64(result, lower, upper, 2)
	assert.Equal(t, values[0], result[0])
	assert.Equal(t, values[1], result[1])
	assert.Equal(t, uint64(0), result[2])
}

func TestAllFitIn32Bits_TrueForSmallValues(t *testing.T) {
	values := []uint64{0, 1, 0xFFFFFFFF, 42, 1000000}
	assert.True(t, allFitIn32Bits(values))
}

func TestAllFitIn32Bits_TrueForZero(t *testing.T) {
	values := make([]uint64, blockSize)
	assert.True(t, allFitIn32Bits(values))
}

func TestAllFitIn32Bits_TrueForMaxUint32(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFFF
	}
	assert.True(t, allFitIn32Bits(values))
}

func TestAllFitIn32Bits_FalseForLargeValue(t *testing.T) {
	values := []uint64{0, 1, 2, 0x100000000}
	assert.False(t, allFitIn32Bits(values))
}

func TestAllFitIn32Bits_FalseForMaxUint64(t *testing.T) {
	values := []uint64{math.MaxUint64}
	assert.False(t, allFitIn32Bits(values))
}

func TestAllFitIn32Bits_FalseForSingleLargeInBlock(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i)
	}
	values[blockSize-1] = 0x100000000
	assert.False(t, allFitIn32Bits(values))
}

func TestNarrowToUint32_Basic(t *testing.T) {
	values := []uint64{0, 1, 255, 0xFFFFFFFF, 1000}
	dst := make([]uint32, len(values))
	narrowToUint32(dst, values, len(values))

	assert.Equal(t, uint32(0), dst[0])
	assert.Equal(t, uint32(1), dst[1])
	assert.Equal(t, uint32(255), dst[2])
	assert.Equal(t, uint32(0xFFFFFFFF), dst[3])
	assert.Equal(t, uint32(1000), dst[4])
}

func TestNarrowToUint32_Truncation(t *testing.T) {
	values := []uint64{0x100000001, 0xFFFFFFFF00000002}
	dst := make([]uint32, len(values))
	narrowToUint32(dst, values, len(values))

	assert.Equal(t, uint32(1), dst[0])
	assert.Equal(t, uint32(2), dst[1])
}

func TestNarrowToUint32_PartialCount(t *testing.T) {
	values := []uint64{100, 200, 300, 400, 500}
	dst := make([]uint32, len(values))
	narrowToUint32(dst, values, 3)

	assert.Equal(t, uint32(100), dst[0])
	assert.Equal(t, uint32(200), dst[1])
	assert.Equal(t, uint32(300), dst[2])
	assert.Equal(t, uint32(0), dst[3])
	assert.Equal(t, uint32(0), dst[4])
}

func TestNarrowToUint32_FullBlock(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i * 77)
	}
	dst := make([]uint32, blockSize)
	narrowToUint32(dst, values, blockSize)

	for i := range blockSize {
		assert.Equal(t, uint32(values[i]), dst[i], "pos %d", i)
	}
}
