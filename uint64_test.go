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
	assert.Equal(t, 3*blockSize, ScratchLen64)
	assert.Equal(t, 384, ScratchLen64)
}

func TestEnsureCapacity64(t *testing.T) {
	maxLen := MaxBlockLength64(0)

	t.Run("nil_dst", func(t *testing.T) {
		dst := ensureCapacity64(nil, 0, 0)
		assert.GreaterOrEqual(t, cap(dst), maxLen)
		assert.Equal(t, 0, len(dst))
	})

	t.Run("sufficient_cap_no_alloc", func(t *testing.T) {
		orig := make([]byte, 10, 10+maxLen)
		for i := range orig {
			orig[i] = byte(i)
		}
		dst := ensureCapacity64(orig, 10, 0)
		assert.Equal(t, orig[:10], dst[:10])
		assert.GreaterOrEqual(t, cap(dst)-10, maxLen)
	})

	t.Run("insufficient_cap_grows", func(t *testing.T) {
		orig := make([]byte, 5, 10)
		for i := range orig {
			orig[i] = byte(i + 1)
		}
		dst := ensureCapacity64(orig, 5, 0)
		assert.GreaterOrEqual(t, cap(dst)-5, maxLen)
		assert.Equal(t, 5, len(dst))
		assert.Equal(t, []byte{1, 2, 3, 4, 5}, dst[:5])
	})

	t.Run("preserves_prefix_on_grow", func(t *testing.T) {
		prefix := []byte("hello world!")
		orig := make([]byte, len(prefix))
		copy(orig, prefix)
		dst := ensureCapacity64(orig, len(prefix), 0)
		assert.GreaterOrEqual(t, cap(dst)-len(prefix), maxLen)
		assert.Equal(t, len(prefix), len(dst))
		assert.Equal(t, prefix, dst[:len(prefix)])
	})

	t.Run("flag_affects_needed_cap", func(t *testing.T) {
		noPatchLen := MaxBlockLength64(NoPatch)
		dst := ensureCapacity64(nil, 0, NoPatch)
		assert.GreaterOrEqual(t, cap(dst), noPatchLen)
	})
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

func TestPackUnpackUint64_FOR64_ClusteredAbove32Bits(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_FOR64_MinAbove32BitsSmallRange(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(0x200000000)
	for i := range values {
		values[i] = base + uint64(i)*7
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_FOR64_MinBelow32BitsCrossBoundary(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFFFFFF00 + uint64(i)*2
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_FOR64_FallsThruToTwoBlock(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = rng.Uint64()
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_FOR64_WithDelta(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(0x100000000)
	for i := range values {
		values[i] = base + uint64(i)*100
	}
	packUnpackUint64RoundTrip(t, Delta, values)
}

func TestPackUnpackUint64_FOR64_WithNoPatch(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(0x100000000)
	for i := range values {
		values[i] = base + uint64(i)
	}
	packUnpackUint64RoundTrip(t, NoPatch, values)
}

func TestPackUnpackUint64_FOR64_WithNoFOR(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(0x100000000)
	for i := range values {
		values[i] = base + uint64(i)
	}
	packUnpackUint64RoundTrip(t, NoFOR, values)
}

func TestPackUnpackUint64_FOR64_PartialBlock(t *testing.T) {
	for _, count := range []int{1, 2, 15, 16, 17, 63, 64, 65, 100, 127} {
		t.Run("", func(t *testing.T) {
			values := make([]uint64, count)
			for i := range values {
				values[i] = 0x100000000 + uint64(i)*3
			}
			packUnpackUint64RoundTrip(t, 0, values)
		})
	}
}

func TestBlockLength_Uint64_FOR64SingleBlock(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(0x100000000)
	for i := range values {
		values[i] = base + uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	bl, err := BlockLength(packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), bl)
}

func TestBlockLength_Uint64_FOR64MatchesConsumed(t *testing.T) {
	scratch := make([]uint32, ScratchLen64)

	testCases := []struct {
		name string
		gen  func() []uint64
	}{
		{"clustered", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0x200000000 + uint64(i)*11
			}
			return v
		}},
		{"boundary_crossing", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0xFFFFFFC0 + uint64(i)
			}
			return v
		}},
		{"timestamps", func() []uint64 {
			v := make([]uint64, blockSize)
			base := uint64(1_719_300_000_000)
			for i := range v {
				v[i] = base + uint64(i)
			}
			return v
		}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			values := tc.gen()
			packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
			require.NoError(t, err)

			bl, err := BlockLength(packed)
			require.NoError(t, err)

			dst := make([]uint64, blockSize)
			_, consumed, err := UnpackUint64(dst, scratch, packed)
			require.NoError(t, err)
			assert.Equal(t, consumed, bl, "BlockLength != consumed")
		})
	}
}

func TestGetUint64_FOR64SingleBlock(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(0x100000000)
	for i := range values {
		values[i] = base + uint64(i)*100
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

func TestGetUint64_FOR64WithDelta(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(0x100000000)
	for i := range values {
		values[i] = base + uint64(i)*7
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

func TestGetUint64_FOR64BoundaryCrossing(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFC0 + uint64(i)
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

func TestFOR64_CompressionImprovement_BoundaryCrossing(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFC0 + uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)

	packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	uncompressedSize := blockSize * 8
	assert.Less(t, len(packed), uncompressedSize,
		"FOR64 block should be smaller than uncompressed (%d vs %d)", len(packed), uncompressedSize)
}

func TestFOR64_CompressionImprovement_Timestamps(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(1_719_300_000_000)
	for i := range values {
		values[i] = base + uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)

	packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	uncompressedSize := blockSize * 8
	assert.Less(t, len(packed), uncompressedSize,
		"FOR64 timestamp block should compress well (%d vs %d)", len(packed), uncompressedSize)
}

func TestForSubtract64(t *testing.T) {
	t.Run("basic", func(t *testing.T) {
		values := []uint64{100, 105, 110, 115}
		dst := make([]uint32, len(values))
		forSubtract64(dst, values, 100)
		assert.Equal(t, []uint32{0, 5, 10, 15}, dst)
	})

	t.Run("zero_base", func(t *testing.T) {
		values := []uint64{0, 1, 2, 3}
		dst := make([]uint32, len(values))
		forSubtract64(dst, values, 0)
		assert.Equal(t, []uint32{0, 1, 2, 3}, dst)
	})

	t.Run("large_base_above_32bit", func(t *testing.T) {
		base := uint64(0x200000000)
		values := []uint64{base, base + 10, base + 255}
		dst := make([]uint32, len(values))
		forSubtract64(dst, values, base)
		assert.Equal(t, []uint32{0, 10, 255}, dst)
	})

	t.Run("boundary_crossing_base", func(t *testing.T) {
		base := uint64(0xFFFFFFC0)
		values := make([]uint64, blockSize)
		for i := range values {
			values[i] = base + uint64(i)
		}
		dst := make([]uint32, blockSize)
		forSubtract64(dst, values, base)
		for i := range blockSize {
			assert.Equal(t, uint32(i), dst[i], "pos %d", i)
		}
	})

	t.Run("single_value", func(t *testing.T) {
		values := []uint64{0x100000042}
		dst := make([]uint32, 1)
		forSubtract64(dst, values, 0x100000042)
		assert.Equal(t, uint32(0), dst[0])
	})

	t.Run("full_block_timestamps", func(t *testing.T) {
		base := uint64(1_719_300_000_000)
		values := make([]uint64, blockSize)
		for i := range values {
			values[i] = base + uint64(i)*7
		}
		dst := make([]uint32, blockSize)
		forSubtract64(dst, values, base)
		for i := range blockSize {
			assert.Equal(t, uint32(i*7), dst[i], "pos %d", i)
		}
	})
}

func TestForAdd64(t *testing.T) {
	t.Run("basic", func(t *testing.T) {
		values := []uint32{0, 5, 10, 15}
		dst := make([]uint64, len(values))
		forAdd64(dst, values, 100, len(values))
		assert.Equal(t, []uint64{100, 105, 110, 115}, dst)
	})

	t.Run("zero_base", func(t *testing.T) {
		values := []uint32{0, 1, 2, 3}
		dst := make([]uint64, len(values))
		forAdd64(dst, values, 0, len(values))
		assert.Equal(t, []uint64{0, 1, 2, 3}, dst)
	})

	t.Run("large_base_above_32bit", func(t *testing.T) {
		base := uint64(0x200000000)
		values := []uint32{0, 10, 255}
		dst := make([]uint64, len(values))
		forAdd64(dst, values, base, len(values))
		assert.Equal(t, []uint64{base, base + 10, base + 255}, dst)
	})

	t.Run("max_uint32_values", func(t *testing.T) {
		base := uint64(0x100000000)
		values := []uint32{0xFFFFFFFF, 0}
		dst := make([]uint64, len(values))
		forAdd64(dst, values, base, len(values))
		assert.Equal(t, []uint64{0x1FFFFFFFF, 0x100000000}, dst)
	})

	t.Run("partial_count", func(t *testing.T) {
		values := []uint32{10, 20, 30, 40}
		dst := make([]uint64, len(values))
		forAdd64(dst, values, 1000, 2)
		assert.Equal(t, uint64(1010), dst[0])
		assert.Equal(t, uint64(1020), dst[1])
		assert.Equal(t, uint64(0), dst[2])
		assert.Equal(t, uint64(0), dst[3])
	})
}

func TestForSubtractAdd64_RoundTrip(t *testing.T) {
	t.Run("basic", func(t *testing.T) {
		original := []uint64{0x100000000, 0x100000005, 0x10000000A, 0x10000000F}
		min64 := uint64(0x100000000)

		subtracted := make([]uint32, len(original))
		forSubtract64(subtracted, original, min64)

		restored := make([]uint64, len(original))
		forAdd64(restored, subtracted, min64, len(original))
		assert.Equal(t, original, restored)
	})

	t.Run("boundary_crossing", func(t *testing.T) {
		original := make([]uint64, blockSize)
		for i := range original {
			original[i] = 0xFFFFFFC0 + uint64(i)
		}
		min64 := uint64(0xFFFFFFC0)

		subtracted := make([]uint32, blockSize)
		forSubtract64(subtracted, original, min64)

		restored := make([]uint64, blockSize)
		forAdd64(restored, subtracted, min64, blockSize)
		assert.Equal(t, original, restored)
	})

	t.Run("timestamps", func(t *testing.T) {
		base := uint64(1_719_300_000_000)
		original := make([]uint64, blockSize)
		for i := range original {
			original[i] = base + uint64(i)*100
		}

		subtracted := make([]uint32, blockSize)
		forSubtract64(subtracted, original, base)

		restored := make([]uint64, blockSize)
		forAdd64(restored, subtracted, base, blockSize)
		assert.Equal(t, original, restored)
	})

	t.Run("single_value", func(t *testing.T) {
		original := []uint64{0xDEADBEEFCAFEBABE}
		subtracted := make([]uint32, 1)
		forSubtract64(subtracted, original, original[0])
		assert.Equal(t, uint32(0), subtracted[0])

		restored := make([]uint64, 1)
		forAdd64(restored, subtracted, original[0], 1)
		assert.Equal(t, original, restored)
	})
}

func TestIsFor64SingleBlock(t *testing.T) {
	t.Run("true_uint64_no_combine_forwidth3", func(t *testing.T) {
		assert.True(t, isFor64SingleBlock(IntTypeUint64, forWidthU32, false))
	})

	t.Run("false_uint32", func(t *testing.T) {
		assert.False(t, isFor64SingleBlock(IntTypeUint32, forWidthU32, false))
	})

	t.Run("false_uint16", func(t *testing.T) {
		assert.False(t, isFor64SingleBlock(IntTypeUint16, forWidthU32, false))
	})

	t.Run("false_uint64_with_combine", func(t *testing.T) {
		assert.False(t, isFor64SingleBlock(IntTypeUint64, forWidthU32, true))
	})

	t.Run("false_uint64_forwidth0", func(t *testing.T) {
		assert.False(t, isFor64SingleBlock(IntTypeUint64, forWidthNone, false))
	})

	t.Run("false_uint64_forwidth1", func(t *testing.T) {
		assert.False(t, isFor64SingleBlock(IntTypeUint64, forWidthU8, false))
	})

	t.Run("false_uint64_forwidth2", func(t *testing.T) {
		assert.False(t, isFor64SingleBlock(IntTypeUint64, forWidthU16, false))
	})

	t.Run("false_uint64_combine_forwidth1", func(t *testing.T) {
		assert.False(t, isFor64SingleBlock(IntTypeUint64, forWidthU8, true))
	})
}

func TestForBaseBytesForBlock(t *testing.T) {
	t.Run("uint32_forwidth0", func(t *testing.T) {
		assert.Equal(t, 0, forBaseBytesForBlock(forWidthNone, IntTypeUint32, false))
	})

	t.Run("uint32_forwidth1", func(t *testing.T) {
		assert.Equal(t, 1, forBaseBytesForBlock(forWidthU8, IntTypeUint32, false))
	})

	t.Run("uint32_forwidth2", func(t *testing.T) {
		assert.Equal(t, 2, forBaseBytesForBlock(forWidthU16, IntTypeUint32, false))
	})

	t.Run("uint32_forwidth3", func(t *testing.T) {
		assert.Equal(t, 4, forBaseBytesForBlock(forWidthU32, IntTypeUint32, false))
	})

	t.Run("uint64_single_forwidth3_is_8bytes", func(t *testing.T) {
		assert.Equal(t, 8, forBaseBytesForBlock(forWidthU32, IntTypeUint64, false))
	})

	t.Run("uint64_combine_forwidth3_is_4bytes", func(t *testing.T) {
		assert.Equal(t, 4, forBaseBytesForBlock(forWidthU32, IntTypeUint64, true))
	})

	t.Run("uint64_single_forwidth0", func(t *testing.T) {
		assert.Equal(t, 0, forBaseBytesForBlock(forWidthNone, IntTypeUint64, false))
	})

	t.Run("uint64_single_forwidth1", func(t *testing.T) {
		assert.Equal(t, 1, forBaseBytesForBlock(forWidthU8, IntTypeUint64, false))
	})

	t.Run("uint64_single_forwidth2", func(t *testing.T) {
		assert.Equal(t, 2, forBaseBytesForBlock(forWidthU16, IntTypeUint64, false))
	})

	t.Run("uint64_combine_forwidth1", func(t *testing.T) {
		assert.Equal(t, 1, forBaseBytesForBlock(forWidthU8, IntTypeUint64, true))
	})

	t.Run("uint16_forwidth3", func(t *testing.T) {
		assert.Equal(t, 4, forBaseBytesForBlock(forWidthU32, IntTypeUint16, false))
	})
}

func TestInsertFor64Base(t *testing.T) {
	t.Run("header_has_forwidth3", func(t *testing.T) {
		values := make([]uint64, blockSize)
		for i := range values {
			values[i] = 0x100000000 + uint64(i)
		}
		scratch := make([]uint32, ScratchLen64)
		packed, err := PackUint64(0, values, nil, scratch)
		require.NoError(t, err)

		header := bo.Uint32(packed)
		_, _, _, _, forWidth, _, _, _, _, _ := decodeHeader(header)
		assert.Equal(t, forWidthU32, forWidth, "FOR64 block should have forWidth=3")
	})

	t.Run("base_at_correct_offset_no_exceptions", func(t *testing.T) {
		values := make([]uint64, blockSize)
		base := uint64(0x100000000)
		for i := range values {
			values[i] = base + uint64(i)
		}
		scratch := make([]uint32, ScratchLen64)
		packed, err := PackUint64(NoPatch, values, nil, scratch)
		require.NoError(t, err)

		header := bo.Uint32(packed)
		_, _, _, excCount, forWidth, _, _, _, _, _ := decodeHeader(header)
		assert.Equal(t, forWidthU32, forWidth)
		assert.Equal(t, 0, excCount)

		for64Base := bo.Uint64(packed[headerBytes:])
		assert.Equal(t, base, for64Base)
	})

	t.Run("base_at_correct_offset_with_exceptions", func(t *testing.T) {
		values := make([]uint64, blockSize)
		base := uint64(0x100000000)
		for i := range values {
			values[i] = base + uint64(i)
		}
		values[0] = base + 0xFFFFF
		scratch := make([]uint32, ScratchLen64)
		packed, err := PackUint64(0, values, nil, scratch)
		require.NoError(t, err)

		header := bo.Uint32(packed)
		_, _, _, excCount, forWidth, _, _, _, _, _ := decodeHeader(header)
		assert.Equal(t, forWidthU32, forWidth, "FOR64 block should have forWidth=3")
		assert.Greater(t, excCount, 0, "should have exceptions")

		// With exceptions, base is after header + svbLen
		expectedMin := base + 1 // min of all values (base+0xFFFFF outlier, rest start at base+1)
		for64Base := bo.Uint64(packed[headerBytes+svbLenBytes:])
		assert.Equal(t, expectedMin, for64Base)
	})

	t.Run("roundtrip_verifies_base", func(t *testing.T) {
		values := make([]uint64, blockSize)
		base := uint64(0x100000000)
		for i := range values {
			values[i] = base + uint64(i)*7
		}
		scratch := make([]uint32, ScratchLen64)
		packed, err := PackUint64(0, values, nil, scratch)
		require.NoError(t, err)

		dst := make([]uint64, blockSize)
		unpacked, _, err := UnpackUint64(dst, scratch, packed)
		require.NoError(t, err)
		assert.Equal(t, values, unpacked)
	})
}

func TestUint64_DeltaWrappingCorrectness(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFC0 + uint64(i)
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(Delta, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)
	unpacked, consumed, err := UnpackUint64(dst, scratch, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
	require.Equal(t, len(values), len(unpacked))
	assert.Equal(t, values, unpacked)

	uncompressed := blockSize * 8
	assert.Less(t, len(packed), uncompressed,
		"delta-encoded boundary-crossing block should compress (got %d, raw %d)",
		len(packed), uncompressed)

	assert.Less(t, len(packed), uncompressed/2,
		"lower-block deltas should be small; packed %d should be well under %d",
		len(packed), uncompressed/2)
}

func TestUint64_DeltaWrappingCorrectness_LargerStep(t *testing.T) {
	for _, step := range []uint64{1, 7, 16, 100, 1000} {
		t.Run("", func(t *testing.T) {
			values := make([]uint64, blockSize)
			for i := range values {
				values[i] = 0xFFFFFFC0 + uint64(i)*step
			}

			scratch := make([]uint32, ScratchLen64)
			packed, err := PackUint64(Delta, slices.Clone(values), nil, scratch)
			require.NoError(t, err)

			dst := make([]uint64, blockSize)
			unpacked, _, err := UnpackUint64(dst, scratch, packed)
			require.NoError(t, err)
			assert.Equal(t, values, unpacked)
		})
	}
}

func TestUint64_ZigzagBoundaryStress(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x80000000 + uint64(i)
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(Delta, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, scratch, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestUint64_ZigzagBoundaryStress_NearIntMin(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(0x7FFFFFF0+i) + uint64(i)*16
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(Delta, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, scratch, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestUint64_ZigzagBoundaryStress_UpperHalves(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(0x80000000)*uint64(1<<32) + uint64(i)
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(Delta, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, scratch, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestUint64_CompressionRatio_Sequential(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i + 1)
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	uncompressed := blockSize * 8
	assert.Less(t, len(packed), uncompressed,
		"sequential values should compress: packed=%d, raw=%d", len(packed), uncompressed)
}

func TestUint64_CompressionRatio_Timestamps(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(1_719_300_000_000)
	for i := range values {
		values[i] = base + uint64(i)
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	uncompressed := blockSize * 8
	assert.Less(t, len(packed), uncompressed/4,
		"timestamps should compress well: packed=%d, raw=%d", len(packed), uncompressed)
}

func TestUint64_CompressionRatio_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = rng.Uint64()
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	uncompressed := blockSize * 8
	assert.Less(t, len(packed), uncompressed*2,
		"random values should not expand excessively: packed=%d, raw=%d", len(packed), uncompressed)
}

func TestUint64_CompressionRatio_ClusteredAbove32(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(0x100000000)
	for i := range values {
		values[i] = base + uint64(i)*7
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	uncompressed := blockSize * 8
	assert.Less(t, len(packed), uncompressed,
		"clustered values above 2^32 should compress: packed=%d, raw=%d", len(packed), uncompressed)
}

func TestUint64_TwoBlockOverhead_UniformUpper_FOR64Eliminates(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	_, _, _, _, _, _, _, _, _, hasCombine := decodeHeader(header)
	assert.False(t, hasCombine,
		"FOR64 should eliminate the two-block path for uniform upper halves")

	uncompressed := blockSize * 8
	assert.Less(t, len(packed), uncompressed/4,
		"FOR64 single-block should compress well: packed=%d, raw=%d", len(packed), uncompressed)
}

func TestUint64_TwoBlockOverhead_ForcedTwoBlock(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(NoFOR, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	_, _, _, _, _, _, _, _, _, hasCombine := decodeHeader(header)
	require.True(t, hasCombine, "NoFOR should force two-block path")

	_, _, _, _, fw, hasExceptions, _, _, _, _ := decodeHeader(header)
	forBBytes := forBaseBytes(fw)
	block2LenVal := int(readBlock2Len(packed, forBBytes, hasExceptions))

	t.Logf("Block 2 size with NoFOR forced two-block: %d bytes", block2LenVal)

	uncompressed := blockSize * 8
	assert.Less(t, len(packed), uncompressed,
		"two-block encoding should still be smaller than uncompressed: packed=%d, raw=%d",
		len(packed), uncompressed)

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, scratch, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestUint64_TwoBlockOverhead_WideSpread(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		if i < blockSize/2 {
			values[i] = uint64(i)
		} else {
			values[i] = 0xFFFFFFFF00000000 + uint64(i)
		}
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	_, _, _, _, _, _, _, _, _, hasCombine := decodeHeader(header)
	assert.True(t, hasCombine,
		"wide-spread values should use two-block path (range >= 2^32)")

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, scratch, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestUint64_FOR64CompressionImprovement(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFC0 + uint64(i)
	}

	scratch := make([]uint32, ScratchLen64)

	packedFOR64, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	packedTwoBlock, err := PackUint64(NoFOR, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	t.Logf("FOR64 block: %d bytes, two-block: %d bytes, savings: %d bytes (%.1f%%)",
		len(packedFOR64), len(packedTwoBlock),
		len(packedTwoBlock)-len(packedFOR64),
		float64(len(packedTwoBlock)-len(packedFOR64))/float64(len(packedTwoBlock))*100)

	assert.Less(t, len(packedFOR64), len(packedTwoBlock),
		"FOR64 should produce smaller output than two-block for boundary-crossing data")

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, scratch, packedFOR64)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)

	unpacked, _, err = UnpackUint64(dst, scratch, packedTwoBlock)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestUint64_FOR64CompressionImprovement_Timestamps(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(1_719_300_000_000)
	for i := range values {
		values[i] = base + uint64(i)
	}

	scratch := make([]uint32, ScratchLen64)

	packedFOR64, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	packedNoFOR, err := PackUint64(NoFOR, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	t.Logf("FOR64 block: %d bytes, NoFOR two-block: %d bytes",
		len(packedFOR64), len(packedNoFOR))

	uncompressed := blockSize * 8
	assert.Less(t, len(packedFOR64), uncompressed/4,
		"FOR64 timestamps should compress well")

	dst := make([]uint64, blockSize)
	unpacked, _, err := UnpackUint64(dst, scratch, packedFOR64)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestUint64_ZeroAllocations_Pack(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)*7
	}

	dst := make([]byte, 0, MaxBlockLength64(0))
	scratch := make([]uint32, ScratchLen64)

	allocs := testing.AllocsPerRun(10, func() {
		dst = dst[:0]
		_, _ = PackUint64(0, values, dst, scratch)
	})
	assert.Equal(t, float64(0), allocs,
		"PackUint64 should have zero allocations with pre-allocated buffers")
}

func TestUint64_ZeroAllocations_Unpack(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)*7
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)

	allocs := testing.AllocsPerRun(10, func() {
		_, _, _ = UnpackUint64(dst, scratch, packed)
	})
	assert.Equal(t, float64(0), allocs,
		"UnpackUint64 should have zero allocations with pre-allocated buffers")
}

func TestUint64_ZeroAllocations_Pack_FitIn32(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i * 100)
	}

	dst := make([]byte, 0, MaxBlockLength64(0))
	scratch := make([]uint32, ScratchLen64)

	allocs := testing.AllocsPerRun(10, func() {
		dst = dst[:0]
		_, _ = PackUint64(0, values, dst, scratch)
	})
	assert.Equal(t, float64(0), allocs,
		"PackUint64 (fit-32) should have zero allocations with pre-allocated buffers")
}

func TestUint64_ZeroAllocations_Unpack_FitIn32(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i * 100)
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, values, nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)

	allocs := testing.AllocsPerRun(10, func() {
		_, _, _ = UnpackUint64(dst, scratch, packed)
	})
	assert.Equal(t, float64(0), allocs,
		"UnpackUint64 (fit-32) should have zero allocations with pre-allocated buffers")
}

func TestUint64_ZeroAllocations_Pack_TwoBlock(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = rng.Uint64()
	}

	dst := make([]byte, 0, MaxBlockLength64(0))
	scratch := make([]uint32, ScratchLen64)

	allocs := testing.AllocsPerRun(10, func() {
		dst = dst[:0]
		_, _ = PackUint64(0, values, dst, scratch)
	})
	assert.Equal(t, float64(0), allocs,
		"PackUint64 (two-block) should have zero allocations with pre-allocated buffers")
}

func TestUint64_ZeroAllocations_Unpack_TwoBlock(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = rng.Uint64()
	}

	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(0, slices.Clone(values), nil, scratch)
	require.NoError(t, err)

	dst := make([]uint64, blockSize)

	allocs := testing.AllocsPerRun(10, func() {
		_, _, _ = UnpackUint64(dst, scratch, packed)
	})
	assert.Equal(t, float64(0), allocs,
		"UnpackUint64 (two-block) should have zero allocations with pre-allocated buffers")
}

func TestPackUnpackUint64_AllPaths_AllFlags(t *testing.T) {
	genFit32 := func() []uint64 {
		v := make([]uint64, blockSize)
		for i := range v {
			v[i] = uint64(i * 100)
		}
		return v
	}
	genFOR64 := func() []uint64 {
		v := make([]uint64, blockSize)
		for i := range v {
			v[i] = 0x100000000 + uint64(i)*7
		}
		return v
	}
	genTwoBlock := func() []uint64 {
		v := make([]uint64, blockSize)
		for i := range v {
			if i < blockSize/2 {
				v[i] = uint64(i)
			} else {
				v[i] = 0xFFFFFFFF00000000 + uint64(i)
			}
		}
		return v
	}

	paths := []struct {
		name string
		gen  func() []uint64
	}{
		{"fit32", genFit32},
		{"for64", genFOR64},
		{"two_block", genTwoBlock},
	}

	flags := []struct {
		name string
		flag Flag
	}{
		{"default", 0},
		{"Delta", Delta},
		{"NoPatch", NoPatch},
		{"NoFOR", NoFOR},
		{"Delta_NoPatch", Delta | NoPatch},
		{"Delta_NoFOR", Delta | NoFOR},
		{"NoPatch_NoFOR", NoPatch | NoFOR},
		{"Delta_NoPatch_NoFOR", Delta | NoPatch | NoFOR},
	}

	for _, p := range paths {
		for _, fl := range flags {
			t.Run(p.name+"/"+fl.name, func(t *testing.T) {
				values := p.gen()
				packUnpackUint64RoundTrip(t, fl.flag, values)
			})
		}
	}
}

func TestPackUnpackUint64_FOR64_SingleValueAbove32Bits(t *testing.T) {
	packUnpackUint64RoundTrip(t, 0, []uint64{0x200000042})
}

func TestPackUnpackUint64_FOR64_AllSameAbove32Bits(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x300000000
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_FOR64_NearMaxUint64SmallRange(t *testing.T) {
	values := make([]uint64, blockSize)
	base := uint64(math.MaxUint64 - blockSize)
	for i := range values {
		values[i] = base + uint64(i)
	}
	packUnpackUint64RoundTrip(t, 0, values)
}

func TestPackUnpackUint64_FOR64_DeltaNoPatch(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)*3
	}
	packUnpackUint64RoundTrip(t, Delta|NoPatch, values)
}

func TestGetUint64_TwoBlock_WithDelta(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		if i < blockSize/2 {
			values[i] = uint64(i)
		} else {
			values[i] = 0xFFFFFFFF00000000 + uint64(i)
		}
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

func TestGetUint64_FOR64_NoPatch(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = 0x100000000 + uint64(i)
	}
	scratch := make([]uint32, ScratchLen64)
	packed, err := PackUint64(NoPatch, values, nil, scratch)
	require.NoError(t, err)

	for i, want := range values {
		got, err := GetUint64(i, packed, scratch)
		require.NoError(t, err, "pos=%d", i)
		assert.Equal(t, want, got, "pos=%d", i)
	}
}

func TestGetUint64_Fit32_WithDelta(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i * 100)
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
