package utlpfor

import (
	"fmt"
	"math/bits"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPackUint32_RoundTrip_AllBitWidths(t *testing.T) {
	for bw := 1; bw <= 32; bw++ {
		t.Run(fmt.Sprintf("bw%d", bw), func(t *testing.T) {
			values := make([]uint32, 128)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range values {
				values[i] = uint32(i*7+3) & mask
			}
			original := slices.Clone(values)
			packed, err := PackUint32(0, values, nil, nil)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, 128))
			require.NoError(t, err)
			assert.Equal(t, original, unpacked)
		})
	}
}

func TestPackUint32_PartialBlock(t *testing.T) {
	for _, count := range []int{1, 15, 50, 100, 127} {
		t.Run(fmt.Sprintf("count%d", count), func(t *testing.T) {
			values := make([]uint32, count)
			for i := range values {
				values[i] = uint32(i)
			}
			original := slices.Clone(values)
			packed, err := PackUint32(0, values, nil, nil)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, 128))
			require.NoError(t, err)
			assert.Equal(t, original, unpacked)
		})
	}
}

func TestPackUint32_AllZeros(t *testing.T) {
	values := make([]uint32, 128)
	packed, err := PackUint32(0, values, nil, nil)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, 128))
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUint32_AllMax(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 0xFFFFFFFF
	}
	original := append([]uint32(nil), values...)
	packed, err := PackUint32(0, values, nil, nil)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, 128))
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_DstGrowth(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	dst := make([]byte, 0, 10)
	packed, err := PackUint32(0, values, dst, nil)
	require.NoError(t, err)
	assert.NotNil(t, packed)
}

func TestPackUint32_DeterministicOutput(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed1, _ := PackUint32(0, values, nil, nil)
	packed2, _ := PackUint32(0, values, nil, nil)
	assert.Equal(t, packed1, packed2)
}

func TestPackUint32_CompressesData(t *testing.T) {
	tests := []struct {
		name   string
		values []uint32
	}{
		{"small_values", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i % 16)
			}
			return v
		}()},
		{"medium_values", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i * 100)
			}
			return v
		}()},
		{"byte_range", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i)
			}
			return v
		}()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			packed, err := PackUint32(0, tt.values, nil, nil)
			require.NoError(t, err)
			rawSize := len(tt.values) * 4
			assert.Less(t, len(packed), rawSize,
				"packed size %d should be less than raw size %d", len(packed), rawSize)
		})
	}
}

func TestPackUint32_EmptyValues(t *testing.T) {
	_, err := PackUint32(0, []uint32{}, nil, nil)
	assert.Error(t, err)
}

func TestPackUint32_TooManyValues(t *testing.T) {
	_, err := PackUint32(0, make([]uint32, 129), nil, nil)
	assert.Error(t, err)
}

func TestPackUint32_DstReuse(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	dst := make([]byte, 1024)
	packed, err := PackUint32(0, values, dst, nil)
	require.NoError(t, err)
	assert.True(t, cap(packed) >= cap(dst), "should reuse provided dst buffer")
}

func TestPackUint32_SpecialFlagSetsHeaderBit(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}

	packed, err := PackUint32(Special, values, nil, nil)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	assert.True(t, header&headerSpecialFlag != 0, "Special flag must set bit 17")
}

func TestPackUint32_SpecialFlagRoundTrip(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i * 13)
	}
	original := slices.Clone(values)

	packed, err := PackUint32(Special|Delta, values, nil, nil)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_NoPatch_RoundTrip_AllBitWidths(t *testing.T) {
	for bw := 1; bw <= 32; bw++ {
		t.Run(fmt.Sprintf("bw%d", bw), func(t *testing.T) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range values {
				values[i] = uint32(i*7+3) & mask
			}
			original := slices.Clone(values)
			packed, err := PackUint32(NoPatch, values, nil, nil)
			require.NoError(t, err)

			header := bo.Uint32(packed)
			excCount := int((header >> headerExcCountShift) & headerExcCountMask)
			assert.Equal(t, 0, excCount, "NoPatch must produce zero exceptions")

			unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
			require.NoError(t, err)
			assert.Equal(t, original, unpacked)
		})
	}
}

func TestPackUint32_NoPatch_PartialBlock(t *testing.T) {
	for _, count := range []int{1, 15, 50, 100, 127} {
		t.Run(fmt.Sprintf("count%d", count), func(t *testing.T) {
			values := make([]uint32, count)
			for i := range values {
				values[i] = uint32(i)
			}
			original := slices.Clone(values)
			packed, err := PackUint32(NoPatch, values, nil, nil)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
			require.NoError(t, err)
			assert.Equal(t, original, unpacked)
		})
	}
}

func TestPackUint32_NoPatch_WithOutliers(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i % 256)
	}
	values[50] = 0xFFFFFF
	values[100] = 0xFFFFFFF
	original := slices.Clone(values)

	packed, err := PackUint32(NoPatch, values, nil, nil)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount, "NoPatch must never produce exceptions")

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_NoPatch_DeltaCombination(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	expected := slices.Clone(values)

	packed, err := PackUint32(Delta|NoPatch, values, nil, nil)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount)

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestPackUint32_NoPatch_AllZeros(t *testing.T) {
	values := make([]uint32, blockSize)
	packed, err := PackUint32(NoPatch, values, nil, nil)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUint32_NoPatch_AllMax(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFFF
	}
	original := slices.Clone(values)
	packed, err := PackUint32(NoPatch, values, nil, nil)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_NoPatch_DictionaryCompressed(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i*8 + i%3)
	}
	original := slices.Clone(values)

	packed, err := PackUint32(NoPatch, values, nil, nil)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)

	header := bo.Uint32(packed)
	encodedBW := int((header >> headerWidthShift) & headerWidthMask)
	bitWidth := encodedBW * 4
	maxVal := values[blockSize-1]
	expectedBW := roundUpToStep(bits.Len32(maxVal))
	assert.Equal(t, expectedBW, bitWidth, "bitwidth should match step-rounded max bits")
}

func TestPackUint32_NoPatch_AllowsFOR(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}

	packed := func() []byte {
		clone := slices.Clone(values)
		p, err := PackUint32(NoPatch, clone, nil, nil)
		require.NoError(t, err)
		return p
	}()

	header := bo.Uint32(packed)
	forWidth := int((header >> forWidthShift) & forWidthMask)
	assert.NotEqual(t, 0, forWidth, "NoPatch should still allow FOR when beneficial")
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount, "NoPatch must still produce zero exceptions")

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUint32_NoPatch_NoFOR_Explicit(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}

	packed := func() []byte {
		clone := slices.Clone(values)
		p, err := PackUint32(NoPatch|NoFOR, clone, nil, nil)
		require.NoError(t, err)
		return p
	}()

	header := bo.Uint32(packed)
	forWidth := int((header >> forWidthShift) & forWidthMask)
	assert.Equal(t, 0, forWidth, "NoPatch|NoFOR must skip FOR")
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount, "NoPatch must produce zero exceptions")

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUint32_NoPatch_FOR_SmallerOutput(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 1000000 + uint32(i%100)
	}

	packedWithFOR := func() []byte {
		clone := slices.Clone(values)
		p, err := PackUint32(NoPatch, clone, nil, nil)
		require.NoError(t, err)
		return p
	}()

	packedNoFOR := func() []byte {
		clone := slices.Clone(values)
		p, err := PackUint32(NoPatch|NoFOR, clone, nil, nil)
		require.NoError(t, err)
		return p
	}()

	assert.Less(t, len(packedWithFOR), len(packedNoFOR),
		"NoPatch with FOR should produce smaller output for clustered data")
}

func TestPackUint32_NoPatch_FOR_RoundTrip(t *testing.T) {
	for _, count := range []int{1, 15, 50, 100, 128} {
		t.Run(fmt.Sprintf("count%d", count), func(t *testing.T) {
			values := make([]uint32, count)
			for i := range values {
				values[i] = 500000 + uint32(i*3)
			}
			original := slices.Clone(values)

			packed, err := PackUint32(NoPatch, values, nil, nil)
			require.NoError(t, err)

			unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
			require.NoError(t, err)
			assert.Equal(t, original, unpacked)
		})
	}
}

func TestPackUint32_NoPatch_Delta_FOR_RoundTrip(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 1000000 + uint32(i*100)
	}
	expected := slices.Clone(values)

	packed, err := PackUint32(Delta|NoPatch, values, nil, nil)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount, "NoPatch must produce zero exceptions")

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestPackUint32_NoPatch_GetMatchesUnpack(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i % 1024)
	}
	packed, err := PackUint32(NoPatch, values, nil, nil)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)

	scratch := make([]uint32, ScratchLen)
	for pos := range unpacked {
		got, err := GetUint32(pos, packed, scratch)
		require.NoError(t, err)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

func TestPackUint32_NoPatch_BlockLengthConsistency(t *testing.T) {
	for _, flag := range []Flag{NoPatch, NoPatch | Delta} {
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = uint32(i % 1024)
		}
		clone := slices.Clone(values)
		packed, err := PackUint32(flag, clone, nil, nil)
		require.NoError(t, err)

		blockLen, err := BlockLength(packed)
		require.NoError(t, err)
		assert.Equal(t, len(packed), blockLen, "flag=%d", flag)
	}
}

func TestPackUint32_ZeroAllocs(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i % 200)
	}
	scratch := make([]uint32, ScratchLen)
	dst := make([]byte, 0, 1024)

	allocs := testing.AllocsPerRun(100, func() {
		dst, _ = PackUint32(0, values, dst[:0], scratch)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestPackUint32_ZeroAllocs_WithExceptions(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i % 10)
	}
	values[10] = 0x10000000
	values[50] = 0x20000000
	values[99] = 0xFFFFFF
	scratch := make([]uint32, ScratchLen)
	dst := make([]byte, 0, 1024)

	allocs := testing.AllocsPerRun(100, func() {
		dst, _ = PackUint32(0, values, dst[:0], scratch)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestPackUint32_ZeroAllocs_Delta(t *testing.T) {
	source := make([]uint32, blockSize)
	for i := range source {
		source[i] = 1000 + uint32(i*3)
	}
	values := make([]uint32, blockSize)
	scratch := make([]uint32, ScratchLen)
	dst := make([]byte, 0, 1024)

	allocs := testing.AllocsPerRun(100, func() {
		copy(values, source)
		dst, _ = PackUint32(Delta, values, dst[:0], scratch)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestPackUint32_ZeroAllocs_DstAndScratchReuse(t *testing.T) {
	scratch := make([]uint32, ScratchLen)
	dst := make([]byte, 0, 1024)

	for trial := range 10 {
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = uint32(trial*blockSize + i)
		}
		original := slices.Clone(values)
		var err error
		dst, err = PackUint32(0, values, dst[:0], scratch)
		require.NoError(t, err)

		unpacked, _, uErr := UnpackUint32(dst, nil, scratch)
		require.NoError(t, uErr)
		assert.Equal(t, original, unpacked)
	}
}

func TestPackUint32_PartialBlock_WithExceptions(t *testing.T) {
	for _, count := range []int{15, 50, 100, 127} {
		values := make([]uint32, count)
		for i := range values {
			values[i] = uint32(i % 16)
		}
		values[0] = 0x10000000
		if count > 10 {
			values[10] = 0x20000000
		}
		original := make([]uint32, count)
		copy(original, values)

		packed, err := PackUint32(0, values, nil, nil)
		require.NoError(t, err)

		unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
		require.NoError(t, err)
		assert.Equal(t, original, unpacked)
	}
}

func TestEnsureCapacity_GeometricGrowth(t *testing.T) {
	// Regression test for the fixed-quantum Append growth policy:
	// ensureCapacity32 used to grow dst by
	// exactly maxBlockLen32 on every realloc. Because one Append consumes
	// blockLen bytes, cap(dst)-off fell back below the quantum after
	// virtually every call, so a sequential Append stream paid ~1 realloc
	// plus a full-prefix copy per call (O(n^2) memcpy, ~1 alloc per Append).
	// Geometric (doubling) growth amortizes to O(log n) reallocs.
	const nBlocks = 300

	block := make([]uint32, blockSize)
	for i := range block {
		block[i] = uint32(i % 1024) // bw10, no exceptions
	}
	scratch := make([]uint32, ScratchLenNoInPlace)

	t.Run("sequential_appends_reallocs_log_n", func(t *testing.T) {
		var dst []byte
		allocs := testing.AllocsPerRun(5, func() {
			dst = nil
			var err error
			for range nBlocks {
				dst, err = PackUint32(Append, block, dst, scratch)
				require.NoError(t, err)
			}
		})
		// Old policy: one realloc per Append (~300 allocs). Doubling policy:
		// O(log2(n/quantum)) reallocs (~7). Pinned at <= 12 so future policy
		// tweaks keep slack without re-enabling quadratic behavior.
		assert.LessOrEqual(t, allocs, float64(12),
			"sequential Append into a fresh buffer must realloc O(log n) times, not once per call")
	})

	t.Run("contract_capacity_and_prefix", func(t *testing.T) {
		var dst []byte
		var err error
		prefixSnapshot := []byte{}
		for i := range nBlocks {
			off := len(dst)
			capBefore := cap(dst)
			dst, err = PackUint32(Append, block, dst, scratch)
			require.NoError(t, err)

			if cap(dst) == capBefore {
				// Fast path: capacity already covered a full block at entry.
				assert.GreaterOrEqual(t, capBefore-off, maxBlockLen32, "call %d", i)
			} else {
				// Realloc path: contract guarantees room for one full block
				// at the entry offset, and the prefix must be preserved.
				assert.GreaterOrEqual(t, cap(dst), off+maxBlockLen32, "call %d", i)
				assert.Equal(t, prefixSnapshot, dst[:off], "realloc must preserve prefix (call %d)", i)
			}
			prefixSnapshot = dst[:len(dst)]

			if i%50 == 0 {
				unpacked, _, uErr := UnpackUint32(dst[off:], nil, scratch)
				require.NoError(t, uErr, "call %d", i)
				assert.Equal(t, block, unpacked, "call %d", i)
			}
		}

		// The accumulated stream must decode as consecutive blocks.
		offset := 0
		for i := range nBlocks {
			unpacked, consumed, err := UnpackUint32(dst[offset:], nil, scratch)
			require.NoError(t, err, "block %d", i)
			assert.Equal(t, block, unpacked, "block %d", i)
			offset += consumed
		}
		assert.Equal(t, len(dst), offset)
	})

	t.Run("preallocated_exact_never_reallocs", func(t *testing.T) {
		// Case (a) contract: a caller following the README hint
		// (dst pre-allocated with MaxBlockLength32) stays zero-alloc.
		dst := make([]byte, 0, MaxBlockLength32(0))
		allocs := testing.AllocsPerRun(100, func() {
			dst = dst[:0]
			_, _ = PackUint32(Append, block, dst, scratch)
		})
		assert.Equal(t, float64(0), allocs)
	})
}
