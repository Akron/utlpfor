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
			packed, err := PackUint32(0, nil, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
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
			packed, err := PackUint32(0, nil, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
			require.NoError(t, err)
			assert.Equal(t, original, unpacked)
		})
	}
}

func TestPackUint32_AllZeros(t *testing.T) {
	values := make([]uint32, 128)
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUint32_AllMax(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 0xFFFFFFFF
	}
	original := append([]uint32(nil), values...)
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_DstGrowth(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	dst := make([]byte, 0, 10)
	packed, err := PackUint32(0, dst, nil, values)
	require.NoError(t, err)
	assert.NotNil(t, packed)
}

func TestPackUint32_DeterministicOutput(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed1, _ := PackUint32(0, nil, nil, values)
	packed2, _ := PackUint32(0, nil, nil, values)
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
			packed, err := PackUint32(0, nil, nil, tt.values)
			require.NoError(t, err)
			rawSize := len(tt.values) * 4
			assert.Less(t, len(packed), rawSize,
				"packed size %d should be less than raw size %d", len(packed), rawSize)
		})
	}
}

func TestPackUint32_EmptyValues(t *testing.T) {
	_, err := PackUint32(0, nil, nil, []uint32{})
	assert.Error(t, err)
}

func TestPackUint32_TooManyValues(t *testing.T) {
	_, err := PackUint32(0, nil, nil, make([]uint32, 129))
	assert.Error(t, err)
}

func TestPackUint32_DstReuse(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	dst := make([]byte, 1024)
	packed, err := PackUint32(0, dst, nil, values)
	require.NoError(t, err)
	assert.True(t, cap(packed) >= cap(dst), "should reuse provided dst buffer")
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
			packed, err := PackUint32(NoPatch, nil, nil, values)
			require.NoError(t, err)

			header := bo.Uint32(packed)
			excCount := int((header >> headerExcCountShift) & headerExcCountMask)
			assert.Equal(t, 0, excCount, "NoPatch must produce zero exceptions")

			unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
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
			packed, err := PackUint32(NoPatch, nil, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
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

	packed, err := PackUint32(NoPatch, nil, nil, values)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount, "NoPatch must never produce exceptions")

	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_NoPatch_DeltaCombination(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	expected := slices.Clone(values)

	packed, err := PackUint32(Delta|NoPatch, nil, nil, values)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount)

	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestPackUint32_NoPatch_AllZeros(t *testing.T) {
	values := make([]uint32, blockSize)
	packed, err := PackUint32(NoPatch, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUint32_NoPatch_AllMax(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFFF
	}
	original := slices.Clone(values)
	packed, err := PackUint32(NoPatch, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUint32_NoPatch_DictionaryCompressed(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i*8 + i%3)
	}
	original := slices.Clone(values)

	packed, err := PackUint32(NoPatch, nil, nil, values)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
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
		p, err := PackUint32(NoPatch, nil, nil, clone)
		require.NoError(t, err)
		return p
	}()

	header := bo.Uint32(packed)
	forWidth := int((header >> forWidthShift) & forWidthMask)
	assert.NotEqual(t, 0, forWidth, "NoPatch should still allow FOR when beneficial")
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount, "NoPatch must still produce zero exceptions")

	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
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
		p, err := PackUint32(NoPatch|NoFOR, nil, nil, clone)
		require.NoError(t, err)
		return p
	}()

	header := bo.Uint32(packed)
	forWidth := int((header >> forWidthShift) & forWidthMask)
	assert.Equal(t, 0, forWidth, "NoPatch|NoFOR must skip FOR")
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount, "NoPatch must produce zero exceptions")

	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
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
		p, err := PackUint32(NoPatch, nil, nil, clone)
		require.NoError(t, err)
		return p
	}()

	packedNoFOR := func() []byte {
		clone := slices.Clone(values)
		p, err := PackUint32(NoPatch|NoFOR, nil, nil, clone)
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

			packed, err := PackUint32(NoPatch, nil, nil, values)
			require.NoError(t, err)

			unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
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

	packed, err := PackUint32(Delta|NoPatch, nil, nil, values)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	excCount := int((header >> headerExcCountShift) & headerExcCountMask)
	assert.Equal(t, 0, excCount, "NoPatch must produce zero exceptions")

	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestPackUint32_NoPatch_GetMatchesUnpack(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i % 1024)
	}
	packed, err := PackUint32(NoPatch, nil, nil, values)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)

	for pos := range unpacked {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

func TestPackUint32_NoPatch_BlockLengthConsistency(t *testing.T) {
	for _, flag := range []byte{NoPatch, NoPatch | Delta} {
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = uint32(i % 1024)
		}
		clone := slices.Clone(values)
		packed, err := PackUint32(flag, nil, nil, clone)
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
		dst, _ = PackUint32(0, dst[:0], scratch, values)
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
		dst, _ = PackUint32(0, dst[:0], scratch, values)
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
		dst, _ = PackUint32(Delta, dst[:0], scratch, values)
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
		dst, err = PackUint32(0, dst[:0], scratch, values)
		require.NoError(t, err)

		unpacked, _, uErr := UnpackUint32(nil, scratch, dst)
		require.NoError(t, uErr)
		assert.Equal(t, original, unpacked)
	}
}
