package utlpfor

import (
	"bytes"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNoInPlace_FlagValue(t *testing.T) {
	assert.Equal(t, Flag(1<<5), NoInPlace, "NoInPlace must be bit 5")
	assert.Equal(t, Flag(0), NoInPlace&Delta, "NoInPlace must not overlap Delta")
	assert.Equal(t, Flag(0), NoInPlace&NoFOR, "NoInPlace must not overlap NoFOR")
	assert.Equal(t, Flag(0), NoInPlace&NoPatch, "NoInPlace must not overlap NoPatch")
	assert.Equal(t, Flag(0), NoInPlace&Special, "NoInPlace must not overlap Special")
	assert.Equal(t, Flag(0), NoInPlace&Append, "NoInPlace must not overlap Append")
	assert.Equal(t, 2*blockSize, ScratchLenNoInPlace, "ScratchLenNoInPlace must be 2*blockSize")
}

func TestNoInPlace_ValuesUnmodified(t *testing.T) {
	scratch := make([]uint32, ScratchLenNoInPlace)

	t.Run("NoInPlace_FOR_active", func(t *testing.T) {
		// min > 0 triggers FOR subtraction
		values := make([]uint32, 128)
		for i := range values {
			values[i] = uint32(1000 + i)
		}
		original := slices.Clone(values)

		_, err := PackUint32(NoInPlace, values, nil, scratch)
		require.NoError(t, err)
		assert.Equal(t, original, values,
			"NoInPlace must not modify the values slice (FOR subtraction)")
	})

	t.Run("NoInPlace_Delta_FOR", func(t *testing.T) {
		// Delta + FOR both modify
		values := make([]uint32, 128)
		for i := range values {
			values[i] = uint32(1000 + i*10)
		}
		original := slices.Clone(values)

		_, err := PackUint32(NoInPlace|Delta, values, nil, scratch)
		require.NoError(t, err)
		assert.Equal(t, original, values,
			"NoInPlace|Delta must not modify the values slice")
	})

	t.Run("NoInPlace_Delta_NoFOR", func(t *testing.T) {
		// Delta only, no FOR
		values := make([]uint32, 128)
		for i := range values {
			values[i] = uint32(i * 10)
		}
		original := slices.Clone(values)

		_, err := PackUint32(NoInPlace|Delta|NoFOR, values, nil, scratch)
		require.NoError(t, err)
		assert.Equal(t, original, values,
			"NoInPlace|Delta|NoFOR must not modify the values slice")
	})

	t.Run("NoInPlace_NoPatch", func(t *testing.T) {
		values := make([]uint32, 128)
		for i := range values {
			values[i] = uint32(500 + i)
		}
		original := slices.Clone(values)

		_, err := PackUint32(NoInPlace|NoPatch, values, nil, scratch)
		require.NoError(t, err)
		assert.Equal(t, original, values,
			"NoInPlace|NoPatch must not modify the values slice")
	})
}

func TestNoInPlace_OutputIdentical(t *testing.T) {
	// Packed output with NoInPlace must be byte-identical to cloning + packing without it.
	scratch := make([]uint32, ScratchLenNoInPlace)

	flagSets := []struct {
		name string
		flag Flag
	}{
		{"NoInPlace", NoInPlace},
		{"NoInPlace_Delta", NoInPlace | Delta},
		{"NoInPlace_NoPatch", NoInPlace | NoPatch},
		{"NoInPlace_Delta_NoFOR", NoInPlace | Delta | NoFOR},
	}

	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(1000 + i*7)
	}

	for _, tc := range flagSets {
		t.Run(tc.name, func(t *testing.T) {
			// Pack with NoInPlace
			v1 := slices.Clone(values)
			out1, err := PackUint32(tc.flag, v1, nil, scratch)
			require.NoError(t, err)

			// Pack without NoInPlace using a clone
			v2 := slices.Clone(values)
			baseFlag := tc.flag &^ NoInPlace
			out2, err := PackUint32(baseFlag, v2, nil, scratch)
			require.NoError(t, err)

			assert.True(t, bytes.Equal(out1, out2),
				"NoInPlace output must be byte-identical to non-NoInPlace output")
		})
	}
}

func TestNoInPlace_RoundTrip(t *testing.T) {
	scratch := make([]uint32, ScratchLenNoInPlace)

	flagSets := []struct {
		name string
		flag Flag
	}{
		{"NoInPlace", NoInPlace},
		{"NoInPlace_Delta", NoInPlace | Delta},
		{"NoInPlace_NoPatch", NoInPlace | NoPatch},
		{"NoInPlace_Delta_NoFOR", NoInPlace | Delta | NoFOR},
		{"NoInPlace_Delta_NoPatch", NoInPlace | Delta | NoPatch},
	}

	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(500 + i*3)
	}

	for _, tc := range flagSets {
		t.Run(tc.name, func(t *testing.T) {
			original := slices.Clone(values)
			packed, err := PackUint32(tc.flag, values, nil, scratch)
			require.NoError(t, err)

			unpacked, _, err := UnpackUint32(packed, nil, scratch)
			require.NoError(t, err)
			assert.Equal(t, original, unpacked)
		})
	}
}

func TestNoInPlace_ShortBlock(t *testing.T) {
	scratch := make([]uint32, ScratchLenNoInPlace)

	for _, count := range []int{1, 5, 16, 64, 100} {
		t.Run("count_"+string(rune('0'+count/100))+string(rune('0'+(count/10)%10))+string(rune('0'+count%10)), func(t *testing.T) {
			values := make([]uint32, count)
			for i := range values {
				values[i] = uint32(200 + i)
			}
			original := slices.Clone(values)

			packed, err := PackUint32(NoInPlace, values, nil, scratch)
			require.NoError(t, err)
			assert.Equal(t, original, values, "values must be unmodified")

			unpacked, _, err := UnpackUint32(packed, nil, scratch)
			require.NoError(t, err)
			assert.Equal(t, original, unpacked)
		})
	}
}

func TestNoInPlace64_ValuesUnmodified(t *testing.T) {
	scratch := make([]uint32, ScratchLen64)

	t.Run("NoInPlace", func(t *testing.T) {
		values := make([]uint64, 128)
		for i := range values {
			values[i] = uint64(1_000_000_000_000 + i*7)
		}
		original := slices.Clone(values)

		_, err := PackUint64(NoInPlace, values, nil, scratch)
		require.NoError(t, err)
		assert.Equal(t, original, values,
			"PackUint64 with NoInPlace must not modify the values slice")
	})

	t.Run("Append", func(t *testing.T) {
		values := make([]uint64, 128)
		for i := range values {
			values[i] = uint64(1_000_000_000_000 + i*7)
		}
		original := slices.Clone(values)

		var dst []byte
		var err error
		for range 3 {
			dst, err = PackUint64(Append, values, dst, scratch)
			require.NoError(t, err)
		}
		assert.Equal(t, original, values,
			"PackUint64 with Append must not modify the values slice")
	})

	t.Run("SmallValues", func(t *testing.T) {
		// All fit in 32 bits
		values := make([]uint64, 128)
		for i := range values {
			values[i] = uint64(1000 + i)
		}
		original := slices.Clone(values)

		_, err := PackUint64(NoInPlace, values, nil, scratch)
		require.NoError(t, err)
		assert.Equal(t, original, values)
	})
}

func TestNoInPlace64_RoundTrip(t *testing.T) {
	scratch := make([]uint32, ScratchLen64)

	flagSets := []struct {
		name string
		flag Flag
	}{
		{"NoInPlace", NoInPlace},
		{"NoInPlace_Delta", NoInPlace | Delta},
		{"Append", Append},
	}

	values := make([]uint64, 128)
	for i := range values {
		values[i] = uint64(1_000_000_000_000 + i*3)
	}

	for _, tc := range flagSets {
		t.Run(tc.name, func(t *testing.T) {
			original := slices.Clone(values)
			packed, err := PackUint64(tc.flag, values, nil, scratch)
			require.NoError(t, err)

			unpacked, _, err := UnpackUint64(packed, nil, scratch)
			require.NoError(t, err)
			assert.Equal(t, original, unpacked)
		})
	}
}

func TestNoInPlace_ScratchTooSmall(t *testing.T) {
	// When scratch is too small, the library auto-allocates internally.
	smallScratch := make([]uint32, ScratchLen) // 128, not 256

	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(1000 + i)
	}
	original := slices.Clone(values)

	packed, err := PackUint32(NoInPlace, values, nil, smallScratch)
	require.NoError(t, err)
	assert.Equal(t, original, values, "values must be unmodified even with small scratch")

	unpacked, _, err := UnpackUint32(packed, nil, smallScratch)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}
