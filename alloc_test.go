package utlpfor

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnpackUint32_ZeroAllocs(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	dst := make([]uint32, blockSize)
	scratch := make([]uint32, blockSize)

	allocs := testing.AllocsPerRun(100, func() {
		UnpackUint32(dst, scratch, packed)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestUnpackUint32_ZeroAllocs_WithExceptions(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}
	values[0] = 0xFFFFFFF
	values[64] = 0xFFFFFFF
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	dst := make([]uint32, blockSize)
	scratch := make([]uint32, blockSize)

	allocs := testing.AllocsPerRun(100, func() {
		UnpackUint32(dst, scratch, packed)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestGetUint32_ZeroAllocs(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	allocs := testing.AllocsPerRun(100, func() {
		GetUint32(42, packed)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestBlockLength_ZeroAllocs(t *testing.T) {
	values := make([]uint32, blockSize)
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	allocs := testing.AllocsPerRun(100, func() {
		BlockLength(packed)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestUnpackUint32_ScratchNotMutatedOnError(t *testing.T) {
	scratch := make([]uint32, blockSize)
	for i := range scratch {
		scratch[i] = 0xDEADBEEF
	}
	original := slices.Clone(scratch)

	_, _, err := UnpackUint32(nil, scratch, []byte{0xFF})
	assert.Error(t, err)
	assert.Equal(t, original, scratch)
}

func TestUnpackUint32_DstReuseAcrossCalls(t *testing.T) {
	dst := make([]uint32, 0, blockSize)
	scratch := make([]uint32, blockSize)

	for trial := range 10 {
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = uint32(trial*blockSize + i)
		}
		original := slices.Clone(values)
		packed, err := PackUint32(0, nil, values)
		require.NoError(t, err)

		var uErr error
		dst, _, uErr = UnpackUint32(dst, scratch, packed)
		require.NoError(t, uErr)
		assert.Equal(t, original, dst)
	}
}
