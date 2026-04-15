package utlpfor

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetUint32_AllPositions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i*17 + 3)
	}
	packed, _ := PackUint32(0, nil, nil, values)

	for pos := range 128 {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, values[pos], got, "pos=%d", pos)
	}
}

func TestGetUint32_AllPositions_AllBitWidths(t *testing.T) {
	for bw := 1; bw <= 32; bw++ {
		t.Run(fmt.Sprintf("bw%d", bw), func(t *testing.T) {
			values := make([]uint32, 128)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range values {
				values[i] = uint32(i*13+7) & mask
			}
			packed, _ := PackUint32(0, nil, nil, values)

			for pos := range 128 {
				got, err := GetUint32(pos, packed)
				require.NoError(t, err)
				assert.Equal(t, values[pos], got, "bw=%d pos=%d", bw, pos)
			}
		})
	}
}

func TestGetUint32_MatchesUnpack(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * i)
	}
	packed, _ := PackUint32(0, nil, nil, values)

	unpacked, _, _ := UnpackUint32(nil, make([]uint32, 128), packed)
	for pos := range unpacked {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

func TestGetUint32_OutOfRange(t *testing.T) {
	values := make([]uint32, 50)
	packed, _ := PackUint32(0, nil, nil, values)

	_, err := GetUint32(50, packed)
	assert.ErrorIs(t, err, ErrPositionOutOfRange)

	_, err = GetUint32(127, packed)
	assert.ErrorIs(t, err, ErrPositionOutOfRange)
}

func TestGetUint32_NegativePosition(t *testing.T) {
	values := make([]uint32, 128)
	packed, _ := PackUint32(0, nil, nil, values)

	_, err := GetUint32(-1, packed)
	assert.ErrorIs(t, err, ErrPositionOutOfRange)
}

func TestGetUint32_ShortBuffer(t *testing.T) {
	_, err := GetUint32(0, []byte{0x01})
	assert.Error(t, err)
}

func TestGetUint32_ZeroAllocations(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, _ := PackUint32(0, nil, nil, values)

	allocs := testing.AllocsPerRun(100, func() {
		GetUint32(42, packed)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestGetUint32_AllZeros(t *testing.T) {
	values := make([]uint32, 128)
	packed, _ := PackUint32(0, nil, nil, values)

	for pos := range 128 {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, uint32(0), got, "pos=%d", pos)
	}
}

func TestGetUint32_AllMax(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 0xFFFFFFFF
	}
	packed, _ := PackUint32(0, nil, nil, values)

	for pos := range 128 {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, uint32(0xFFFFFFFF), got, "pos=%d", pos)
	}
}

func TestGetUint32_ZeroAllocs(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)

	allocs := testing.AllocsPerRun(100, func() {
		GetUint32(42, packed)
	})
	assert.Equal(t, float64(0), allocs)
}
