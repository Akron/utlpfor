package utlpfor

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPackUnpack_WithExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[10] = 0x10000
	values[50] = 0x1000000
	values[100] = 0xFFFFFF

	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUnpack_AllExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(0x10000000 + i)
	}
	original := slices.Clone(values)
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, original, unpacked)
}

func TestPackUnpack_NoExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i & 0xFF)
	}
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUnpack_SparseExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[0] = 0xFFFFFFF
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUnpack_ExactlyAtBitmapThreshold(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	for i := range 16 {
		values[i*8] = 0x10000000 + uint32(i)
	}
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUnpack_OneAboveBitmapThreshold(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	for i := range 17 {
		values[i*7] = 0x10000000 + uint32(i)
	}
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUnpack_VariousExceptionCounts(t *testing.T) {
	for excCount := 1; excCount <= 128; excCount += 13 {
		t.Run(fmt.Sprintf("exc%d", excCount), func(t *testing.T) {
			values := make([]uint32, 128)
			for i := range values {
				values[i] = uint32(i)
			}
			for i := 0; i < excCount && i < 128; i++ {
				idx := i * 128 / excCount
				if idx >= 128 {
					idx = 127
				}
				values[idx] = 0x10000000 + uint32(i)
			}
			packed, err := PackUint32(0, nil, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
			require.NoError(t, err)
			assert.Equal(t, values, unpacked)
		})
	}
}

func TestGetUint32_WithExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[42] = 0x100000
	packed, _ := PackUint32(0, nil, nil, values)

	got, err := GetUint32(10, packed)
	require.NoError(t, err)
	assert.Equal(t, uint32(10), got)

	got, err = GetUint32(42, packed)
	require.NoError(t, err)
	assert.Equal(t, uint32(0x100000), got)
}

func TestPackUnpack_ExceptionsWithCompression(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[0] = 0x100000
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	rawSize := len(values) * 4
	assert.Less(t, len(packed), rawSize,
		"even with exceptions, packed should be smaller than raw")
}

func TestPackUnpack_RandomVectors(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))
	for trial := range 100 {
		count := rng.IntN(128) + 1
		values := make([]uint32, count)
		for i := range values {
			values[i] = rng.Uint32()
		}
		original := slices.Clone(values)
		packed, err := PackUint32(0, nil, nil, values)
		require.NoError(t, err, "trial %d", trial)
		unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
		require.NoError(t, err, "trial %d", trial)
		assert.Equal(t, original, unpacked, "trial %d", trial)
	}
}

func TestGetUint32_WithExceptions_AllPositions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[10] = 0x10000
	values[50] = 0x1000000
	values[100] = 0xFFFFFF
	packed, _ := PackUint32(0, nil, nil, values)

	for pos := range 128 {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err, "pos=%d", pos)
		assert.Equal(t, values[pos], got, "pos=%d", pos)
	}
}

func TestGetUint32_WithBitmapExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	for i := range 20 {
		values[i*6] = 0x10000000 + uint32(i)
	}
	packed, _ := PackUint32(0, nil, nil, values)

	for pos := range 128 {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err, "pos=%d", pos)
		assert.Equal(t, values[pos], got, "pos=%d", pos)
	}
}

func TestPackUnpack_ConsumedMatchesBlockLength(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[10] = 0x10000
	values[50] = 0x1000000
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)

	bl, err := BlockLength(packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), bl, "BlockLength should match packed size")

	_, consumed, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed, "consumed should match packed size")
}

func TestCollectExceptionsDirect_Counts(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[10] = 0x10000
	values[50] = 0x1000000

	var positions [128]byte
	var bitmap [16]byte
	var highBits [128]uint32

	n := collectExceptionsDirect(values, 7, positions[:], bitmap[:], highBits[:])
	assert.Equal(t, 2, n)
	assert.Equal(t, byte(10), positions[0])
	assert.Equal(t, byte(50), positions[1])
	assert.Equal(t, uint32(0x10000>>7), highBits[0])
	assert.Equal(t, uint32(0x1000000>>7), highBits[1])
}

func TestFindExceptionIndex_SortedPositions(t *testing.T) {
	buf := make([]byte, 20)
	buf[0] = 5
	buf[1] = 10
	buf[2] = 42
	excStart := 0
	excCount := 3

	assert.Equal(t, 0, findExceptionIndex(buf, excStart, excCount, 5))
	assert.Equal(t, 1, findExceptionIndex(buf, excStart, excCount, 10))
	assert.Equal(t, 2, findExceptionIndex(buf, excStart, excCount, 42))
	assert.Equal(t, -1, findExceptionIndex(buf, excStart, excCount, 0))
	assert.Equal(t, -1, findExceptionIndex(buf, excStart, excCount, 7))
	assert.Equal(t, -1, findExceptionIndex(buf, excStart, excCount, 127))
}

func TestFindExceptionIndex_Bitmap(t *testing.T) {
	buf := make([]byte, 20)
	buf[0] = 0b00100010 // positions 1, 5
	buf[1] = 0b00000001 // position 8
	excStart := 0
	excCount := 20

	assert.Equal(t, 0, findExceptionIndex(buf, excStart, excCount, 1))
	assert.Equal(t, 1, findExceptionIndex(buf, excStart, excCount, 5))
	assert.Equal(t, 2, findExceptionIndex(buf, excStart, excCount, 8))
	assert.Equal(t, -1, findExceptionIndex(buf, excStart, excCount, 0))
	assert.Equal(t, -1, findExceptionIndex(buf, excStart, excCount, 3))
}
