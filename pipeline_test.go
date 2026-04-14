package utlpfor

import (
	"encoding/binary"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFullPipeline_DeltaPlusExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	values[50] = 0xFFFFFF
	expected := append([]uint32(nil), values...)

	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestFullPipeline_BlockLengthConsistency(t *testing.T) {
	rng := rand.New(rand.NewPCG(99, 0))
	for trial := range 100 {
		count := rng.IntN(128) + 1
		values := make([]uint32, count)
		for i := range values {
			values[i] = rng.Uint32()
		}

		for _, flag := range []byte{0, Delta} {
			work := append([]uint32(nil), values...)
			packed, err := PackUint32(flag, nil, nil, work)
			require.NoError(t, err)

			blockLen, err := BlockLength(packed)
			require.NoError(t, err)
			assert.Equal(t, len(packed), blockLen,
				"trial %d flag %d: BlockLength mismatch", trial, flag)
		}
	}
}

func TestFullPipeline_GetMatchesUnpack(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 7)
	}
	values[10] = 0x100000
	values[100] = 0xFFFFFF

	for _, flag := range []byte{0, Delta} {
		work := append([]uint32(nil), values...)
		packed, err := PackUint32(flag, nil, nil, work)
		require.NoError(t, err)
		unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
		require.NoError(t, err)

		for pos := range unpacked {
			got, err := GetUint32(pos, packed)
			require.NoError(t, err)
			assert.Equal(t, unpacked[pos], got, "flag=%d pos=%d", flag, pos)
		}
	}
}

func TestFullPipeline_PlainNoExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)

	header := binary.LittleEndian.Uint32(packed)
	excCount := int((header >> 24) & 0xFF)
	assert.Equal(t, 0, excCount, "small values should have no exceptions")
}

func TestFullPipeline_DeltaDescendingWithExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(10000 - i*50)
	}
	values[64] = 0xFFFFFFF
	expected := append([]uint32(nil), values...)

	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestFullPipeline_CompressionRatio(t *testing.T) {
	datasets := []struct {
		name   string
		values []uint32
		flag   byte
	}{
		{"small_plain", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i % 16)
			}
			return v
		}(), 0},
		{"sorted_delta", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i * 100)
			}
			return v
		}(), Delta},
		{"byte_range", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i)
			}
			return v
		}(), 0},
	}
	for _, ds := range datasets {
		t.Run(ds.name, func(t *testing.T) {
			work := append([]uint32(nil), ds.values...)
			packed, err := PackUint32(ds.flag, nil, nil, work)
			require.NoError(t, err)
			rawSize := len(ds.values) * 4
			assert.Less(t, len(packed), rawSize,
				"packed %d vs raw %d", len(packed), rawSize)
		})
	}
}

func TestFullPipeline_GetMatchesUnpack_Delta(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	expected := append([]uint32(nil), values...)
	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)

	for pos := range unpacked {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

func TestFullPipeline_GetMatchesUnpack_DeltaWithExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 7)
	}
	values[10] = 0x100000
	values[64] = 0xFFFFFFF
	values[100] = 0xFFFFFF
	expected := append([]uint32(nil), values...)

	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)

	for pos := range unpacked {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

func TestFullPipeline_GetMatchesUnpack_DeltaDescending(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(10000 - i*50)
	}
	expected := append([]uint32(nil), values...)

	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)

	for pos := range unpacked {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

func TestFullPipeline_GetMatchesUnpack_RandomDelta(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))
	for trial := range 50 {
		count := rng.IntN(128) + 1
		values := make([]uint32, count)
		for i := range values {
			values[i] = rng.Uint32()
		}
		expected := append([]uint32(nil), values...)

		packed, err := PackUint32(Delta, nil, nil, values)
		require.NoError(t, err, "trial %d", trial)

		unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
		require.NoError(t, err, "trial %d", trial)
		assert.Equal(t, expected, unpacked, "trial %d: unpack mismatch", trial)

		for pos := range unpacked {
			got, err := GetUint32(pos, packed)
			require.NoError(t, err, "trial %d pos %d", trial, pos)
			assert.Equal(t, unpacked[pos], got, "trial %d pos %d", trial, pos)
		}
	}
}

func TestFullPipeline_IntTypeValidation(t *testing.T) {
	h := encodeHeader(128, 8, 0, uint32(IntTypeUint64)<<headerTypeShift)
	buf := make([]byte, 4+utlPayloadBytes(8))
	bo.PutUint32(buf, h)

	_, _, err := UnpackUint32(nil, make([]uint32, 128), buf)
	assert.ErrorIs(t, err, ErrUnsupportedType)

	_, err = GetUint32(0, buf)
	assert.ErrorIs(t, err, ErrUnsupportedType)

	h = encodeHeader(128, 8, 0, uint32(IntTypeUint8)<<headerTypeShift)
	bo.PutUint32(buf, h)

	_, _, err = UnpackUint32(nil, make([]uint32, 128), buf)
	assert.ErrorIs(t, err, ErrUnsupportedType)

	_, err = GetUint32(0, buf)
	assert.ErrorIs(t, err, ErrUnsupportedType)
}
