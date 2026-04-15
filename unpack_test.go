package utlpfor

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnpackUint32_ReturnsConsumedLength(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, _ := PackUint32(0, nil, nil, values)

	_, consumed, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
}

func TestUnpackUint32_ScratchBufferReuse(t *testing.T) {
	scratch := make([]uint32, 128)
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 3)
	}
	packed, _ := PackUint32(0, nil, nil, values)

	unpacked1, _, _ := UnpackUint32(nil, scratch, packed)
	unpacked2, _, _ := UnpackUint32(nil, scratch, packed)
	assert.Equal(t, unpacked1, unpacked2)
}

func TestUnpackUint32_MalformedInput(t *testing.T) {
	tests := []struct {
		name string
		buf  []byte
	}{
		{"nil", nil},
		{"empty", []byte{}},
		{"short_header", []byte{0x01, 0x02}},
		{"truncated_payload", func() []byte {
			buf := make([]byte, 4)
			bo.PutUint32(buf,
				encodeHeader(128, 8, 0, headerTypeUint32Flag))
			return buf
		}()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := UnpackUint32(nil, make([]uint32, 128), tt.buf)
			assert.Error(t, err)
		})
	}
}

func TestUnpackUint32_ZeroAllocations(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, _ := PackUint32(0, nil, nil, values)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)

	allocs := testing.AllocsPerRun(100, func() {
		UnpackUint32(dst, scratch, packed)
	})
	assert.Equal(t, float64(0), allocs, "expected zero allocations")
}

func TestUnpackUint32_UnsupportedType(t *testing.T) {
	h := encodeHeader(128, 8, 0, uint32(IntTypeUint64)<<headerTypeShift)
	buf := make([]byte, 4+128)
	bo.PutUint32(buf, h)
	_, _, err := UnpackUint32(nil, make([]uint32, 128), buf)
	assert.ErrorIs(t, err, ErrUnsupportedType)
}

func TestUnpackUint32_Uint16TypeAccepted(t *testing.T) {
	h := encodeHeader(128, 8, 0, headerTypeUint16Flag)
	payloadBytes := utlPayloadBytes(8)
	buf := make([]byte, headerBytes+payloadBytes)
	bo.PutUint32(buf, h)
	_, _, err := UnpackUint32(nil, make([]uint32, 128), buf)
	assert.NoError(t, err)
}

func TestUnpackUint32_Uint8TypeRejected(t *testing.T) {
	h := encodeHeader(128, 8, 0, uint32(IntTypeUint8)<<headerTypeShift)
	buf := make([]byte, 4+128)
	bo.PutUint32(buf, h)
	_, _, err := UnpackUint32(nil, make([]uint32, 128), buf)
	assert.ErrorIs(t, err, ErrUnsupportedType)
}

func TestUnpackUint32_ZeroAllocs(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, err := PackUint32(0, nil, nil, values)
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
	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	dst := make([]uint32, blockSize)
	scratch := make([]uint32, blockSize)

	allocs := testing.AllocsPerRun(100, func() {
		UnpackUint32(dst, scratch, packed)
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
		packed, err := PackUint32(0, nil, nil, values)
		require.NoError(t, err)

		var uErr error
		dst, _, uErr = UnpackUint32(dst, scratch, packed)
		require.NoError(t, uErr)
		assert.Equal(t, original, dst)
	}
}
