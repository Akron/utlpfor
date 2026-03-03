package utlpfor

import (
	"fmt"
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
			packed, err := PackUint32(0, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
			require.NoError(t, err)
			assert.Equal(t, values, unpacked)
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
			packed, err := PackUint32(0, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
			require.NoError(t, err)
			assert.Equal(t, values, unpacked)
		})
	}
}

func TestPackUint32_AllZeros(t *testing.T) {
	values := make([]uint32, 128)
	packed, err := PackUint32(0, nil, values)
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
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestPackUint32_DstGrowth(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	dst := make([]byte, 0, 10)
	packed, err := PackUint32(0, dst, values)
	require.NoError(t, err)
	assert.NotNil(t, packed)
}

func TestPackUint32_DeterministicOutput(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed1, _ := PackUint32(0, nil, values)
	packed2, _ := PackUint32(0, nil, values)
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
			packed, err := PackUint32(0, nil, tt.values)
			require.NoError(t, err)
			rawSize := len(tt.values) * 4
			assert.Less(t, len(packed), rawSize,
				"packed size %d should be less than raw size %d", len(packed), rawSize)
		})
	}
}

func TestPackUint32_EmptyValues(t *testing.T) {
	_, err := PackUint32(0, nil, []uint32{})
	assert.Error(t, err)
}

func TestPackUint32_TooManyValues(t *testing.T) {
	_, err := PackUint32(0, nil, make([]uint32, 129))
	assert.Error(t, err)
}

func TestPackUint32_DstReuse(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	dst := make([]byte, 1024)
	packed, err := PackUint32(0, dst, values)
	require.NoError(t, err)
	assert.True(t, cap(packed) >= cap(dst), "should reuse provided dst buffer")
}
