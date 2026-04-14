package utlpfor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Zigzag round-trip tests (3.4) ---

func TestZigzagEncodeDecode_AllValues(t *testing.T) {
	testCases := []int32{0, 1, -1, 127, -128, 32767, -32768, math.MaxInt32, math.MinInt32}
	for _, v := range testCases {
		encoded := zigzagEncode32(v)
		decoded := zigzagDecode32(encoded)
		assert.Equal(t, v, decoded, "zigzag roundtrip for %d", v)
	}
}

// --- Per-lane delta unit tests (3.4) ---

func TestDeltaRoundTrip_PerLane(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 10)
	}
	original := append([]uint32(nil), values...)
	needZZ := deltaEncodePerLaneScalar(values)
	deltaDecodePerLaneScalar(values, needZZ)
	assert.Equal(t, original, values)
}

func TestDeltaRoundTrip_PerLane_Descending(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(10000 - i*50)
	}
	original := append([]uint32(nil), values...)
	needZZ := deltaEncodePerLaneScalar(values)
	assert.True(t, needZZ, "descending data requires zigzag")
	deltaDecodePerLaneScalar(values, needZZ)
	assert.Equal(t, original, values)
}

func TestDeltaRoundTrip_PerLane_AllZeros(t *testing.T) {
	values := make([]uint32, 128)
	original := append([]uint32(nil), values...)
	needZZ := deltaEncodePerLaneScalar(values)
	assert.False(t, needZZ)
	deltaDecodePerLaneScalar(values, needZZ)
	assert.Equal(t, original, values)
}

func TestDeltaRoundTrip_PerLane_SingleValue(t *testing.T) {
	values := []uint32{42}
	expected := append([]uint32(nil), values...)
	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

// --- Per-lane delta specific tests (3.2) ---

func TestDeltaEncodePerLane_LaneIndependence(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 10)
	}
	original := append([]uint32(nil), values...)
	deltaEncodePerLaneScalar(values)

	assert.Equal(t, original[0], values[0])
	assert.Equal(t, original[1], values[1])
	assert.Equal(t, uint32(160), values[16])
	assert.Equal(t, uint32(160), values[32])
}

func TestDeltaEncodeDecodePerLane_AllPatterns(t *testing.T) {
	patterns := []struct {
		name string
		gen  func() []uint32
	}{
		{"sorted_ascending", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i * 100)
			}
			return v
		}},
		{"constant", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = 42
			}
			return v
		}},
		{"descending", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(128 - i)
			}
			return v
		}},
		{"random", func() []uint32 {
			rng := rand.New(rand.NewPCG(7, 0))
			v := make([]uint32, 128)
			for i := range v {
				v[i] = rng.Uint32()
			}
			return v
		}},
		{"alternating", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				if i%2 == 0 {
					v[i] = 1000
				} else {
					v[i] = 1
				}
			}
			return v
		}},
	}
	for _, p := range patterns {
		t.Run(p.name, func(t *testing.T) {
			values := p.gen()
			expected := append([]uint32(nil), values...)
			packed, err := PackUint32(Delta, nil, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
			require.NoError(t, err)
			assert.Equal(t, expected, unpacked)
		})
	}
}

func TestDeltaRoundTrip_PerLane_AllBlockSizes(t *testing.T) {
	for count := 1; count <= 128; count++ {
		t.Run(fmt.Sprintf("count%d", count), func(t *testing.T) {
			values := make([]uint32, count)
			for i := range values {
				values[i] = uint32(i * 7)
			}
			expected := append([]uint32(nil), values...)
			packed, err := PackUint32(Delta, nil, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
			require.NoError(t, err)
			assert.Equal(t, expected, unpacked)
		})
	}
}

// --- Basic delta round-trip tests (3.2) ---

func TestPackUint32_DeltaFlag(t *testing.T) {
	values := []uint32{10, 20, 30, 40, 50}
	expected := append([]uint32(nil), values...)
	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestPackUint32_DeltaFlag_Sorted128(t *testing.T) {
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
}

func TestPackUint32_DeltaFlag_Unsorted(t *testing.T) {
	values := []uint32{100, 50, 200, 10, 300}
	expected := append([]uint32(nil), values...)
	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestPackUint32_DeltaMutatesInputInPlace(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 10)
	}
	original := append([]uint32(nil), values...)

	_, err := PackUint32(Delta, make([]byte, 0, headerBytes+utlPayloadBytes(32)), nil, values)
	require.NoError(t, err)

	assert.NotEqual(t, original, values, "delta path should encode in place")
	assert.Equal(t, original[0], values[0], "lane base value is preserved")
	assert.Equal(t, uint32(160), values[16], "lane delta should be written in place")
}

func TestPackUint32_DeltaCompressesSortedData(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 10)
	}
	plainPacked, _ := PackUint32(0, nil, nil, values)
	deltaPacked, _ := PackUint32(Delta, nil, nil, values)
	assert.Less(t, len(deltaPacked), len(plainPacked),
		"delta should compress sorted data better than plain")
}

func TestPackUint32_DeltaPath_NoAllocsWithPreallocatedDst(t *testing.T) {
	template := make([]uint32, 128)
	for i := range template {
		template[i] = uint32(i * 10)
	}
	work := make([]uint32, len(template))
	dst := make([]byte, 0, headerBytes+utlPayloadBytes(32))

	allocs := testing.AllocsPerRun(1000, func() {
		copy(work, template)
		_, err := PackUint32(Delta, dst[:0], nil, work)
		require.NoError(t, err)
	})
	assert.Equal(t, 0.0, allocs)
}

// --- Overflow detection test ---

func TestDeltaDecodePerLane_OverflowDetection(t *testing.T) {
	values := make([]uint32, 128)
	values[0] = 0xFFFFFFF0
	values[16] = 0x20

	pos := deltaDecodePerLaneWithOverflowScalar(values, false)
	assert.Equal(t, 16, pos, "overflow should be detected at lane-order index 16")
}

func TestDeltaDecodePerLane_NoOverflowWithZigZag(t *testing.T) {
	values := make([]uint32, 128)
	values[0] = 0xFFFFFFF0
	values[16] = 0x20

	pos := deltaDecodePerLaneWithOverflowScalar(values, true)
	assert.Equal(t, 0, pos, "zigzag mode should not report overflow")
}
