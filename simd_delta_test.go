//go:build goexperiment.simd && amd64

package utlpfor

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// zigzagEncodeSlice applies zigzag encoding to all values (scalar reference).
func zigzagEncodeSlice(buf []uint32, n int) {
	for i := range n {
		buf[i] = zigzagEncode32(int32(buf[i]))
	}
}

// zigzagDecodeSlice applies zigzag decoding to all values (scalar reference).
func zigzagDecodeSlice(dst, src []uint32) {
	for i := range src {
		dst[i] = uint32(zigzagDecode32(src[i]))
	}
}

func TestZigzagEncode_SIMDMatchesScalar(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i * 3)
	}

	scalarBuf := make([]uint32, blockSize)
	copy(scalarBuf, values)
	zigzagEncodeSlice(scalarBuf, len(scalarBuf))

	if simdLevel >= simdLevelAVX2 {
		simdBuf := make([]uint32, blockSize)
		copy(simdBuf, values)
		zigzagEncodeAVX2(simdBuf, len(simdBuf))
		assert.Equal(t, scalarBuf, simdBuf, "AVX2 zigzag encode mismatch")
	}

	if simdLevel >= simdLevelSSE2 {
		simdBuf := make([]uint32, blockSize)
		copy(simdBuf, values)
		zigzagEncodeSSE2(simdBuf, len(simdBuf))
		assert.Equal(t, scalarBuf, simdBuf, "SSE2 zigzag encode mismatch")
	}
}

func TestZigzagEncode_SIMDMatchesScalar_EdgeValues(t *testing.T) {
	values := []uint32{0, 1, math.MaxUint32, uint32(math.MaxInt32), 0x80000000}
	padded := make([]uint32, blockSize)
	copy(padded, values)

	scalarBuf := make([]uint32, blockSize)
	copy(scalarBuf, padded)
	zigzagEncodeSlice(scalarBuf, len(scalarBuf))

	if simdLevel >= simdLevelAVX2 {
		simdBuf := make([]uint32, blockSize)
		copy(simdBuf, padded)
		zigzagEncodeAVX2(simdBuf, len(simdBuf))
		assert.Equal(t, scalarBuf, simdBuf, "AVX2 zigzag encode edge mismatch")
	}

	if simdLevel >= simdLevelSSE2 {
		simdBuf := make([]uint32, blockSize)
		copy(simdBuf, padded)
		zigzagEncodeSSE2(simdBuf, len(simdBuf))
		assert.Equal(t, scalarBuf, simdBuf, "SSE2 zigzag encode edge mismatch")
	}
}

func TestZigzagDecode_SIMDMatchesScalar(t *testing.T) {
	encoded := make([]uint32, blockSize)
	for i := range encoded {
		encoded[i] = uint32(i * 5)
	}
	zigzagEncodeSlice(encoded, len(encoded))

	scalarDst := make([]uint32, blockSize)
	zigzagDecodeSlice(scalarDst, encoded)

	if simdLevel >= simdLevelAVX2 {
		simdBuf := make([]uint32, blockSize)
		copy(simdBuf, encoded)
		zigzagDecodeAVX2(simdBuf)
		assert.Equal(t, scalarDst, simdBuf, "AVX2 zigzag decode mismatch")
	}

	if simdLevel >= simdLevelSSE2 {
		simdBuf := make([]uint32, blockSize)
		copy(simdBuf, encoded)
		zigzagDecodeSSE2(simdBuf)
		assert.Equal(t, scalarDst, simdBuf, "SSE2 zigzag decode mismatch")
	}
}

func TestZigzagRoundTrip_SIMD(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := range 100 {
		original := make([]uint32, blockSize)
		for i := range original {
			original[i] = rng.Uint32()
		}

		if simdLevel >= simdLevelAVX2 {
			buf := make([]uint32, blockSize)
			copy(buf, original)
			zigzagEncodeAVX2(buf, len(buf))
			zigzagDecodeAVX2(buf)
			assert.Equal(t, original, buf, "AVX2 trial=%d", trial)
		}

		if simdLevel >= simdLevelSSE2 {
			buf := make([]uint32, blockSize)
			copy(buf, original)
			zigzagEncodeSSE2(buf, len(buf))
			zigzagDecodeSSE2(buf)
			assert.Equal(t, original, buf, "SSE2 trial=%d", trial)
		}
	}
}

func TestDeltaEncode_SIMDMatchesScalar(t *testing.T) {
	for _, tc := range []struct {
		name string
		gen  func() []uint32
	}{
		{"sorted", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = uint32(i * 10)
			}
			return v
		}},
		{"constant", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = 42
			}
			return v
		}},
		{"descending", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = uint32(blockSize - i)
			}
			return v
		}},
		{"random", func() []uint32 {
			rng := rand.New(rand.NewSource(7))
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = rng.Uint32()
			}
			return v
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := tc.gen()
			scalarBuf := make([]uint32, blockSize)
			copy(scalarBuf, values)
			scalarZZ := deltaEncodePerLaneScalar(scalarBuf)

			if simdLevel >= simdLevelAVX2 {
				simdBuf := make([]uint32, blockSize)
				copy(simdBuf, values)
				simdZZ := deltaEncodePerLaneAVX2(simdBuf)
				assert.Equal(t, scalarZZ, simdZZ, "AVX2 needZigZag mismatch")
				assert.Equal(t, scalarBuf, simdBuf, "AVX2 delta encode mismatch")
			}

			if simdLevel >= simdLevelSSE2 {
				simdBuf := make([]uint32, blockSize)
				copy(simdBuf, values)
				simdZZ := deltaEncodePerLaneSSE2(simdBuf)
				assert.Equal(t, scalarZZ, simdZZ, "SSE2 needZigZag mismatch")
				assert.Equal(t, scalarBuf, simdBuf, "SSE2 delta encode mismatch")
			}
		})
	}
}

func TestDeltaDecode_SIMDMatchesScalar(t *testing.T) {
	for _, tc := range []struct {
		name string
		gen  func() []uint32
	}{
		{"sorted", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = uint32(i * 10)
			}
			return v
		}},
		{"constant", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = 42
			}
			return v
		}},
		{"descending", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = uint32(blockSize - i)
			}
			return v
		}},
		{"random", func() []uint32 {
			rng := rand.New(rand.NewSource(7))
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = rng.Uint32()
			}
			return v
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := tc.gen()
			scalarBuf := make([]uint32, blockSize)
			copy(scalarBuf, values)
			useZZ := deltaEncodePerLaneScalar(scalarBuf)
			deltaDecodePerLaneScalar(scalarBuf, useZZ)

			if simdLevel >= simdLevelAVX2 {
				simdBuf := make([]uint32, blockSize)
				copy(simdBuf, values)
				deltaEncodePerLaneScalar(simdBuf)
				deltaDecodePerLaneAVX2(simdBuf, useZZ)
				assert.Equal(t, scalarBuf, simdBuf, "AVX2 decode mismatch")
			}

			if simdLevel >= simdLevelSSE2 {
				simdBuf := make([]uint32, blockSize)
				copy(simdBuf, values)
				deltaEncodePerLaneScalar(simdBuf)
				deltaDecodePerLaneSSE2(simdBuf, useZZ)
				assert.Equal(t, scalarBuf, simdBuf, "SSE2 decode mismatch")
			}
		})
	}
}

func TestDeltaDecodeOverflow_SIMDMatchesScalar(t *testing.T) {
	makeDeltas := func() []uint32 {
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = 0xFFFFFFF0
		}
		values[0] = 100
		deltaEncodePerLaneScalar(values)
		return values
	}

	scalarBuf := makeDeltas()
	scalarPos := deltaDecodePerLaneWithOverflowScalar(scalarBuf, false)

	if simdLevel >= simdLevelAVX2 {
		simdBuf := makeDeltas()
		simdPos := deltaDecodePerLaneWithOverflowAVX2(simdBuf, false)
		assert.Equal(t, scalarPos, simdPos, "AVX2 overflow position mismatch")
	}

	if simdLevel >= simdLevelSSE2 {
		simdBuf := makeDeltas()
		simdPos := deltaDecodePerLaneWithOverflowSSE2(simdBuf, false)
		assert.Equal(t, scalarPos, simdPos, "SSE2 overflow position mismatch")
	}
}

func TestDeltaDecodeOverflow_NoOverflowWithZigZag_SIMD(t *testing.T) {
	makeDeltas := func() []uint32 {
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = 0xFFFFFFF0
		}
		values[0] = 100
		deltaEncodePerLaneScalar(values)
		return values
	}

	if simdLevel >= simdLevelAVX2 {
		buf := makeDeltas()
		pos := deltaDecodePerLaneWithOverflowAVX2(buf, true)
		assert.Equal(t, 0, pos, "AVX2: zigzag mode should not report overflow")
	}

	if simdLevel >= simdLevelSSE2 {
		buf := makeDeltas()
		pos := deltaDecodePerLaneWithOverflowSSE2(buf, true)
		assert.Equal(t, 0, pos, "SSE2: zigzag mode should not report overflow")
	}
}

func TestDeltaDecodeOverflow_NoFalsePositive_SIMD(t *testing.T) {
	makeDeltas := func() []uint32 {
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = uint32(i * 10)
		}
		deltaEncodePerLaneScalar(values)
		return values
	}

	scalarBuf := makeDeltas()
	scalarPos := deltaDecodePerLaneWithOverflowScalar(scalarBuf, false)
	assert.Equal(t, 0, scalarPos, "no overflow expected in sorted data")

	if simdLevel >= simdLevelAVX2 {
		simdBuf := makeDeltas()
		simdPos := deltaDecodePerLaneWithOverflowAVX2(simdBuf, false)
		assert.Equal(t, 0, simdPos, "AVX2: no false positive overflow")
	}

	if simdLevel >= simdLevelSSE2 {
		simdBuf := makeDeltas()
		simdPos := deltaDecodePerLaneWithOverflowSSE2(simdBuf, false)
		assert.Equal(t, 0, simdPos, "SSE2: no false positive overflow")
	}
}

func TestDeltaRoundTrip_SIMD_AllPatterns(t *testing.T) {
	patterns := []struct {
		name string
		gen  func() []uint32
	}{
		{"sorted", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = uint32(i * 100)
			}
			return v
		}},
		{"constant", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = 42
			}
			return v
		}},
		{"descending", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = uint32(blockSize - i)
			}
			return v
		}},
		{"random", func() []uint32 {
			rng := rand.New(rand.NewSource(7))
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = rng.Uint32()
			}
			return v
		}},
	}
	for _, p := range patterns {
		t.Run(p.name, func(t *testing.T) {
			values := p.gen()
			expected := append([]uint32(nil), values...)
			packed, err := PackUint32(Delta, values, nil, nil)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
			require.NoError(t, err)
			assert.Equal(t, expected, unpacked)
		})
	}
}

func TestDeltaRoundTrip_SIMD_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	for seed := range int64(500) {
		rng.Seed(seed)
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = rng.Uint32()
		}
		expected := append([]uint32(nil), values...)

		packed, err := PackUint32(Delta, values, nil, nil)
		require.NoError(t, err, "seed=%d", seed)

		unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
		require.NoError(t, err, "seed=%d", seed)
		assert.Equal(t, expected, unpacked, "seed=%d", seed)
	}
}

func TestDeltaEncodeDecodeRoundTrip_SIMD(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for trial := range 100 {
		original := make([]uint32, blockSize)
		for i := range original {
			original[i] = rng.Uint32()
		}

		if simdLevel >= simdLevelAVX2 {
			buf := make([]uint32, blockSize)
			copy(buf, original)
			useZZ := deltaEncodePerLaneAVX2(buf)
			deltaDecodePerLaneAVX2(buf, useZZ)
			assert.Equal(t, original, buf, "AVX2 round-trip trial=%d", trial)
		}

		if simdLevel >= simdLevelSSE2 {
			buf := make([]uint32, blockSize)
			copy(buf, original)
			useZZ := deltaEncodePerLaneSSE2(buf)
			deltaDecodePerLaneSSE2(buf, useZZ)
			assert.Equal(t, original, buf, "SSE2 round-trip trial=%d", trial)
		}
	}
}
