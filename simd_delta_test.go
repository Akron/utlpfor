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
	for i := 0; i < n; i++ {
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
	src := make([]uint32, blockSize)
	for i := range src {
		src[i] = uint32(i * 5)
	}
	zigzagEncodeSlice(src, len(src))

	scalarDst := make([]uint32, blockSize)
	zigzagDecodeSlice(scalarDst, src)

	if simdLevel >= simdLevelAVX2 {
		simdDst := make([]uint32, blockSize)
		zigzagDecodeAVX2(simdDst, src)
		assert.Equal(t, scalarDst, simdDst, "AVX2 zigzag decode mismatch")
	}

	if simdLevel >= simdLevelSSE2 {
		simdDst := make([]uint32, blockSize)
		zigzagDecodeSSE2(simdDst, src)
		assert.Equal(t, scalarDst, simdDst, "SSE2 zigzag decode mismatch")
	}
}

func TestZigzagRoundTrip_SIMD(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 100; trial++ {
		original := make([]uint32, blockSize)
		for i := range original {
			original[i] = rng.Uint32()
		}

		if simdLevel >= simdLevelAVX2 {
			buf := make([]uint32, blockSize)
			copy(buf, original)
			zigzagEncodeAVX2(buf, len(buf))

			decoded := make([]uint32, blockSize)
			zigzagDecodeAVX2(decoded, buf)
			assert.Equal(t, original, decoded, "AVX2 trial=%d", trial)
		}

		if simdLevel >= simdLevelSSE2 {
			buf := make([]uint32, blockSize)
			copy(buf, original)
			zigzagEncodeSSE2(buf, len(buf))

			decoded := make([]uint32, blockSize)
			zigzagDecodeSSE2(decoded, buf)
			assert.Equal(t, original, decoded, "SSE2 trial=%d", trial)
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
			scalarDst := make([]uint32, blockSize)
			scalarZZ := deltaEncodePerLaneScalar(scalarDst, values)

			if simdLevel >= simdLevelAVX2 {
				simdDst := make([]uint32, blockSize)
				simdZZ := deltaEncodePerLaneAVX2(simdDst, values)
				assert.Equal(t, scalarZZ, simdZZ, "AVX2 needZigZag mismatch")
				assert.Equal(t, scalarDst, simdDst, "AVX2 delta encode mismatch")
			}

			if simdLevel >= simdLevelSSE2 {
				simdDst := make([]uint32, blockSize)
				simdZZ := deltaEncodePerLaneSSE2(simdDst, values)
				assert.Equal(t, scalarZZ, simdZZ, "SSE2 needZigZag mismatch")
				assert.Equal(t, scalarDst, simdDst, "SSE2 delta encode mismatch")
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
			deltas := make([]uint32, blockSize)
			useZZ := deltaEncodePerLaneScalar(deltas, values)

			scalarResult := make([]uint32, blockSize)
			deltaDecodePerLaneScalar(scalarResult, deltas, useZZ)

			if simdLevel >= simdLevelAVX2 {
				simdResult := make([]uint32, blockSize)
				deltaDecodePerLaneAVX2(simdResult, deltas, useZZ)
				assert.Equal(t, scalarResult, simdResult, "AVX2 decode mismatch")
			}

			if simdLevel >= simdLevelSSE2 {
				simdResult := make([]uint32, blockSize)
				deltaDecodePerLaneSSE2(simdResult, deltas, useZZ)
				assert.Equal(t, scalarResult, simdResult, "SSE2 decode mismatch")
			}
		})
	}
}

func TestDeltaDecodeOverflow_SIMDMatchesScalar(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFF0
	}
	values[0] = 100

	deltas := make([]uint32, blockSize)
	deltaEncodePerLaneScalar(deltas, values)

	scalarPos := deltaDecodePerLaneWithOverflowScalar(
		make([]uint32, blockSize), deltas, false)

	if simdLevel >= simdLevelAVX2 {
		simdPos := deltaDecodePerLaneWithOverflowAVX2(
			make([]uint32, blockSize), deltas, false)
		assert.Equal(t, scalarPos, simdPos, "AVX2 overflow position mismatch")
	}

	if simdLevel >= simdLevelSSE2 {
		simdPos := deltaDecodePerLaneWithOverflowSSE2(
			make([]uint32, blockSize), deltas, false)
		assert.Equal(t, scalarPos, simdPos, "SSE2 overflow position mismatch")
	}
}

func TestDeltaDecodeOverflow_NoOverflowWithZigZag_SIMD(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 0xFFFFFFF0
	}
	values[0] = 100

	deltas := make([]uint32, blockSize)
	deltaEncodePerLaneScalar(deltas, values)

	if simdLevel >= simdLevelAVX2 {
		pos := deltaDecodePerLaneWithOverflowAVX2(
			make([]uint32, blockSize), deltas, true)
		assert.Equal(t, 0, pos, "AVX2: zigzag mode should not report overflow")
	}

	if simdLevel >= simdLevelSSE2 {
		pos := deltaDecodePerLaneWithOverflowSSE2(
			make([]uint32, blockSize), deltas, true)
		assert.Equal(t, 0, pos, "SSE2: zigzag mode should not report overflow")
	}
}

func TestDeltaDecodeOverflow_NoFalsePositive_SIMD(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i * 10)
	}

	deltas := make([]uint32, blockSize)
	deltaEncodePerLaneScalar(deltas, values)

	scalarPos := deltaDecodePerLaneWithOverflowScalar(
		make([]uint32, blockSize), deltas, false)
	assert.Equal(t, 0, scalarPos, "no overflow expected in sorted data")

	if simdLevel >= simdLevelAVX2 {
		simdPos := deltaDecodePerLaneWithOverflowAVX2(
			make([]uint32, blockSize), deltas, false)
		assert.Equal(t, 0, simdPos, "AVX2: no false positive overflow")
	}

	if simdLevel >= simdLevelSSE2 {
		simdPos := deltaDecodePerLaneWithOverflowSSE2(
			make([]uint32, blockSize), deltas, false)
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
			packed, err := PackUint32(Delta, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
			require.NoError(t, err)
			assert.Equal(t, expected, unpacked)
		})
	}
}

func TestDeltaRoundTrip_SIMD_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	for seed := int64(0); seed < 500; seed++ {
		rng.Seed(seed)
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = rng.Uint32()
		}
		expected := append([]uint32(nil), values...)

		packed, err := PackUint32(Delta, nil, values)
		require.NoError(t, err, "seed=%d", seed)

		unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
		require.NoError(t, err, "seed=%d", seed)
		assert.Equal(t, expected, unpacked, "seed=%d", seed)
	}
}

func TestDeltaEncodeDecodeRoundTrip_SIMD(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for trial := 0; trial < 100; trial++ {
		original := make([]uint32, blockSize)
		for i := range original {
			original[i] = rng.Uint32()
		}

		if simdLevel >= simdLevelAVX2 {
			deltas := make([]uint32, blockSize)
			useZZ := deltaEncodePerLaneAVX2(deltas, original)
			decoded := make([]uint32, blockSize)
			deltaDecodePerLaneAVX2(decoded, deltas, useZZ)
			assert.Equal(t, original, decoded, "AVX2 round-trip trial=%d", trial)
		}

		if simdLevel >= simdLevelSSE2 {
			deltas := make([]uint32, blockSize)
			useZZ := deltaEncodePerLaneSSE2(deltas, original)
			decoded := make([]uint32, blockSize)
			deltaDecodePerLaneSSE2(decoded, deltas, useZZ)
			assert.Equal(t, original, decoded, "SSE2 round-trip trial=%d", trial)
		}
	}
}
