//go:build goexperiment.simd && amd64

package utlpfor

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func FuzzSIMDScalarConsistency(f *testing.F) {
	f.Add(encodeValuesSeed([]uint32{1, 2, 3, 4}))
	f.Add(encodeValuesSeed([]uint32{0, 0xFF, 0xFFFF, 0xFFFFFFFF}))
	f.Add(encodeValuesSeed(make([]uint32, 128)))

	f.Fuzz(func(t *testing.T, data []byte) {
		values := decodeValuesSeed(data)
		if len(values) == 0 || len(values) > blockSize {
			return
		}

		scratch := make([]uint32, blockSize)
		scalarValues := make([]uint32, len(values))
		copy(scalarValues, values)
		simdValues := make([]uint32, len(values))
		copy(simdValues, values)

		scalarPacked, err := packUint32Scalar(0, nil, scratch, scalarValues)
		if err != nil {
			return
		}
		simdPacked, err := PackUint32(0, simdValues, nil, nil)
		if err != nil {
			return
		}
		require.Equal(t, scalarPacked, simdPacked,
			"SIMD and scalar must produce identical bytes")
	})
}

func TestUint64_SIMDvsScalar_Pack(t *testing.T) {
	if simdLevel == simdLevelScalar {
		t.Skip("no SIMD available")
	}

	testCases := []struct {
		name string
		gen  func() []uint64
	}{
		{"fit32_sequential", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(i * 100)
			}
			return v
		}},
		{"for64_clustered", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0x100000000 + uint64(i)*7
			}
			return v
		}},
		{"for64_boundary", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0xFFFFFFC0 + uint64(i)
			}
			return v
		}},
		{"two_block_random", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(i)*0x100000001 + uint64(i*i)
			}
			return v
		}},
		{"timestamps", func() []uint64 {
			v := make([]uint64, blockSize)
			base := uint64(1_719_300_000_000)
			for i := range v {
				v[i] = base + uint64(i)
			}
			return v
		}},
		{"all_max", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = ^uint64(0)
			}
			return v
		}},
		{"single_value", func() []uint64 {
			return []uint64{0xDEADBEEF12345678}
		}},
		{"partial_block", func() []uint64 {
			v := make([]uint64, 17)
			for i := range v {
				v[i] = 0x200000000 + uint64(i)*13
			}
			return v
		}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			values := tc.gen()
			scratch := make([]uint32, ScratchLen64)

			for _, flag := range []Flag{0, Delta, NoPatch, NoFOR, NoPatch | NoFOR} {
				scalarPacked, err := packUint64Scalar(flag, slices.Clone(values), nil, make([]uint32, ScratchLen64))
				require.NoError(t, err)

				simdPacked, err := PackUint64(flag, slices.Clone(values), nil, scratch)
				require.NoError(t, err)

				assert.Equal(t, scalarPacked, simdPacked,
					"flag=%d: SIMD and scalar pack must produce identical bytes", flag)
			}
		})
	}
}

func TestUint64_SIMDvsScalar_Unpack(t *testing.T) {
	if simdLevel == simdLevelScalar {
		t.Skip("no SIMD available")
	}

	testCases := []struct {
		name string
		gen  func() []uint64
	}{
		{"fit32", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(i * 100)
			}
			return v
		}},
		{"for64", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0x100000000 + uint64(i)*7
			}
			return v
		}},
		{"two_block", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(i)*0x100000001 + uint64(i*i)
			}
			return v
		}},
		{"boundary_crossing_delta", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0xFFFFFFC0 + uint64(i)
			}
			return v
		}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			values := tc.gen()
			scratch := make([]uint32, ScratchLen64)

			for _, flag := range []Flag{0, Delta} {
				packed, err := PackUint64(flag, slices.Clone(values), nil, scratch)
				require.NoError(t, err)

				scalarDst := make([]uint64, blockSize)
				scalarResult, scalarConsumed, err := unpackUint64Scalar(packed, scalarDst, make([]uint32, ScratchLen64))
				require.NoError(t, err)

				simdDst := make([]uint64, blockSize)
				simdResult, simdConsumed, err := UnpackUint64(packed, simdDst, scratch)
				require.NoError(t, err)

				assert.Equal(t, scalarConsumed, simdConsumed,
					"flag=%d: consumed bytes differ", flag)
				assert.Equal(t, scalarResult, simdResult,
					"flag=%d: SIMD and scalar unpack must produce identical values", flag)
			}
		})
	}
}
