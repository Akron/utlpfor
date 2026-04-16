package utlpfor

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetUint32_AllPositions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i*17 + 3)
	}
	packed, _ := PackUint32(0, values, nil, nil)

	scratch := make([]uint32, ScratchLen)
	for pos := range 128 {
		got, err := GetUint32(pos, packed, scratch)
		require.NoError(t, err)
		assert.Equal(t, values[pos], got, "pos=%d", pos)
	}
}

func TestGetUint32_AllPositions_AllBitWidths(t *testing.T) {
	scratch := make([]uint32, ScratchLen)
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
			packed, _ := PackUint32(0, values, nil, nil)

			for pos := range 128 {
				got, err := GetUint32(pos, packed, scratch)
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
	packed, _ := PackUint32(0, values, nil, nil)

	scratch := make([]uint32, ScratchLen)
	unpacked, _, _ := UnpackUint32(packed, nil, make([]uint32, 128))
	for pos := range unpacked {
		got, err := GetUint32(pos, packed, scratch)
		require.NoError(t, err)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

func TestGetUint32_OutOfRange(t *testing.T) {
	values := make([]uint32, 50)
	packed, _ := PackUint32(0, values, nil, nil)

	_, err := GetUint32(50, packed, nil)
	assert.ErrorIs(t, err, ErrPositionOutOfRange)

	_, err = GetUint32(127, packed, nil)
	assert.ErrorIs(t, err, ErrPositionOutOfRange)
}

func TestGetUint32_NegativePosition(t *testing.T) {
	values := make([]uint32, 128)
	packed, _ := PackUint32(0, values, nil, nil)

	_, err := GetUint32(-1, packed, nil)
	assert.ErrorIs(t, err, ErrPositionOutOfRange)
}

func TestGetUint32_ShortBuffer(t *testing.T) {
	_, err := GetUint32(0, []byte{0x01}, nil)
	assert.Error(t, err)
}

func TestGetUint32_ZeroAllocations(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, _ := PackUint32(0, values, nil, nil)

	scratch := make([]uint32, ScratchLen)
	allocs := testing.AllocsPerRun(100, func() {
		GetUint32(42, packed, scratch)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestGetUint32_AllZeros(t *testing.T) {
	values := make([]uint32, 128)
	packed, _ := PackUint32(0, values, nil, nil)

	scratch := make([]uint32, ScratchLen)
	for pos := range 128 {
		got, err := GetUint32(pos, packed, scratch)
		require.NoError(t, err)
		assert.Equal(t, uint32(0), got, "pos=%d", pos)
	}
}

func TestGetUint32_AllMax(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 0xFFFFFFFF
	}
	packed, _ := PackUint32(0, values, nil, nil)

	scratch := make([]uint32, ScratchLen)
	for pos := range 128 {
		got, err := GetUint32(pos, packed, scratch)
		require.NoError(t, err)
		assert.Equal(t, uint32(0xFFFFFFFF), got, "pos=%d", pos)
	}
}

func TestGetUint32_ZeroAllocs(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, err := PackUint32(0, values, nil, nil)
	require.NoError(t, err)

	scratch := make([]uint32, ScratchLen)
	allocs := testing.AllocsPerRun(100, func() {
		GetUint32(42, packed, scratch)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestGetUint32_ZeroAllocs_WithExceptions(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i % 10)
	}
	values[50] = 0xFFFF0000
	values[100] = 0x00FF0000
	packed, err := PackUint32(0, values, nil, nil)
	require.NoError(t, err)

	scratch := make([]uint32, ScratchLen)
	allocs := testing.AllocsPerRun(100, func() {
		GetUint32(50, packed, scratch)
	})
	assert.Equal(t, float64(0), allocs, "exception position")

	allocs = testing.AllocsPerRun(100, func() {
		GetUint32(25, packed, scratch)
	})
	assert.Equal(t, float64(0), allocs, "non-exception position")
}

func TestGetUint32_ZeroAllocs_WithDelta(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 1000 + uint32(i*3)
	}
	packed, err := PackUint32(Delta, values, nil, nil)
	require.NoError(t, err)

	scratch := make([]uint32, ScratchLen)
	allocs := testing.AllocsPerRun(100, func() {
		GetUint32(120, packed, scratch)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestGetUint32_ZeroAllocs_WithDeltaAndExceptions(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 1000 + uint32(i*3)
	}
	values[64] = 0xFFFF0000
	packed, err := PackUint32(Delta, values, nil, nil)
	require.NoError(t, err)

	scratch := make([]uint32, ScratchLen)
	allocs := testing.AllocsPerRun(100, func() {
		GetUint32(64, packed, scratch)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestGetUint32_ZeroAllocs_WithFOR(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 1000000 + uint32(i*3)
	}
	packed, err := PackUint32(Delta, values, nil, nil)
	require.NoError(t, err)

	scratch := make([]uint32, ScratchLen)
	allocs := testing.AllocsPerRun(100, func() {
		GetUint32(120, packed, scratch)
	})
	assert.Equal(t, float64(0), allocs)
}

func TestGetUint32_NilScratch_StillCorrect(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = 1000 + uint32(i*3)
	}
	packed, err := PackUint32(Delta, values, nil, nil)
	require.NoError(t, err)

	unpacked, _, _ := UnpackUint32(packed, nil, make([]uint32, ScratchLen))
	for pos := range len(unpacked) {
		got, err := GetUint32(pos, packed, nil)
		require.NoError(t, err)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

func TestGetUint32_Optimized_AllConfigs(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))
	configs := []struct {
		name string
		flag Flag
		gen  func(*rand.Rand) []uint32
	}{
		{"plain_no_exc", 0, func(r *rand.Rand) []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(r.IntN(200))
			}
			return v
		}},
		{"plain_few_exc", 0, func(r *rand.Rand) []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(r.IntN(50))
			}
			for i := range 5 {
				v[(i*23)%128] = uint32(r.IntN(0x1000000))
			}
			return v
		}},
		{"plain_many_exc", 0, func(r *rand.Rand) []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(r.IntN(50))
			}
			for i := range 48 {
				v[(i*11)%128] = uint32(r.IntN(0x1000000))
			}
			return v
		}},
		{"delta_no_exc", Delta, func(r *rand.Rand) []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = 1000 + uint32(i*3) + uint32(r.IntN(10))
			}
			return v
		}},
		{"delta_few_exc", Delta, func(r *rand.Rand) []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = 1000 + uint32(i*3) + uint32(r.IntN(10))
			}
			v[64] = 0xFFFF0000
			return v
		}},
		{"delta_for", Delta, func(r *rand.Rand) []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = 1000000 + uint32(i*3) + uint32(r.IntN(10))
			}
			return v
		}},
		{"delta_for_exc", Delta, func(r *rand.Rand) []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = 1000000 + uint32(i*3) + uint32(r.IntN(10))
			}
			v[64] = 0xFFFFF000
			return v
		}},
	}

	scratch := make([]uint32, ScratchLen)
	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			for trial := range 50 {
				values := cfg.gen(rng)
				packed, err := PackUint32(cfg.flag, values, nil, nil)
				require.NoError(t, err, "trial %d", trial)

				unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, ScratchLen))
				require.NoError(t, err, "trial %d", trial)

				for pos := range len(unpacked) {
					got, err := GetUint32(pos, packed, scratch)
					require.NoError(t, err, "trial %d pos %d", trial, pos)
					assert.Equal(t, unpacked[pos], got,
						"trial %d pos %d (scratch)", trial, pos)
				}

				for pos := range len(unpacked) {
					got, err := GetUint32(pos, packed, nil)
					require.NoError(t, err, "trial %d pos %d", trial, pos)
					assert.Equal(t, unpacked[pos], got,
						"trial %d pos %d (nil scratch)", trial, pos)
				}
			}
		})
	}
}

func BenchmarkGetUint32_Approaches(b *testing.B) {
	configs := []struct {
		name  string
		flag  Flag
		genFn func([]uint32)
	}{
		{"plain", 0, func(v []uint32) {
			for i := range v {
				v[i] = uint32(i % 200)
			}
		}},
		{"plain_few_exc", 0, func(v []uint32) {
			for i := range v {
				v[i] = uint32(i % 10)
			}
			v[7] = 0xFFFF0000
			v[50] = 0xFFFF0000
			v[100] = 0x00FF0000
		}},
		{"plain_many_exc", 0, func(v []uint32) {
			for i := range v {
				v[i] = uint32(i % 10)
			}
			for j := range 32 {
				v[(j*4)%128] = 0x00FF0000 + uint32(j)
			}
		}},
		{"delta", Delta, func(v []uint32) {
			for i := range v {
				v[i] = 1000 + uint32(i*3)
			}
		}},
		{"delta_few_exc", Delta, func(v []uint32) {
			for i := range v {
				v[i] = 1000 + uint32(i*3)
			}
			v[12] = 0xFFFF0000
			v[64] = 0xFFFF0000
		}},
		{"delta_many_exc", Delta, func(v []uint32) {
			for i := range v {
				v[i] = 1000 + uint32(i*3)
			}
			for j := range 32 {
				v[(j*4)%128] = 0xFFFF0000 + uint32(j)
			}
		}},
		{"delta_for", Delta, func(v []uint32) {
			for i := range v {
				v[i] = 1000000 + uint32(i*3)
			}
		}},
	}

	type posRange struct {
		name     string
		lo, hi   int
		probePos int
	}
	ranges := []posRange{
		{"pos_00_15", 0, 15, 8},
		{"pos_16_31", 16, 31, 24},
		{"pos_32_63", 32, 63, 48},
		{"pos_64_95", 64, 95, 80},
		{"pos_96_127", 96, 127, 112},
	}

	for _, cfg := range configs {
		values := make([]uint32, 128)
		cfg.genFn(values)
		packed, _ := PackUint32(cfg.flag, values, nil, nil)

		scratch := make([]uint32, ScratchLen)
		for _, pr := range ranges {
			b.Run(fmt.Sprintf("%s/%s/single", cfg.name, pr.name), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					GetUint32(pr.probePos, packed, scratch)
				}
			})
			b.Run(fmt.Sprintf("%s/%s/sweep", cfg.name, pr.name), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					for pos := pr.lo; pos <= pr.hi; pos++ {
						GetUint32(pos, packed, scratch)
					}
				}
			})
		}
	}
}
