package utlpfor

import (
	"fmt"
	"math/bits"
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

func TestGetUint32_DeltaHighBitWidth_DeepPosition(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i) * 1000000
	}
	original := make([]uint32, blockSize)
	copy(original, values)

	packed, err := PackUint32(Delta, values, nil, nil)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
	require.NoError(t, err)

	scratch := make([]uint32, ScratchLen)
	for _, pos := range []int{0, 15, 48, 64, 96, 112, 127} {
		got, err := GetUint32(pos, packed, scratch)
		require.NoError(t, err, "pos=%d", pos)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
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

// extractPackedValueUTLRef is a byte-exact copy of the pre-optimization
// bounds-checked implementation, kept as the equivalence reference.
func extractPackedValueUTLRef(pos int, payload []byte, bitWidth int) uint32 {
	lane := pos % utlLaneCount
	posInLane := pos / utlLaneCount
	bitPos := posInLane * bitWidth
	wordInLane := bitPos / 32
	bitOffset := bitPos % 32

	byteOffset := wordInLane*utlSuperWordBytes + lane*4

	var acc uint64
	if byteOffset+4 <= len(payload) {
		acc = uint64(bo.Uint32(payload[byteOffset:]))
	}
	if bitWidth > 32-bitOffset {
		nextByteOffset := byteOffset + utlSuperWordBytes
		if nextByteOffset+4 <= len(payload) {
			acc |= uint64(bo.Uint32(payload[nextByteOffset:])) << 32
		}
	}

	acc >>= uint(bitOffset)
	var mask uint64
	if bitWidth >= 32 {
		mask = 0xFFFFFFFF
	} else {
		mask = uint64((1 << bitWidth) - 1)
	}
	return uint32(acc & mask)
}

// bitmapRank128Ref is a byte-exact copy of the pre-optimization byte-wise
// bitmap popcount, kept as the equivalence reference for bitmapRank128.
func bitmapRank128Ref(bitmap []byte, byteIdx int, bitIdx uint) int {
	rank := 0
	for b := range byteIdx {
		rank += bits.OnesCount8(bitmap[b])
	}
	rank += bits.OnesCount8(bitmap[byteIdx] & ((1 << bitIdx) - 1))
	return rank
}

// TestExtractPackedValueUTL_ReferenceEquivalence pins the pointer-load
// variant to the bounds-checked reference over the full valid domain:
// every position, bitwidths 1..32, plus the documented truncated-payload
// corner cases (degraded reads yield zero bits).
func TestExtractPackedValueUTL_ReferenceEquivalence(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 1))
	payload := make([]byte, utlSuperWordBytes*2)
	for i := range payload {
		payload[i] = byte(rng.Uint32())
	}
	for bw := 1; bw <= 32; bw++ {
		t.Run("bwFull", func(t *testing.T) {
			for pos := range blockSize {
				want := extractPackedValueUTLRef(pos, payload, bw)
				got := extractPackedValueUTL(pos, payload, bw)
				assert.Equal(t, want, got, "bw=%d pos=%d", bw, pos)
			}
		})
	}
	// Degenerate inputs: payload ends before the first read (guard misses).
	for bw := 1; bw <= 32; bw++ {
		for _, n := range []int{0, 1, 3, 7} {
			trunc := payload[:n]
			assert.Equal(t,
				extractPackedValueUTLRef(0, trunc, bw),
				extractPackedValueUTL(0, trunc, bw),
				"truncated payload len=%d bw=%d", n, bw)
		}
	}
}

// TestBitmapRank128_ReferenceEquivalence pins the OnesCount64 rank
// against the byte-wise reference for every byte/bit position and
// pseudo-random bitmap contents (exercises both byteIdx < 8 and >= 8).
func TestBitmapRank128_ReferenceEquivalence(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 2))
	for iter := range 200 {
		var bitmap [16]byte
		for i := range bitmap {
			bitmap[i] = byte(rng.Uint32())
		}
		for byteIdx := range 16 {
			for bitIdx := uint(0); bitIdx < 8; bitIdx++ {
				want := bitmapRank128Ref(bitmap[:], byteIdx, bitIdx)
				got := bitmapRank128(bitmap[:], byteIdx, bitIdx)
				assert.Equal(t, want, got, "iter=%d byteIdx=%d bitIdx=%d bitmap=% x", iter, byteIdx, bitIdx, bitmap)
			}
		}
	}
}

// applyLaneExceptionsRef is a byte-exact copy of the pre-optimization
// function, kept as the equivalence reference for the production variant.
func applyLaneExceptionsRef(laneValues []uint32, excRegion []byte,
	excCount, lane, posInLane, count, bitWidth int) {
	shift := uint(bitWidth)
	// The SVB stream follows the exception index (positions or bitmap).
	svbData := excRegion[excIndexSize(excCount):]
	if excCount <= excBitmapThreshold {
		for i := range excCount {
			p := int(excRegion[i])
			if p%utlLaneCount != lane {
				continue
			}
			v := p / utlLaneCount
			if v > posInLane {
				break
			}
			highBits := svbDecodeOneInternal(svbData, excCount, i)
			laneValues[v] |= highBits << shift
		}
	} else {
		// The bitmap occupies the first 16 bytes of the region.
		bitmap := excRegion[:16]
		for v := 0; v <= posInLane; v++ {
			seqIdx := lane + v*utlLaneCount
			if seqIdx >= count {
				break
			}
			byteIdx := seqIdx / 8
			bitIdx := uint(seqIdx % 8)
			if bitmap[byteIdx]&(1<<bitIdx) == 0 {
				continue
			}
			rank := 0
			for b := range byteIdx {
				rank += bits.OnesCount8(bitmap[b])
			}
			rank += bits.OnesCount8(bitmap[byteIdx] & ((1 << bitIdx) - 1))
			highBits := svbDecodeOneInternal(svbData, excCount, rank)
			laneValues[v] |= highBits << shift
		}
	}
}

// TestApplyLaneExceptions_ReferenceEquivalence compares the production
// applyLaneExceptions (bitmap rank) with the reference for both index
// modes, random region contents and every lane x posInLane combination.
func TestApplyLaneExceptions_ReferenceEquivalence(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 3))
	for excCount := 1; excCount <= blockSize; excCount++ {
		// Both variants share the same region; highBits only sanity-checks
		// that the region actually encodes excCount values.
		region, highBits := buildSVBRegion(t, rng, excCount)
		require.Len(t, highBits, excCount)
		for _, bw := range []int{4, 8, 12, 16, 20, 32} {
			for lane := range utlLaneCount {
				for posInLane := range utlValuesPerLane {
					ref := make([]uint32, utlValuesPerLane)
					applyLaneExceptionsRef(ref, region, excCount, lane, posInLane, blockSize, bw)
					got := make([]uint32, utlValuesPerLane)
					applyLaneExceptions(got, region, excCount, lane, posInLane, blockSize, bw)
					assert.Equal(t, ref, got,
						"excCount=%d bw=%d lane=%d posInLane=%d", excCount, bw, lane, posInLane)
				}
			}
		}
	}
}

// buildSVBRegion builds a synthetic exception region (index + SVB data)
// so applyLaneExceptions can be unit-tested directly. Returns the region
// and the high-bit values encoded into it.
func buildSVBRegion(t *testing.T, rng *rand.Rand, excCount int) ([]byte, []uint32) {
	t.Helper()
	highBits := make([]uint32, excCount)
	for i := range highBits {
		highBits[i] = uint32(rng.Uint32()) >> 16 // 16-bit high bits fit any bw
	}
	region := make([]byte, excIndexSize(excCount)+maxSVBEncodedLen(excCount))
	n := encodeSVBIntoDst(region[excIndexSize(excCount):], highBits)
	region = region[:excIndexSize(excCount)+n]
	if excCount <= excBitmapThreshold {
		// Sorted distinct positions: random subset of 0..127.
		for i, p := range rng.Perm(blockSize)[:excCount] {
			region[i] = byte(p)
		}
	} else {
		// Bitmap with exactly excCount distinct set bits (sorted by rank).
		clear(region[:excIndexSize(excCount)])
		for _, p := range rng.Perm(blockSize)[:excCount] {
			region[p/8] |= 1 << (p % 8)
		}
	}
	return region, highBits
}

// TestGetUint32_DeltaZigZagManyExceptions is the end-to-end guard for the
// bitmap+zigzag+delta combination: alternating-sign deltas force zigzag
// while the periodic jumps force >16 exceptions (bitmap mode). The
// bitmap-rank rewrite sits on this previously-untested path.
func TestGetUint32_DeltaZigZagManyExceptions(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		v := 1000 + 2*i
		if i%7 == 0 {
			v += 60000 // negative deltas -> zigzag; jumps -> >16 exceptions
		}
		values[i] = uint32(v)
	}
	packed, err := PackUint32(Delta, values, nil, nil)
	require.NoError(t, err)

	count, bw, excCount, hasDelta, hasFOR, hasZZ, _, err := Header(packed)
	require.NoError(t, err)
	assert.True(t, hasDelta, "delta must be on")
	assert.True(t, hasZZ, "zigzag must be on for this dataset")
	assert.False(t, hasFOR)
	assert.Greater(t, excCount, excBitmapThreshold, "dataset must exercise bitmap path")
	assert.Equal(t, blockSize, count)

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, ScratchLen))
	require.NoError(t, err)
	scratch := make([]uint32, ScratchLen)
	for pos := range count {
		got, err := GetUint32(pos, packed, scratch)
		require.NoError(t, err, "pos=%d bw=%d", pos, bw)
		assert.Equal(t, unpacked[pos], got, "pos=%d bw=%d", pos, bw)
	}
}

// TestGetUint32_DeltaZigZagFewExceptions covers the position-list mode of
// the zigzag+delta combination (excCount <= 16).
func TestGetUint32_DeltaZigZagFewExceptions(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		v := 100 + 2*i
		if i%32 == 0 {
			v += 50000 // ~4 exceptions at bw16, zigzag on
		}
		values[i] = uint32(v)
	}
	packed, err := PackUint32(Delta, values, nil, nil)
	require.NoError(t, err)

	_, _, excCount, hasDelta, _, hasZZ, _, err := Header(packed)
	require.NoError(t, err)
	assert.True(t, hasDelta)
	assert.True(t, hasZZ)
	require.Greater(t, excCount, 0)
	require.LessOrEqual(t, excCount, excBitmapThreshold)

	unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, ScratchLen))
	require.NoError(t, err)
	scratch := make([]uint32, ScratchLen)
	for pos := range unpacked {
		got, err := GetUint32(pos, packed, scratch)
		require.NoError(t, err, "pos=%d", pos)
		assert.Equal(t, unpacked[pos], got, "pos=%d", pos)
	}
}

// sink prevents the compiler from optimizing away benchmark bodies.
var sink int64

// BenchmarkGetExtractValueUTL isolates extractPackedValueUTL (bw16, the
// worst case for the bounds checks: two 4-byte reads + slice headers).
func BenchmarkGetExtractValueUTL(b *testing.B) {
	payload := make([]byte, utlSuperWordBytes*2)
	for i := range payload {
		payload[i] = byte(i * 7)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		sink += int64(extractPackedValueUTL(b.N%blockSize, payload, 16))
	}
}

// BenchmarkGetFindExceptionIndexBitmap isolates the bitmap probe of
// findExceptionIndex (bitmap mode, bit set at the probed position).
func BenchmarkGetFindExceptionIndexBitmap(b *testing.B) {
	var bitmap [16]byte
	for i := range 20 { // >16 set bits: bitmap mode
		bitmap[i*6%blockSize/8] |= 1 << ((i * 6) % 8)
	}
	pos := 63
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		sink += int64(findExceptionIndex(bitmap[:], 20, pos))
	}
}

// BenchmarkGetBitmapRank128 isolates the full 128-bit rank computation
// (the per-set-bit cost inside the bitmap path of applyLaneExceptions).
func BenchmarkGetBitmapRank128(b *testing.B) {
	var bitmap [16]byte
	for i := range 20 {
		bitmap[i*6%blockSize/8] |= 1 << ((i * 6) % 8)
	}
	byteIdx, bitIdx := 7, uint(3)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		sink += int64(bitmapRank128(bitmap[:], byteIdx, bitIdx))
	}
}
