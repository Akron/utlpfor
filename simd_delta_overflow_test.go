//go:build goexperiment.simd && amd64

package utlpfor

// Bit-exactness harness for the delta-decode overflow detection
// (step 3.1 of improvement-plan-2.md).
//
// The overflow position reported by deltaDecodePerLaneWithOverflow* is part of
// the observable behaviour: UnpackUint32 surfaces it via ErrOverflow.Position.
// These tests pin the SIMD kernels to the scalar reference: for random blocks
// with injected overflows at random lanes, the SIMD overflow position and the
// fully decoded values must equal the scalar result exactly.

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// overflowPosRef is the scalar reference for the overflow checks. Keeping a
// named alias documents which function is the contract under test.
var overflowPosRef = deltaDecodePerLaneWithOverflowScalar

// makeOverflowDeltas builds a delta block (UTL lane order) whose prefix sums
// overflow. Setting any value to 0xFFFFFFFF makes the prefix sum 16 positions
// later (same lane, next row) wrap for every non-zero delta, so each injected
// index guarantees at least one overflow in its lane. The exact first-overflow
// position is NOT asserted anywhere - only scalar/SIMD equality is.
func makeOverflowDeltas(rng *rand.Rand, overflowIdxs ...int) []uint32 {
	// Fill with random deltas first; every prefix sum then risks overflow,
	// which is exactly the multi-overflow scenario the kernels must handle.
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = rng.Uint32()
	}
	// Inject 0xFFFFFFFF at the requested lane-order indices.
	for _, idx := range overflowIdxs {
		if idx < 0 || idx >= blockSize {
			continue
		}
		values[idx] = 0xFFFFFFFF
		// Make the next row in the same lane a guaranteed wrap candidate
		// (0xFFFFFFFF + delta > 2^32 for any non-zero delta).
		if next := idx + utlLaneCount; next < blockSize && values[next] == 0 {
			values[next] = uint32(rng.Intn(0xFFFFFFFF)) + 1
		}
	}
	return values
}

// assertOverflowMatchesScalar compares SIMD overflow position and decoded
// values against the scalar reference for one input block.
func assertOverflowMatchesScalar(t *testing.T, deltas []uint32, label string) {
	t.Helper()

	// Scalar reference: position + fully decoded values.
	scalarBuf := append([]uint32(nil), deltas...)
	wantPos := overflowPosRef(scalarBuf, false)

	if simdLevel >= simdLevelAVX2 {
		simdBuf := append([]uint32(nil), deltas...)
		gotPos := deltaDecodePerLaneWithOverflowAVX2(simdBuf, false)
		assert.Equal(t, wantPos, gotPos, "%s: AVX2 overflow position", label)
		assert.Equal(t, scalarBuf, simdBuf, "%s: AVX2 decoded values", label)
	}
	if simdLevel >= simdLevelSSE2 {
		simdBuf := append([]uint32(nil), deltas...)
		gotPos := deltaDecodePerLaneWithOverflowSSE2(simdBuf, false)
		assert.Equal(t, wantPos, gotPos, "%s: SSE2 overflow position", label)
		assert.Equal(t, scalarBuf, simdBuf, "%s: SSE2 decoded values", label)
	}
}

// TestDeltaDecodeOverflow_RandomInjection_SIMDMatchesScalar is the main
// harness: 500 random blocks with 1..4 injected overflow lanes each, plus
// blocks that overflow at every lane without any injection.
func TestDeltaDecodeOverflow_RandomInjection_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(20260907))

	for range 500 {
		// Inject overflow sources at 1..4 random lane-order positions per trial.
		n := 1 + rng.Intn(4)
		idxs := make([]int, 0, n)
		for range n {
			idxs = append(idxs, rng.Intn(blockSize))
		}
		deltas := makeOverflowDeltas(rng, idxs...)
		assertOverflowMatchesScalar(t, deltas, "random-injection")
	}
}

// TestDeltaDecodeOverflow_AllLanesOverflow_SIMDMatchesScalar builds blocks
// where every prefix sum wraps (all deltas near 2^32), so every lane
// overflows at every row - the v-major and lane-major scan orders must still
// agree on the first overflow position.
func TestDeltaDecodeOverflow_AllLanesOverflow_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(314159))

	for range 100 {
		values := make([]uint32, blockSize)
		for i := range values {
			// Large deltas: every lane overflows at v=1 already.
			values[i] = 0xFFFFFFFF - uint32(rng.Intn(16))
		}
		assertOverflowMatchesScalar(t, values, "all-overflow")
	}
}

// TestDeltaDecodeOverflow_NoOverflow_Random_SIMDMatchesScalar runs random
// non-overflow blocks through the harness (control group: position 0,
// values equal). Uses small deltas so no lane accumulates past 2^32.
func TestDeltaDecodeOverflow_NoOverflow_Random_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(271828))

	for range 200 {
		values := make([]uint32, blockSize)
		// Small lane bases and small deltas keep all prefix sums far from 2^32.
		for lane := range utlLaneCount {
			base := uint32(rng.Intn(1 << 20))
			for v := range utlValuesPerLane {
				values[lane+v*utlLaneCount] = base + uint32(rng.Intn(1<<16))
			}
		}
		assertOverflowMatchesScalar(t, values, "no-overflow")
	}
}

// TestDeltaDecodeOverflow_PartialBlocks_SIMDMatchesScalar covers the scalar
// fallback used by the unpack kernels for count < blockSize (partial blocks
// never run the SIMD overflow kernel). The SIMD kernels are called directly
// on a padded buffer to verify the padded result equals the scalar one.
func TestDeltaDecodeOverflow_PartialBlocks_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(161803))

	// Sweep sub-block lengths including non-full lanes.
	for _, n := range []int{1, 2, 3, 15, 16, 17, 31, 63, 64, 65, 100, 127} {
		values := make([]uint32, n)
		for i := range values {
			values[i] = rng.Uint32()
		}
		// Force an overflow somewhere in the partial block: value 0xFFFFFFFF
		// makes the next same-lane prefix sum wrap when the lane is full; for
		// short lanes the other random values still risk wrapping.
		if n > 1 {
			values[n-2] = 0xFFFFFFFF
			if values[n-1] == 0 {
				values[n-1] = 1
			}
		}
		// Scalar reference on the zero-padded buffer (padding never
		// introduces an overflow: 0 + delta == delta).
		pad := make([]uint32, blockSize)
		copy(pad, values)
		wantPos := overflowPosRef(append([]uint32(nil), pad...), false)

		if simdLevel >= simdLevelAVX2 {
			simdPad := append([]uint32(nil), pad...)
			gotPos := deltaDecodePerLaneWithOverflowAVX2(simdPad, false)
			assert.Equal(t, wantPos, gotPos, "n=%d: AVX2 overflow position (padded)", n)
		}
		if simdLevel >= simdLevelSSE2 {
			simdPad := append([]uint32(nil), pad...)
			gotPos := deltaDecodePerLaneWithOverflowSSE2(simdPad, false)
			assert.Equal(t, wantPos, gotPos, "n=%d: SSE2 overflow position (padded)", n)
		}
	}
}

// TestDeltaDecodeOverflow_SingleOverflow_EveryRowLane pins the reported
// position exactly: for every (row, lane) pair, a block whose ONLY prefix-sum
// wrap is at that position must report exactly v*utlLaneCount+lane. This
// deterministically covers the high lanes 8-15, where a too-narrow mask
// accumulator (uint8 instead of uint16) silently drops the overflow bits -
// random-data tests miss that whenever an earlier low-lane overflow exists.
func TestDeltaDecodeOverflow_SingleOverflow_EveryRowLane(t *testing.T) {
	for v := 1; v < utlValuesPerLane; v++ {
		for lane := 0; lane < utlLaneCount; lane++ {
			// All-zero deltas never wrap; wrap exactly one sum: the prefix
			// at row v of lane L is 0xFFFFFFFF, adding delta 1 wraps to 0.
			deltas := make([]uint32, blockSize)
			deltas[lane+(v-1)*utlLaneCount] = 0xFFFFFFFF
			deltas[lane+v*utlLaneCount] = 1
			wantPos := lane + v*utlLaneCount

			// The scalar reference must reproduce the constructed position.
			scalarBuf := append([]uint32(nil), deltas...)
			assert.Equal(t, wantPos, overflowPosRef(scalarBuf, false),
				"v=%d lane=%d: scalar reference", v, lane)

			if simdLevel >= simdLevelAVX2 {
				simdBuf := append([]uint32(nil), deltas...)
				assert.Equal(t, wantPos, deltaDecodePerLaneWithOverflowAVX2(simdBuf, false),
					"v=%d lane=%d: AVX2 position", v, lane)
			}
			if simdLevel >= simdLevelSSE2 {
				simdBuf := append([]uint32(nil), deltas...)
				assert.Equal(t, wantPos, deltaDecodePerLaneWithOverflowSSE2(simdBuf, false),
					"v=%d lane=%d: SSE2 position", v, lane)
			}
		}
	}
}

// TestUnpackDeltaOverflow_ErrorPosition_SIMDMatchesScalar pins the end-to-end
// behaviour: a corrupt delta block (valid header, corrupted delta payload)
// whose prefix sums overflow must return an ErrOverflow whose Position matches
// the scalar reference, on every level.
//
// Valid packed blocks never overflow - the encoder zigzag-encodes negative
// deltas, and the encoder never produces deltas whose prefix sums wrap - so
// the corruption ORs 0xFF000000 into every packed payload word of a
// Delta|NoFOR block. That forces large positive deltas whose prefix sums
// wrap. The exact position is derived by the scalar reference unpack; the
// API unpack (active SIMD level) must report the same position.
func TestUnpackDeltaOverflow_ErrorPosition_SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(999983))

	for trial := range 50 {
		// Large random raw values: packed deltas are already huge, so the
		// corruption reliably wraps the prefix sums in most lanes.
		values := make([]uint32, blockSize)
		for i := range values {
			// Keep the per-lane deltas non-negative-ish so the encoder does
			// not pick zigzag; raw 29+-bit magnitudes guarantee overflow.
			values[i] = uint32(rng.Uint32()>>4) << 4
		}
		packed, err := PackUint32(Delta|NoFOR, append([]uint32(nil), values...), nil, nil)
		require.NoError(t, err)

		// Header sanity: delta set; zigzag tolerated (corruption works
		// either way because OR-ing 0xFF000000 into zigzag deltas also
		// wraps after zigzag decode: 0xFF800000 decodes to a huge negative,
		// whose uint32 interpretation makes the next sum wrap).
		_, _, _, _, _, _, hasDelta, _, _, _ := decodeHeader(bo.Uint32(packed))
		require.True(t, hasDelta, "trial=%d", trial)

		// Corrupt: every payload word gets 0xFF in the top byte. The payload
		// region starts after the header (and svbLen + exc index when
		// exceptions exist), found via the decoded header.
		payloadStart, payloadBytes := corruptPayloadRegion(t, packed)
		for b := payloadStart; b < payloadStart+payloadBytes; b += 4 {
			packed[b] |= 0xFF
		}

		// Scalar reference unpack.
		_, _, scalarErr := unpackUint32Scalar(nil, make([]uint32, blockSize), append([]byte(nil), packed...), false)
		if scalarErr == nil {
			continue // corruption did not overflow this block; skip
		}
		overflowErr, ok := scalarErr.(*ErrOverflow)
		require.True(t, ok, "trial=%d: expected ErrOverflow, got %v", trial, scalarErr)
		wantPos := overflowErr.Position

		// Unpack via the API (dispatches to the active SIMD level).
		_, _, apiErr := UnpackUint32(packed, nil, make([]uint32, blockSize))
		require.Error(t, apiErr, "trial=%d", trial)
		apiOverflow, ok := apiErr.(*ErrOverflow)
		require.True(t, ok, "trial=%d: expected ErrOverflow, got %v", trial, apiErr)
		assert.Equal(t, wantPos, apiOverflow.Position, "trial=%d: overflow position mismatch", trial)
	}
}

// corruptPayloadRegion locates the UTL payload inside a packed uint32
// Delta|NoFOR block: header [+ svbLen + exc index] + payload. Returns the
// payload start offset and its byte size.
func corruptPayloadRegion(t *testing.T, packed []byte) (int, int) {
	t.Helper()
	header := bo.Uint32(packed)
	_, bitWidth, _, _, forWidth, hasExceptions, _, _, _, _ := decodeHeader(header)
	pOff := payloadOffset(forBaseBytes(forWidth), hasExceptions, false)
	return pOff, utlPayloadBytes(bitWidth)
}
