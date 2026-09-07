package utlpfor

import (
	"errors"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// corruptTarget is a valid packed block plus the value count it encodes.
// The corruption tests truncate and byte-flip these blocks and require
// that every public reader stays panic-free.
type corruptTarget struct {
	name   string
	packed []byte
	count  int
}

// packUint32Values packs values with the given flag for the corruption
// targets, failing the test on pack errors.
func packUint32Values(t testing.TB, name string, flag Flag, values []uint32) []byte {
	t.Helper()
	packed, err := PackUint32(flag, values, nil, nil)
	require.NoError(t, err, "target %s", name)
	return packed
}

// buildCorruptTargets32 creates uint32 blocks covering all decode paths:
// plain, few exceptions (position list), many exceptions (bitmap), delta,
// delta with FOR, delta with negative deltas (zigzag), and delta with
// exceptions.
func buildCorruptTargets32(t testing.TB) []corruptTarget {
	t.Helper()

	rng := rand.New(rand.NewPCG(7, 42))

	fill := func(v []uint32, fn func(i int) uint32) []uint32 {
		for i := range v {
			v[i] = fn(i)
		}
		return v
	}

	plain := fill(make([]uint32, blockSize), func(i int) uint32 { return uint32(i % 200) })

	fewExc := fill(make([]uint32, blockSize), func(i int) uint32 { return uint32(i % 50) })
	fewExc[7] = 0xFFFF0000
	fewExc[50] = 0x00FF0000

	manyExc := fill(make([]uint32, blockSize), func(i int) uint32 { return uint32(i % 50) })
	for j := range 48 {
		manyExc[(j*11)%blockSize] = 0xFF000000 | uint32(rng.IntN(0xFFFFFF))
	}

	delta := fill(make([]uint32, blockSize), func(i int) uint32 { return 1000 + uint32(i*3) })

	deltaFor := fill(make([]uint32, blockSize), func(i int) uint32 { return 1000000 + uint32(i*3) })

	deltaZigZag := fill(make([]uint32, blockSize), func(i int) uint32 {
		if i%2 == 0 {
			return 1000 + uint32(i*3)
		}
		return 1000 + uint32((i-1)*3) - 97
	})

	deltaExc := fill(make([]uint32, blockSize), func(i int) uint32 { return 1000 + uint32(i*3) })
	deltaExc[64] = 0xFFFF0000

	defs := []struct {
		name   string
		flag   Flag
		values []uint32
	}{
		{"plain", 0, plain},
		{"few_exceptions", 0, fewExc},
		{"many_exceptions", 0, manyExc},
		{"delta", Delta, delta},
		{"delta_for", Delta, deltaFor},
		{"delta_zigzag", Delta, deltaZigZag},
		{"delta_exceptions", Delta, deltaExc},
	}

	targets := make([]corruptTarget, 0, len(defs))
	for _, d := range defs {
		targets = append(targets, corruptTarget{
			name:   d.name,
			packed: packUint32Values(t, d.name, d.flag, d.values),
			count:  len(d.values),
		})
	}
	return targets
}

// buildCorruptTargets64 creates uint64 blocks covering the three uint64
// layouts: all-fit-32 single block, FOR64 single block, and the
// combine-with-next two-block encoding.
func buildCorruptTargets64(t testing.TB) []corruptTarget {
	t.Helper()

	fill64 := func(v []uint64, fn func(i int) uint64) []uint64 {
		for i := range v {
			v[i] = fn(i)
		}
		return v
	}

	fit32 := fill64(make([]uint64, blockSize), func(i int) uint64 { return uint64(i % 200) })

	for64 := fill64(make([]uint64, blockSize), func(i int) uint64 { return 0x100000000 + uint64(i*3) })

	twoBlock := fill64(make([]uint64, blockSize), func(i int) uint64 {
		return 0x1234567800000000 | uint64(i*7)
	})

	defs := []struct {
		name   string
		flag   Flag
		values []uint64
	}{
		{"u64_fit32", 0, fit32},
		{"u64_for64", 0, for64},
		{"u64_two_block", 0, twoBlock},
	}

	targets := make([]corruptTarget, 0, len(defs))
	for _, d := range defs {
		packed, err := PackUint64(d.flag, d.values, nil, make([]uint32, ScratchLen64))
		require.NoError(t, err, "target %s", d.name)
		targets = append(targets, corruptTarget{name: d.name, packed: packed, count: len(d.values)})
	}
	return targets
}

// probePositions returns the positions walked by the corruption tests:
// block edges plus positions across all lane rows for the delta path.
func probePositions(count int) []int {
	positions := make([]int, 0, 9)
	for _, p := range []int{0, 1, 7, 15, 16, 31, 64, 100, count - 1} {
		if p >= 0 && p < count {
			positions = append(positions, p)
		}
	}
	return positions
}

// assertKnownUTLError fails the test when err is non-nil but not one of
// the documented package errors.
func assertKnownUTLError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		return
	}
	var overflow *ErrOverflow
	assert.True(t,
		errors.Is(err, ErrInvalidBuffer) ||
			errors.Is(err, ErrPositionOutOfRange) ||
			errors.Is(err, ErrInvalidBlockLength) ||
			errors.Is(err, ErrInvalidFlags) ||
			errors.Is(err, ErrUnsupportedType) ||
			errors.As(err, &overflow),
		"unexpected error type: %v", err)
}

// TestGetUint32_TruncatedBlocks_NeverPanics cuts a valid block at every
// possible length. GetUint32 must never panic; when it succeeds on a
// truncated buffer, that value must match the untruncated block, because
// Get validates every region it reads before reading it.
func TestGetUint32_TruncatedBlocks_NeverPanics(t *testing.T) {
	for _, target := range buildCorruptTargets32(t) {
		t.Run(target.name, func(t *testing.T) {
			scratch := make([]uint32, ScratchLen)
			for cut := 1; cut < len(target.packed); cut++ {
				truncated := target.packed[:cut]
				for _, pos := range probePositions(target.count) {
					got, err := GetUint32(pos, truncated, scratch)
					assertKnownUTLError(t, err)
					if err == nil {
						want, err := GetUint32(pos, target.packed, scratch)
						require.NoError(t, err)
						assert.Equal(t, want, got,
							"cut=%d pos=%d: successful Get on truncated block must match full block",
							cut, pos)
					}
				}
			}
		})
	}
}

// TestGetUint64_TruncatedBlocks_NeverPanics applies the same truncation
// sweep to the uint64 layouts, including the combine-with-next metadata
// region between the two sub-blocks.
func TestGetUint64_TruncatedBlocks_NeverPanics(t *testing.T) {
	for _, target := range buildCorruptTargets64(t) {
		t.Run(target.name, func(t *testing.T) {
			scratch := make([]uint32, ScratchLen64)
			for cut := range len(target.packed) {
				truncated := target.packed[:cut]
				for _, pos := range probePositions(target.count) {
					got, err := GetUint64(pos, truncated, scratch)
					assertKnownUTLError(t, err)
					if err == nil {
						want, err := GetUint64(pos, target.packed, scratch)
						require.NoError(t, err)
						assert.Equal(t, want, got,
							"cut=%d pos=%d: successful Get on truncated block must match full block",
							cut, pos)
					}
				}
			}
		})
	}
}

// TestGetUint32_CorruptSVBLen overwrites the svbLen header field with the
// maximum 16-bit value. This is the repro from the analysis: the block
// claims far more StreamVByte data than the buffer holds.
func TestGetUint32_CorruptSVBLen(t *testing.T) {
	for _, target := range buildCorruptTargets32(t) {
		t.Run(target.name, func(t *testing.T) {
			header := bo.Uint32(target.packed)
			_, _, _, excCount, _, hasExceptions, _, _, _, _ := decodeHeader(header)
			if !hasExceptions {
				return
			}
			require.Greater(t, excCount, 0)

			corrupt := make([]byte, len(target.packed))
			copy(corrupt, target.packed)
			bo.PutUint16(corrupt[headerBytes:], 0xFFFF)

			scratch := make([]uint32, ScratchLen)
			for _, pos := range probePositions(target.count) {
				_, err := GetUint32(pos, corrupt, scratch)
				assert.ErrorIs(t, err, ErrInvalidBuffer, "pos=%d", pos)
			}
		})
	}
}

// TestGet_CorruptBytes_NeverPanics flips every single byte of a valid
// block to a few probe values and requires that GetUint32, GetUint64,
// UnpackUint32, UnpackUint64, BlockLength, and Header all stay
// panic-free and only return documented errors.
func TestGet_CorruptBytes_NeverPanics(t *testing.T) {
	targets := append(buildCorruptTargets32(t), buildCorruptTargets64(t)...)

	scratch := make([]uint32, ScratchLen64)
	for _, target := range targets {
		t.Run(target.name, subtestCorruptBytes(t, target, scratch))
	}
}

// subtestCorruptBytes runs the byte-flip sweep for a single target.
func subtestCorruptBytes(t *testing.T, target corruptTarget, scratch []uint32) func(t *testing.T) {
	return func(t *testing.T) {
		// Zero, all-ones, and bitwise inversion per byte position.
		for i := range target.packed {
			for _, probe := range []byte{0x00, 0xFF} {
				corrupt := make([]byte, len(target.packed))
				copy(corrupt, target.packed)
				corrupt[i] = probe

				for _, pos := range probePositions(target.count) {
					_, err := GetUint32(pos, corrupt, scratch)
					assertKnownUTLError(t, err)

					_, err = GetUint64(pos, corrupt, scratch)
					assertKnownUTLError(t, err)
				}

				_, _, err := UnpackUint32(corrupt, nil, scratch)
				assertKnownUTLError(t, err)

				_, _, err = UnpackUint64(corrupt, nil, scratch)
				assertKnownUTLError(t, err)

				_, err = BlockLength(corrupt)
				assertKnownUTLError(t, err)

				_, _, _, _, _, _, _, err = Header(corrupt)
				assertKnownUTLError(t, err)
			}
		}
	}
}

// TestUnpackUint32_CorruptExceptionPositions pins the exception-position
// clamp: a corrupt position byte in the exception index must not index
// the destination slice out of range during unpack or get.
func TestUnpackUint32_CorruptExceptionPositions(t *testing.T) {
	for _, target := range buildCorruptTargets32(t) {
		header := bo.Uint32(target.packed)
		_, bitWidth, _, excCount, _, hasExceptions, _, _, _, _ := decodeHeader(header)
		if !hasExceptions || excCount > excBitmapThreshold {
			continue
		}

		excStart := headerBytes + svbLenBytes + utlPayloadBytes(bitWidth)

		for excIdx := range excCount {
			corrupt := make([]byte, len(target.packed))
			copy(corrupt, target.packed)
			corrupt[excStart+excIdx] = 0xFF

			_, _, err := UnpackUint32(corrupt, nil, make([]uint32, ScratchLen))
			assertKnownUTLError(t, err)

			_, err = GetUint32(0, corrupt, nil)
			assertKnownUTLError(t, err)
		}
	}
}

// FuzzGetUint32CorruptBlock feeds arbitrary bytes to the point-lookup
// API. Neither GetUint32 nor GetUint64 may panic regardless of input;
// any returned error must be a documented package error.
func FuzzGetUint32CorruptBlock(f *testing.F) {
	for _, target := range buildCorruptTargets32(f) {
		f.Add(target.packed)
	}
	for _, target := range buildCorruptTargets64(f) {
		f.Add(target.packed)
	}
	f.Add([]byte{})
	f.Add([]byte{0xFF, 0xFF, 0xFF, 0xFF})
	f.Add([]byte{0x80, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF})

	f.Fuzz(func(t *testing.T, data []byte) {
		scratch := make([]uint32, ScratchLen64)
		for _, pos := range []int{0, 7, 64, 127} {
			_, err := GetUint32(pos, data, scratch)
			assertKnownUTLError(t, err)

			_, err = GetUint64(pos, data, scratch)
			assertKnownUTLError(t, err)
		}
	})
}

// FuzzCorruptBlockAPIs applies every public reader to arbitrary bytes.
// Complements the round-trip fuzz targets, which only ever feed valid
// blocks, with the corrupt-input class that uncovered the svbLen bug.
func FuzzCorruptBlockAPIs(f *testing.F) {
	for _, target := range buildCorruptTargets32(f) {
		f.Add(target.packed)
	}
	for _, target := range buildCorruptTargets64(f) {
		f.Add(target.packed)
	}
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, data []byte) {
		scratch := make([]uint32, ScratchLen64)

		for _, pos := range []int{0, 7, 64, 127} {
			_, err := GetUint32(pos, data, scratch)
			assertKnownUTLError(t, err)

			_, err = GetUint64(pos, data, scratch)
			assertKnownUTLError(t, err)
		}

		_, _, err := UnpackUint32(data, nil, scratch)
		assertKnownUTLError(t, err)

		_, _, err = UnpackUint64(data, nil, scratch)
		assertKnownUTLError(t, err)

		_, err = BlockLength(data)
		assertKnownUTLError(t, err)

		_, _, _, _, _, _, _, err = Header(data)
		assertKnownUTLError(t, err)
	})
}
