package utlpfor

import (
	"bytes"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAppend_EmptyDst_IdenticalOutput(t *testing.T) {
	// Append with an empty dst must be byte-identical to normal packing.
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 3)
	}

	v1 := slices.Clone(values)
	v2 := slices.Clone(values)

	dst1, err := PackUint32(0, v1, nil, nil)
	require.NoError(t, err)

	dst2 := make([]byte, 0, 1024)
	dst2, err = PackUint32(Append, v2, dst2, nil)
	require.NoError(t, err)

	assert.True(t, bytes.Equal(dst1, dst2),
		"Append with dst[:0] must produce identical output to non-Append")
}

func TestAppend_TwoBlockRoundTrip(t *testing.T) {
	// Pack two blocks back-to-back and decode them using consumed length.
	block1 := make([]uint32, 128)
	for i := range block1 {
		block1[i] = uint32(i)
	}
	block2 := make([]uint32, 128)
	for i := range block2 {
		block2[i] = uint32(i*7 + 1000)
	}

	orig1 := slices.Clone(block1)
	orig2 := slices.Clone(block2)

	scratch := make([]uint32, ScratchLen)
	var dst []byte
	var err error

	dst, err = PackUint32(Append, block1, dst, scratch)
	require.NoError(t, err)

	dst, err = PackUint32(Append, block2, dst, scratch)
	require.NoError(t, err)

	// Decode first block and use consumed bytes as offset for second block.
	unpacked1, consumed1, err := UnpackUint32(dst, nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig1, unpacked1)

	unpacked2, _, err := UnpackUint32(dst[consumed1:], nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig2, unpacked2)
}

func TestAppend_WithDelta(t *testing.T) {
	// Append must work when Delta encoding is enabled.
	block1 := make([]uint32, 128)
	for i := range block1 {
		block1[i] = uint32(i * 10)
	}
	block2 := make([]uint32, 128)
	for i := range block2 {
		block2[i] = uint32(5000 + i*3)
	}

	orig1 := slices.Clone(block1)
	orig2 := slices.Clone(block2)

	scratch := make([]uint32, ScratchLen)
	var dst []byte
	var err error

	dst, err = PackUint32(Delta|Append, block1, dst, scratch)
	require.NoError(t, err)

	dst, err = PackUint32(Delta|Append, block2, dst, scratch)
	require.NoError(t, err)

	unpacked1, consumed1, err := UnpackUint32(dst, nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig1, unpacked1)

	unpacked2, _, err := UnpackUint32(dst[consumed1:], nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig2, unpacked2)
}

func TestAppend_WithNoPatch(t *testing.T) {
	// Append must work when exception patching is disabled.
	block1 := make([]uint32, 128)
	for i := range block1 {
		block1[i] = uint32(i & 0xF)
	}
	block2 := make([]uint32, 128)
	for i := range block2 {
		block2[i] = uint32((i + 5) & 0xF)
	}

	orig1 := slices.Clone(block1)
	orig2 := slices.Clone(block2)

	scratch := make([]uint32, ScratchLen)
	var dst []byte
	var err error

	dst, err = PackUint32(NoPatch|Append, block1, dst, scratch)
	require.NoError(t, err)

	dst, err = PackUint32(NoPatch|Append, block2, dst, scratch)
	require.NoError(t, err)

	unpacked1, consumed1, err := UnpackUint32(dst, nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig1, unpacked1)

	unpacked2, _, err := UnpackUint32(dst[consumed1:], nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig2, unpacked2)
}

func TestAppend_WithNoFOR(t *testing.T) {
	// Append must work when FOR analysis is disabled.
	block1 := make([]uint32, 128)
	for i := range block1 {
		block1[i] = uint32(i * 2)
	}
	block2 := make([]uint32, 128)
	for i := range block2 {
		block2[i] = uint32(i * 4)
	}

	orig1 := slices.Clone(block1)
	orig2 := slices.Clone(block2)

	scratch := make([]uint32, ScratchLen)
	var dst []byte
	var err error

	dst, err = PackUint32(NoFOR|Append, block1, dst, scratch)
	require.NoError(t, err)

	dst, err = PackUint32(NoFOR|Append, block2, dst, scratch)
	require.NoError(t, err)

	unpacked1, consumed1, err := UnpackUint32(dst, nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig1, unpacked1)

	unpacked2, _, err := UnpackUint32(dst[consumed1:], nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig2, unpacked2)
}

func TestAppend_AllFlagsCombined(t *testing.T) {
	// Append must stay correct with all pack-time flags combined.
	block1 := make([]uint32, 128)
	for i := range block1 {
		block1[i] = uint32(i * 5)
	}

	orig1 := slices.Clone(block1)

	scratch := make([]uint32, ScratchLen)
	var dst []byte
	var err error

	dst, err = PackUint32(Delta|NoPatch|NoFOR|Append, block1, dst, scratch)
	require.NoError(t, err)

	unpacked, _, err := UnpackUint32(dst, nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig1, unpacked)
}

func TestAppend_PreservesPrefix(t *testing.T) {
	// Existing prefix bytes must remain untouched in Append mode.
	prefix := []byte("HELLO")
	dst := make([]byte, len(prefix), 1024)
	copy(dst, prefix)

	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	orig := slices.Clone(values)

	scratch := make([]uint32, ScratchLen)
	result, err := PackUint32(Append, values, dst, scratch)
	require.NoError(t, err)

	assert.Equal(t, prefix, result[:5],
		"Append must preserve existing prefix data")

	unpacked, _, err := UnpackUint32(result[5:], nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig, unpacked)
}

func TestAppend_ThreeConsecutiveBlocks(t *testing.T) {
	// Repeated appends should form a valid stream of consecutive blocks.
	blocks := [3][]uint32{
		make([]uint32, 128),
		make([]uint32, 128),
		make([]uint32, 128),
	}
	for i := range blocks[0] {
		blocks[0][i] = uint32(i)
	}
	for i := range blocks[1] {
		blocks[1][i] = uint32(i*3 + 100)
	}
	for i := range blocks[2] {
		blocks[2][i] = uint32(i*7 + 9999)
	}

	originals := [3][]uint32{
		slices.Clone(blocks[0]),
		slices.Clone(blocks[1]),
		slices.Clone(blocks[2]),
	}

	scratch := make([]uint32, ScratchLen)
	var dst []byte
	var err error

	for i := range 3 {
		dst, err = PackUint32(Append, blocks[i], dst, scratch)
		require.NoError(t, err)
	}

	offset := 0
	for i := range 3 {
		unpacked, consumed, err := UnpackUint32(dst[offset:], nil, scratch)
		require.NoError(t, err, "block %d", i)
		assert.Equal(t, originals[i], unpacked, "block %d", i)
		offset += consumed
	}
}

func TestAppend_ValuesUnmodified(t *testing.T) {
	// Append must guarantee that the original values slice is not modified.
	scratch := make([]uint32, ScratchLen)

	t.Run("Append_FOR_active", func(t *testing.T) {
		// Values with min > 0 trigger FOR subtraction (in-place modification).
		values := make([]uint32, 128)
		for i := range values {
			values[i] = uint32(1000 + i)
		}
		original := slices.Clone(values)

		_, err := PackUint32(Append, values, nil, scratch)
		require.NoError(t, err)
		assert.Equal(t, original, values,
			"Append must not modify the values slice (FOR subtraction)")
	})

	t.Run("Append_Delta", func(t *testing.T) {
		// Delta encoding modifies values in-place.
		values := make([]uint32, 128)
		for i := range values {
			values[i] = uint32(i * 10)
		}
		original := slices.Clone(values)

		_, err := PackUint32(Delta|Append, values, nil, scratch)
		require.NoError(t, err)
		assert.Equal(t, original, values,
			"Append|Delta must not modify the values slice")
	})

	t.Run("MultiBlock_SharedArray", func(t *testing.T) {
		// Simulates the Krawfish forward writer pattern: a fixed-size buffer
		// is packed multiple times without cloning.
		var buf [128]uint32
		for i := range buf {
			buf[i] = uint32(500 + i*3)
		}
		original := buf

		var dst []byte
		var err error
		for range 3 {
			dst, err = PackUint32(Append, buf[:], dst, scratch)
			require.NoError(t, err)
		}
		assert.Equal(t, original, buf,
			"shared fixed-size array must be unchanged after multiple Append packs")
	})
}

func TestAppend_StaleDataWithExceptions(t *testing.T) {
	// When the Append path's packBlockScalar allocates a new buffer
	// (because maxTotalLen > remaining cap) but the actual encoded size
	// fits in the original dst, the Append code must copy the new data
	// back -- otherwise dst contains stale bytes.
	scratch := make([]uint32, ScratchLen)

	makeBlock := func(outlierVal uint32) []uint32 {
		vals := make([]uint32, 128)
		for i := range vals {
			vals[i] = uint32(i % 16)
		}
		// Outliers at regular intervals force exception encoding,
		// creating a gap between worst-case and actual SVB size.
		for i := 0; i < 128; i += 4 {
			vals[i] = outlierVal
		}
		return vals
	}

	block1 := makeBlock(0x10000)
	block2 := makeBlock(0x20000)
	orig1 := slices.Clone(block1)
	orig2 := slices.Clone(block2)

	// Probe actual packed size.
	probe, err := PackUint32(0, slices.Clone(block1), nil, scratch)
	require.NoError(t, err)
	actualLen := len(probe)

	// Cap = 2*actualLen: the second block's actual size fits in the
	// remaining capacity, but its worst-case size (with max SVB) does not.
	dst := make([]byte, 0, 2*actualLen)

	dst, err = PackUint32(Append, block1, dst, scratch)
	require.NoError(t, err)

	dst, err = PackUint32(Append, block2, dst, scratch)
	require.NoError(t, err)

	decoded1, consumed1, err := UnpackUint32(dst, nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig1, decoded1, "block 1 must round-trip")

	decoded2, _, err := UnpackUint32(dst[consumed1:], nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig2, decoded2, "block 2 must not contain stale data")
}

func TestAppend_DeltaWithExceptions(t *testing.T) {
	// Append + Delta where the encoded block contains exceptions.
	scratch := make([]uint32, ScratchLen)

	makeBlock := func(base, outlier uint32) []uint32 {
		vals := make([]uint32, 128)
		for i := range vals {
			vals[i] = base + uint32(i*3)
		}
		for i := 0; i < 128; i += 4 {
			vals[i] = outlier
		}
		return vals
	}

	block1 := makeBlock(1000, 500000)
	block2 := makeBlock(2000, 600000)
	orig1 := slices.Clone(block1)
	orig2 := slices.Clone(block2)

	probe, err := PackUint32(Delta, slices.Clone(block1), nil, scratch)
	require.NoError(t, err)

	dst := make([]byte, 0, 2*len(probe))

	dst, err = PackUint32(Delta|Append, block1, dst, scratch)
	require.NoError(t, err)

	dst, err = PackUint32(Delta|Append, block2, dst, scratch)
	require.NoError(t, err)

	decoded1, consumed1, err := UnpackUint32(dst, nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig1, decoded1, "block 1 round-trip with Delta")

	decoded2, _, err := UnpackUint32(dst[consumed1:], nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig2, decoded2, "block 2 round-trip with Delta")
}

func TestAppend_NoFORWithExceptions(t *testing.T) {
	// Append + NoFOR where the encoded block contains exceptions.
	scratch := make([]uint32, ScratchLen)

	makeBlock := func(outlier uint32) []uint32 {
		vals := make([]uint32, 128)
		for i := range vals {
			vals[i] = uint32(i % 16)
		}
		for i := 0; i < 128; i += 4 {
			vals[i] = outlier
		}
		return vals
	}

	block1 := makeBlock(0x10000)
	block2 := makeBlock(0x20000)
	orig1 := slices.Clone(block1)
	orig2 := slices.Clone(block2)

	probe, err := PackUint32(NoFOR, slices.Clone(block1), nil, scratch)
	require.NoError(t, err)

	dst := make([]byte, 0, 2*len(probe))

	dst, err = PackUint32(NoFOR|Append, block1, dst, scratch)
	require.NoError(t, err)

	dst, err = PackUint32(NoFOR|Append, block2, dst, scratch)
	require.NoError(t, err)

	decoded1, consumed1, err := UnpackUint32(dst, nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig1, decoded1, "block 1 round-trip with NoFOR")

	decoded2, _, err := UnpackUint32(dst[consumed1:], nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig2, decoded2, "block 2 round-trip with NoFOR")
}

func TestAppend_NonAppendThenAppendWithExceptions(t *testing.T) {
	// Common pattern: pack block 1 without Append, then Append block 2
	// into the resulting buffer. The first call sets dst capacity;
	// the second must handle a potential reallocation correctly.
	scratch := make([]uint32, ScratchLen)

	block1 := make([]uint32, 128)
	for i := range block1 {
		block1[i] = uint32(i * 7)
	}
	orig1 := slices.Clone(block1)

	block2 := make([]uint32, 128)
	for i := range block2 {
		block2[i] = uint32(i % 16)
	}
	for i := 0; i < 128; i += 4 {
		block2[i] = 0x10000
	}
	orig2 := slices.Clone(block2)

	dst, err := PackUint32(0, block1, nil, scratch)
	require.NoError(t, err)

	dst, err = PackUint32(Append, block2, dst, scratch)
	require.NoError(t, err)

	decoded1, consumed1, err := UnpackUint32(dst, nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig1, decoded1, "block 1 from non-Append call")

	decoded2, _, err := UnpackUint32(dst[consumed1:], nil, scratch)
	require.NoError(t, err)
	assert.Equal(t, orig2, decoded2, "block 2 from Append call")
}

func TestAppend_FlagNotInHeader(t *testing.T) {
	// Append is a control flag only and must not be persisted to header bits.
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}

	scratch := make([]uint32, ScratchLen)
	var dst []byte
	var err error

	dst, err = PackUint32(Append, values, dst, scratch)
	require.NoError(t, err)
	require.True(t, len(dst) >= 4)

	header := bo.Uint32(dst)
	appendBitInHeader := uint32(1 << 4)
	assert.Zero(t, header&appendBitInHeader,
		"Append flag must NOT appear in the encoded block header (bit 4 of raw header must be 0)")
}
