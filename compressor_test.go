package utlpfor_test

import (
	"slices"
	"testing"

	utlpfor "github.com/Akron/utlpfor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCompressor_RoundTrip_Raw(t *testing.T) {
	c := utlpfor.NewUint32()
	values := []uint32{10, 20, 30, 40, 50}
	original := slices.Clone(values)

	packed, err := c.Compress(0, nil, values)
	require.NoError(t, err)
	require.NotEmpty(t, packed)

	unpacked, consumed, err := c.Decompress(nil, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
	assert.Equal(t, original, unpacked)
}

func TestCompressor_RoundTrip_Delta(t *testing.T) {
	c := utlpfor.NewUint32()
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(100 + i*3)
	}
	original := slices.Clone(values)

	packed, err := c.Compress(utlpfor.Delta, nil, values)
	require.NoError(t, err)

	unpacked, consumed, err := c.Decompress(nil, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
	assert.Equal(t, original, unpacked)
}

func TestCompressor_RoundTrip_NoPatchNoFOR(t *testing.T) {
	c := utlpfor.NewUint32()
	values := []uint32{5, 12, 3, 7, 15, 1, 0, 8}
	original := slices.Clone(values)

	packed, err := c.Compress(utlpfor.NoPatch|utlpfor.NoFOR, nil, values)
	require.NoError(t, err)

	unpacked, consumed, err := c.Decompress(nil, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
	assert.Equal(t, original, unpacked)
}

func TestCompressor_Append(t *testing.T) {
	c := utlpfor.NewUint32()
	block1 := []uint32{1, 2, 3, 4}
	block2 := []uint32{100, 200, 300, 400}
	orig2 := slices.Clone(block2)

	dst, err := c.Compress(utlpfor.Append, nil, block1)
	require.NoError(t, err)
	block1Len := len(dst)

	dst, err = c.Compress(utlpfor.Append, dst, block2)
	require.NoError(t, err)
	require.Greater(t, len(dst), block1Len)

	// Skip past first block using BlockLength, decompress second.
	bl, err := utlpfor.BlockLength(dst)
	require.NoError(t, err)
	assert.Equal(t, block1Len, bl)

	unpacked, _, err := c.Decompress(nil, dst[bl:])
	require.NoError(t, err)
	assert.Equal(t, orig2, unpacked)
}

func TestCompressor_AppendPreservesValues(t *testing.T) {
	c := utlpfor.NewUint32()

	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(1000 + i*7)
	}
	original := slices.Clone(values)

	var dst []byte
	var err error
	for range 3 {
		dst, err = c.Compress(utlpfor.Append, dst, values)
		require.NoError(t, err)
	}
	assert.Equal(t, original, values,
		"compressor.Compress with Append must not modify the values slice")
}

func TestCompressor_NoInPlacePreservesValues(t *testing.T) {
	c := utlpfor.NewUint32()

	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(1000 + i*7)
	}
	original := slices.Clone(values)

	_, err := c.Compress(utlpfor.NoInPlace, nil, values)
	require.NoError(t, err)
	assert.Equal(t, original, values,
		"compressor.Compress with NoInPlace must not modify the values slice")
}

func TestCompressor_AppendWithExceptions(t *testing.T) {
	// Compressor-level Append where packed blocks contain exceptions.
	c := utlpfor.NewUint32()

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

	// Probe actual size for controlled capacity.
	probe, err := c.Compress(0, nil, slices.Clone(block1))
	require.NoError(t, err)

	dst := make([]byte, 0, 2*len(probe))
	dst, err = c.Compress(utlpfor.Append, dst, block1)
	require.NoError(t, err)

	dst, err = c.Compress(utlpfor.Append, dst, block2)
	require.NoError(t, err)

	decoded1, consumed1, err := c.Decompress(nil, dst)
	require.NoError(t, err)
	assert.Equal(t, orig1, decoded1, "block 1 round-trip")

	decoded2, _, err := c.Decompress(nil, dst[consumed1:])
	require.NoError(t, err)
	assert.Equal(t, orig2, decoded2, "block 2 must not contain stale data")
}

func TestCompressor_Get(t *testing.T) {
	c := utlpfor.NewUint32()
	values := []uint32{100, 200, 300, 400, 500}

	packed, err := c.Compress(0, nil, slices.Clone(values))
	require.NoError(t, err)

	for i, want := range values {
		got, err := c.Get(i, packed)
		require.NoError(t, err)
		assert.Equal(t, want, got, "position %d", i)
	}
}

func TestCompressor_Get_Delta(t *testing.T) {
	c := utlpfor.NewUint32()
	values := make([]uint32, 64)
	for i := range values {
		values[i] = uint32(1000 + i*7)
	}
	original := slices.Clone(values)

	packed, err := c.Compress(utlpfor.Delta, nil, values)
	require.NoError(t, err)

	for i, want := range original {
		got, err := c.Get(i, packed)
		require.NoError(t, err)
		assert.Equal(t, want, got, "position %d", i)
	}
}

func TestCompressor_ScratchReuse(t *testing.T) {
	c := utlpfor.NewUint32()

	for round := range 5 {
		values := make([]uint32, 32)
		for i := range values {
			values[i] = uint32(round*1000 + i)
		}
		original := slices.Clone(values)

		packed, err := c.Compress(utlpfor.Delta, nil, values)
		require.NoError(t, err)

		unpacked, _, err := c.Decompress(nil, packed)
		require.NoError(t, err)
		assert.Equal(t, original, unpacked, "round %d", round)
	}
}

func TestCompressor_Decompress_WithPreallocatedTarget(t *testing.T) {
	c := utlpfor.NewUint32()
	values := []uint32{10, 20, 30, 40, 50}
	original := slices.Clone(values)

	packed, err := c.Compress(0, nil, values)
	require.NoError(t, err)

	target := make([]uint32, 0, 128)
	unpacked, consumed, err := c.Decompress(target, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
	assert.Equal(t, original, unpacked)
}

func TestCompressor_FullBlock(t *testing.T) {
	c := utlpfor.NewUint32()
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	original := slices.Clone(values)

	packed, err := c.Compress(utlpfor.Delta, nil, values)
	require.NoError(t, err)

	unpacked, consumed, err := c.Decompress(nil, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
	assert.Equal(t, original, unpacked)
}

// --- Compressor64 tests ---

func TestCompressor64_RoundTrip_Raw(t *testing.T) {
	c := utlpfor.NewUint64()
	values := []uint64{1_000_000_000_000, 1_000_000_000_001, 1_000_000_000_002}
	original := slices.Clone(values)

	packed, err := c.Compress(0, nil, values)
	require.NoError(t, err)
	require.NotEmpty(t, packed)

	unpacked, consumed, err := c.Decompress(nil, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
	assert.Equal(t, original, unpacked)
}

func TestCompressor64_RoundTrip_Delta(t *testing.T) {
	c := utlpfor.NewUint64()
	values := make([]uint64, 128)
	for i := range values {
		values[i] = uint64(1_000_000_000_000 + i*5)
	}
	original := slices.Clone(values)

	packed, err := c.Compress(utlpfor.Delta, nil, values)
	require.NoError(t, err)

	unpacked, consumed, err := c.Decompress(nil, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
	assert.Equal(t, original, unpacked)
}

func TestCompressor64_RoundTrip_SmallValues(t *testing.T) {
	c := utlpfor.NewUint64()
	values := []uint64{10, 20, 30, 40, 50}
	original := slices.Clone(values)

	packed, err := c.Compress(0, nil, values)
	require.NoError(t, err)

	unpacked, consumed, err := c.Decompress(nil, packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), consumed)
	assert.Equal(t, original, unpacked)
}

func TestCompressor64_Get(t *testing.T) {
	c := utlpfor.NewUint64()
	values := []uint64{1_000_000_000_000, 2_000_000_000_000, 3_000_000_000_000}

	packed, err := c.Compress(0, nil, values)
	require.NoError(t, err)

	for i, want := range values {
		got, err := c.Get(i, packed)
		require.NoError(t, err)
		assert.Equal(t, want, got, "position %d", i)
	}
}

func TestCompressor64_ScratchReuse(t *testing.T) {
	c := utlpfor.NewUint64()

	for round := range 5 {
		values := make([]uint64, 32)
		for i := range values {
			values[i] = uint64(round)*1_000_000_000_000 + uint64(i)
		}
		original := slices.Clone(values)

		packed, err := c.Compress(0, nil, values)
		require.NoError(t, err)

		unpacked, _, err := c.Decompress(nil, packed)
		require.NoError(t, err)
		assert.Equal(t, original, unpacked, "round %d", round)
	}
}

func TestCompressor64_Append(t *testing.T) {
	c := utlpfor.NewUint64()
	block1 := []uint64{100, 200, 300}
	block2 := []uint64{1_000_000_000_000, 2_000_000_000_000}
	orig2 := slices.Clone(block2)

	dst, err := c.Compress(utlpfor.Append, nil, block1)
	require.NoError(t, err)
	block1Len := len(dst)

	dst, err = c.Compress(utlpfor.Append, dst, block2)
	require.NoError(t, err)
	require.Greater(t, len(dst), block1Len)

	bl, err := utlpfor.BlockLength(dst)
	require.NoError(t, err)
	assert.Equal(t, block1Len, bl)

	unpacked, _, err := c.Decompress(nil, dst[bl:])
	require.NoError(t, err)
	assert.Equal(t, orig2, unpacked)
}
