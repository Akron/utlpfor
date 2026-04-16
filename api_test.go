package utlpfor_test

import (
	"slices"
	"testing"

	utlpfor "github.com/Akron/utlpfor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAPISignatures(t *testing.T) {
	var buf []byte
	var dst []uint32
	var scratch []uint32

	_, _ = utlpfor.BlockLength(buf)
	_, _, _ = utlpfor.UnpackUint32(buf, dst, scratch)
	_, _ = utlpfor.GetUint32(0, buf, scratch)
	_, _ = utlpfor.PackUint32(0, []uint32{}, buf, nil)
	_, _, _, _, _, _, _, _ = utlpfor.Header(buf)
}

func TestBlockLength_EmptyBuffer(t *testing.T) {
	_, err := utlpfor.BlockLength(nil)
	assert.Error(t, err)
}

func TestBlockLength_ShortBuffer(t *testing.T) {
	_, err := utlpfor.BlockLength([]byte{0x01, 0x02})
	assert.Error(t, err)
}

func TestUnpackUint32_EmptyBuffer(t *testing.T) {
	_, _, err := utlpfor.UnpackUint32(nil, nil, nil)
	assert.Error(t, err)
}

func TestGetUint32_EmptyBuffer(t *testing.T) {
	_, err := utlpfor.GetUint32(0, nil, nil)
	assert.Error(t, err)
}

func TestPackUint32_ReturnsError(t *testing.T) {
	_, err := utlpfor.PackUint32(0, []uint32{}, nil, nil)
	assert.Error(t, err)
}

func TestDeltaFlagConstant(t *testing.T) {
	assert.Equal(t, utlpfor.Flag(1), utlpfor.Delta)
}

func TestNoFORFlagConstant(t *testing.T) {
	assert.Equal(t, utlpfor.Flag(2), utlpfor.NoFOR)
	assert.Equal(t, utlpfor.Flag(0), utlpfor.Delta&utlpfor.NoFOR, "flags must not overlap")
}

func TestNoPatchFlagConstant(t *testing.T) {
	assert.Equal(t, utlpfor.Flag(4), utlpfor.NoPatch)
	assert.Equal(t, utlpfor.Flag(0), utlpfor.Delta&utlpfor.NoPatch, "flags must not overlap")
	assert.Equal(t, utlpfor.Flag(0), utlpfor.NoFOR&utlpfor.NoPatch, "flags must not overlap")
}

func TestSpecialFlagConstant(t *testing.T) {
	assert.Equal(t, utlpfor.Flag(8), utlpfor.Special)
	assert.Equal(t, utlpfor.Flag(0), utlpfor.Delta&utlpfor.Special, "flags must not overlap")
	assert.Equal(t, utlpfor.Flag(0), utlpfor.NoFOR&utlpfor.Special, "flags must not overlap")
	assert.Equal(t, utlpfor.Flag(0), utlpfor.NoPatch&utlpfor.Special, "flags must not overlap")
}

func TestHeader_NilBuffer(t *testing.T) {
	_, _, _, _, _, _, _, err := utlpfor.Header(nil)
	assert.ErrorIs(t, err, utlpfor.ErrInvalidBuffer)
}

func TestHeader_ShortBuffer(t *testing.T) {
	_, _, _, _, _, _, _, err := utlpfor.Header([]byte{0x01, 0x02})
	assert.ErrorIs(t, err, utlpfor.ErrInvalidBuffer)
}

func TestHeader_PlainBlock(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i % 16)
	}
	packed, err := utlpfor.PackUint32(0, slices.Clone(values), nil, nil)
	require.NoError(t, err)

	count, bitWidth, excCount, hasDelta, hasFOR, hasZigZag, hasSpecial, err := utlpfor.Header(packed)
	require.NoError(t, err)
	assert.Equal(t, 128, count)
	assert.Greater(t, bitWidth, 0)
	assert.Equal(t, 0, excCount)
	assert.False(t, hasDelta)
	assert.False(t, hasFOR)
	assert.False(t, hasZigZag)
	assert.False(t, hasSpecial)
}

func TestHeader_DeltaBlock(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 10)
	}
	packed, err := utlpfor.PackUint32(utlpfor.Delta, slices.Clone(values), nil, nil)
	require.NoError(t, err)

	_, _, _, hasDelta, _, _, _, err := utlpfor.Header(packed)
	require.NoError(t, err)
	assert.True(t, hasDelta)
}

func TestHeader_WithExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i % 16)
	}
	values[10] = 0x10000
	values[50] = 0x1000000

	packed, err := utlpfor.PackUint32(0, slices.Clone(values), nil, nil)
	require.NoError(t, err)

	_, _, excCount, _, _, _, _, err := utlpfor.Header(packed)
	require.NoError(t, err)
	assert.Greater(t, excCount, 0)
}

func TestHeader_SpecialFlag(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i % 16)
	}
	packed, err := utlpfor.PackUint32(utlpfor.Special, slices.Clone(values), nil, nil)
	require.NoError(t, err)

	_, _, _, _, _, _, hasSpecial, err := utlpfor.Header(packed)
	require.NoError(t, err)
	assert.True(t, hasSpecial)
}

func TestHeader_SpecialFlagWithDelta(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 10)
	}
	packed, err := utlpfor.PackUint32(utlpfor.Delta|utlpfor.Special, slices.Clone(values), nil, nil)
	require.NoError(t, err)

	_, _, _, hasDelta, _, _, hasSpecial, err := utlpfor.Header(packed)
	require.NoError(t, err)
	assert.True(t, hasDelta)
	assert.True(t, hasSpecial)
}

func TestHeader_FORBlock(t *testing.T) {
	values := make([]uint32, 128)
	base := uint32(100000)
	for i := range values {
		values[i] = base + uint32(i%16)
	}
	packed, err := utlpfor.PackUint32(0, slices.Clone(values), nil, nil)
	require.NoError(t, err)

	_, _, _, _, hasFOR, _, _, err := utlpfor.Header(packed)
	require.NoError(t, err)
	assert.True(t, hasFOR)
}

func TestHeader_ConsistentWithBlockLength(t *testing.T) {
	values := make([]uint32, 100)
	for i := range values {
		values[i] = uint32(i * 3)
	}
	packed, err := utlpfor.PackUint32(0, slices.Clone(values), nil, nil)
	require.NoError(t, err)

	count, _, _, _, _, _, _, err := utlpfor.Header(packed)
	require.NoError(t, err)
	assert.Equal(t, 100, count)

	blkLen, err := utlpfor.BlockLength(packed)
	require.NoError(t, err)
	assert.Equal(t, len(packed), blkLen)
}
