package utlpfor

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMaxBlockLength32_GeneralCase(t *testing.T) {
	// Worst case: bitWidth=28, FOR=uint32 (4 bytes), excCount=128 (all exceptions), bitmap mode
	// header(4) + svbLen(2) + forBase(4) + payload(28*16=448) + excIdx(16) + svbData(544)
	expected := headerBytes + svbLenBytes + forBaseBytes(forWidthU32) +
		utlPayloadBytes(28) + excBitmapThreshold + maxSVBEncodedLen(blockSize)
	assert.Equal(t, 1018, expected, "sanity check formula")
	assert.Equal(t, expected, MaxBlockLength32(0))
}

func TestMaxBlockLength32_NoPatch(t *testing.T) {
	// No exceptions: bitWidth=32, FOR=uint32 (4 bytes)
	// header(4) + forBase(4) + payload(32*16=512)
	expected := headerBytes + forBaseBytes(forWidthU32) + utlPayloadBytes(32)
	assert.Equal(t, 520, expected, "sanity check formula")
	assert.Equal(t, expected, MaxBlockLength32(NoPatch))
}

func TestMaxBlockLength32_NoPatchNoFOR(t *testing.T) {
	// No exceptions, no FOR: bitWidth=32
	// header(4) + payload(32*16=512)
	expected := headerBytes + utlPayloadBytes(32)
	assert.Equal(t, 516, expected, "sanity check formula")
	assert.Equal(t, expected, MaxBlockLength32(NoPatch|NoFOR))
}

func TestMaxBlockLength32_NoFOR(t *testing.T) {
	// Exceptions possible, no FOR base: bitWidth=28
	// header(4) + svbLen(2) + payload(28*16=448) + excIdx(16) + svbData(544)
	expected := headerBytes + svbLenBytes + utlPayloadBytes(28) +
		excBitmapThreshold + maxSVBEncodedLen(blockSize)
	assert.Equal(t, 1014, expected, "sanity check formula")
	assert.Equal(t, expected, MaxBlockLength32(NoFOR))
}

func TestMaxBlockLength32_ActualPackedNeverExceeds_Default(t *testing.T) {
	maxSize := MaxBlockLength32(0)
	rng := rand.New(rand.NewSource(42))
	scratch := make([]uint32, blockSize)

	for trial := range 1000 {
		values := make([]uint32, blockSize)
		switch trial % 4 {
		case 0: // all same
			v := rng.Uint32()
			for i := range values {
				values[i] = v
			}
		case 1: // sequential
			base := rng.Uint32() & 0x00FFFFFF
			for i := range values {
				values[i] = base + uint32(i)
			}
		case 2: // random
			for i := range values {
				values[i] = rng.Uint32()
			}
		case 3: // extreme outliers
			for i := range values {
				values[i] = rng.Uint32() & 0xFF
			}
			for i := range 10 {
				values[i*12%blockSize] = 0xFFFFFFFF - rng.Uint32()&0xF
			}
		}
		packed, err := PackUint32(0, slices.Clone(values), nil, scratch)
		require.NoError(t, err, "trial %d", trial)
		assert.LessOrEqual(t, len(packed), maxSize,
			"trial %d: packed %d bytes > MaxBlockLength32 %d", trial, len(packed), maxSize)
	}
}

func TestMaxBlockLength32_ActualPackedNeverExceeds_NoPatch(t *testing.T) {
	maxSize := MaxBlockLength32(NoPatch)
	rng := rand.New(rand.NewSource(99))
	scratch := make([]uint32, blockSize)

	for trial := range 1000 {
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = rng.Uint32()
		}
		packed, err := PackUint32(NoPatch, slices.Clone(values), nil, scratch)
		require.NoError(t, err, "trial %d", trial)
		assert.LessOrEqual(t, len(packed), maxSize,
			"trial %d: packed %d bytes > MaxBlockLength32(NoPatch) %d", trial, len(packed), maxSize)
	}
}

func TestMaxBlockLength32_ActualPackedNeverExceeds_NoPatchNoFOR(t *testing.T) {
	maxSize := MaxBlockLength32(NoPatch | NoFOR)
	rng := rand.New(rand.NewSource(77))
	scratch := make([]uint32, blockSize)

	for trial := range 1000 {
		values := make([]uint32, blockSize)
		for i := range values {
			values[i] = rng.Uint32()
		}
		packed, err := PackUint32(NoPatch|NoFOR, slices.Clone(values), nil, scratch)
		require.NoError(t, err, "trial %d", trial)
		assert.LessOrEqual(t, len(packed), maxSize,
			"trial %d: packed %d bytes > MaxBlockLength32(NoPatch|NoFOR) %d", trial, len(packed), maxSize)
	}
}

func TestMaxBlockLength32_IgnoresNonSizeFlags(t *testing.T) {
	base := MaxBlockLength32(0)
	assert.Equal(t, base, MaxBlockLength32(Delta))
	assert.Equal(t, base, MaxBlockLength32(Special))
	assert.Equal(t, base, MaxBlockLength32(Append))
}

func TestMaxBlockLength32_CombinedFlagsIgnoreNonSize(t *testing.T) {
	assert.Equal(t, MaxBlockLength32(NoPatch|NoFOR), MaxBlockLength32(Delta|NoPatch|NoFOR))
}
