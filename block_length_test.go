package utlpfor

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// makeValidBlock creates a minimal valid block buffer for testing BlockLength.
func makeValidBlock(count, bitWidth int, _ bool) []byte {
	payloadBytes := utlPayloadBytes(bitWidth)
	total := headerBytes + payloadBytes
	buf := make([]byte, total)
	h := encodeHeader(count, bitWidth, 0, headerTypeUint32Flag)
	bo.PutUint32(buf, h)
	return buf
}

// makeValidBlockWithExceptions creates a block buffer with exception metadata.
func makeValidBlockWithExceptions(count, bitWidth, excCount int) []byte {
	payloadBytes := utlPayloadBytes(bitWidth)
	excIdxSize := excCount
	if excCount > excBitmapThreshold {
		excIdxSize = 16
	}
	svbLen := excCount * 2
	total := headerBytes + svbLenBytes + payloadBytes + excIdxSize + svbLen
	buf := make([]byte, total)
	h := encodeHeader(count, bitWidth, excCount, headerTypeUint32Flag)
	bo.PutUint32(buf, h)
	bo.PutUint16(buf[headerBytes:], uint16(svbLen))
	return buf
}

func TestBlockLength_ValidNoExceptions(t *testing.T) {
	tests := []struct {
		name     string
		count    int
		bitWidth int
		wantLen  int
	}{
		{"bw0", 128, 0, 4},
		{"bw4", 128, 4, 4 + 64},
		{"bw8", 128, 8, 4 + 128},
		{"bw16", 128, 16, 4 + 256},
		{"bw32", 128, 32, 4 + 512},
		{"partial", 50, 8, 4 + 128},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := makeValidBlock(tt.count, tt.bitWidth, false)
			got, err := BlockLength(buf)
			require.NoError(t, err)
			assert.Equal(t, tt.wantLen, got)
		})
	}
}

func TestBlockLength_AllStepBitWidths(t *testing.T) {
	for _, bw := range stepBitWidths {
		t.Run(fmt.Sprintf("bw%d", bw), func(t *testing.T) {
			buf := makeValidBlock(128, bw, false)
			got, err := BlockLength(buf)
			require.NoError(t, err)
			assert.Equal(t, 4+utlPayloadBytes(bw), got)
		})
	}
}

func TestBlockLength_TruncatedBuffer(t *testing.T) {
	buf := makeValidBlock(128, 8, false)
	_, err := BlockLength(buf[:2])
	assert.Error(t, err)
}

func TestBlockLength_WithExceptions(t *testing.T) {
	buf := makeValidBlockWithExceptions(128, 4, 10)
	got, err := BlockLength(buf)
	require.NoError(t, err)
	assert.Greater(t, got, 4+64)

	excIndexSize := 10
	svbLen := 10 * 2
	expected := headerBytes + svbLenBytes + 64 + excIndexSize + svbLen
	assert.Equal(t, expected, got)
}

func TestBlockLength_WithExceptionsBitmapFormat(t *testing.T) {
	buf := makeValidBlockWithExceptions(128, 4, 20)
	got, err := BlockLength(buf)
	require.NoError(t, err)

	svbLen := 20 * 2
	expected := headerBytes + svbLenBytes + 64 + excBitmapThreshold + svbLen
	assert.Equal(t, expected, got)
}

func TestBlockLength_WithExceptionsTruncatedSvbLen(t *testing.T) {
	buf := make([]byte, headerBytes+1)
	h := encodeHeader(128, 8, 5, headerTypeUint32Flag)
	bo.PutUint32(buf, h)
	_, err := BlockLength(buf)
	assert.Error(t, err)
}

func TestBlockLength_CountAboveBlockSizeReturnsError(t *testing.T) {
	buf := make([]byte, headerBytes+512)
	h := encodeHeader(200, 8, 0, headerTypeUint32Flag)
	bo.PutUint32(buf, h)
	_, err := BlockLength(buf)
	assert.Error(t, err)
}

func TestBlockLength_ZeroAllocs(t *testing.T) {
	values := make([]uint32, blockSize)
	packed, err := PackUint32(0, values, nil, nil)
	require.NoError(t, err)

	allocs := testing.AllocsPerRun(100, func() {
		BlockLength(packed)
	})
	assert.Equal(t, float64(0), allocs)
}
