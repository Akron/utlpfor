package utlpfor

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/mhr3/streamvbyte"
	"github.com/stretchr/testify/assert"
)

func TestSVBDecodeOneInternal_MatchesMhr3FullDecode(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))
	for trial := range 200 {
		excCount := rng.IntN(128) + 1
		highBits := make([]uint32, excCount)
		for i := range highBits {
			highBits[i] = uint32(rng.IntN(0x1000000))
		}

		encoded := streamvbyte.EncodeUint32(highBits, nil)

		fullDecoded := make([]uint32, excCount)
		streamvbyte.DecodeUint32(encoded, excCount,
			&streamvbyte.DecodeOptions[uint32]{Buffer: fullDecoded})

		for idx := range excCount {
			got := svbDecodeOneInternal(encoded, excCount, idx)
			assert.Equal(t, fullDecoded[idx], got,
				"trial %d excCount %d idx %d", trial, excCount, idx)
		}
	}
}

func TestSVBDecodeOneInternal_EdgeCases(t *testing.T) {
	t.Run("single_value", func(t *testing.T) {
		values := []uint32{12345}
		encoded := streamvbyte.EncodeUint32(values, nil)
		got := svbDecodeOneInternal(encoded, 1, 0)
		assert.Equal(t, uint32(12345), got)
	})

	t.Run("four_values_all_slots", func(t *testing.T) {
		values := []uint32{1, 256, 70000, 0xFFFFFFFF}
		encoded := streamvbyte.EncodeUint32(values, nil)

		fullDecoded := make([]uint32, 4)
		streamvbyte.DecodeUint32(encoded, 4,
			&streamvbyte.DecodeOptions[uint32]{Buffer: fullDecoded})

		for idx := range 4 {
			got := svbDecodeOneInternal(encoded, 4, idx)
			assert.Equal(t, fullDecoded[idx], got, "idx %d", idx)
		}
	})

	t.Run("max_exception_count", func(t *testing.T) {
		values := make([]uint32, 128)
		for i := range values {
			values[i] = uint32(i*1000 + 1)
		}
		encoded := streamvbyte.EncodeUint32(values, nil)

		fullDecoded := make([]uint32, 128)
		streamvbyte.DecodeUint32(encoded, 128,
			&streamvbyte.DecodeOptions[uint32]{Buffer: fullDecoded})

		for idx := range 128 {
			got := svbDecodeOneInternal(encoded, 128, idx)
			assert.Equal(t, fullDecoded[idx], got, "idx %d", idx)
		}
	})
}

func TestSVBControlBlockSizeLUT(t *testing.T) {
	for i := range 256 {
		ctrl := byte(i)
		expected := 0
		for s := range 4 {
			expected += int((ctrl>>(s*2))&0x03) + 1
		}
		assert.Equal(t, expected, svbControlBlockSize(ctrl), "ctrl byte 0x%02x", ctrl)
	}
}

func TestSVBControlByteCount(t *testing.T) {
	tests := []struct {
		count    int
		expected int
	}{
		{0, 0},
		{1, 1},
		{3, 1},
		{4, 1},
		{5, 2},
		{8, 2},
		{128, 32},
		{255, 64},
	}
	for _, tt := range tests {
		got := svbControlByteCount(tt.count)
		assert.Equal(t, tt.expected, got, "count=%d", tt.count)
	}
}

func BenchmarkSVBDecodeOneInternal(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i*1000 + 1)
	}
	encoded := streamvbyte.EncodeUint32(values, nil)

	positions := []int{0, 32, 64, 127}
	for _, idx := range positions {
		b.Run(fmt.Sprintf("idx=%d", idx), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				svbDecodeOneInternal(encoded, 128, idx)
			}
		})
	}
}
