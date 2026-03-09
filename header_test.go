package utlpfor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEncodeDecodeHeader_Plain(t *testing.T) {
	h := encodeHeader(128, 8, 0, headerTypeUint32Flag)
	count, bw, intType, excCount, hasExc, hasDelta, hasZZ, hasFOR := decodeHeader(h)
	assert.Equal(t, 128, count)
	assert.Equal(t, 8, bw)
	assert.Equal(t, IntTypeUint32, intType)
	assert.Equal(t, 0, excCount)
	assert.False(t, hasExc)
	assert.False(t, hasDelta)
	assert.False(t, hasZZ)
	assert.False(t, hasFOR)
}

func TestEncodeDecodeHeader_DeltaZigzag(t *testing.T) {
	flags := headerTypeUint32Flag | headerDeltaFlag | headerZigZagFlag
	h := encodeHeader(100, 12, 0, flags)
	count, bw, _, _, _, hasDelta, hasZZ, _ := decodeHeader(h)
	assert.Equal(t, 100, count)
	assert.Equal(t, 12, bw)
	assert.True(t, hasDelta)
	assert.True(t, hasZZ)
}

func TestEncodeDecodeHeader_WithExceptions(t *testing.T) {
	h := encodeHeader(128, 4, 10, headerTypeUint32Flag)
	_, _, _, excCount, hasExc, _, _, _ := decodeHeader(h)
	assert.True(t, hasExc)
	assert.Equal(t, 10, excCount)
}

func TestEncodeDecodeHeader_AllExceptionCounts(t *testing.T) {
	for ec := 1; ec <= 128; ec++ {
		h := encodeHeader(128, 8, ec, headerTypeUint32Flag)
		_, _, _, gotEC, hasExc, _, _, _ := decodeHeader(h)
		assert.True(t, hasExc)
		assert.Equal(t, ec, gotEC, "excCount %d", ec)
	}
}

func TestEncodeDecodeHeader_NoExceptionsExcCountZero(t *testing.T) {
	h := encodeHeader(128, 8, 0, headerTypeUint32Flag)
	_, _, _, excCount, hasExc, _, _, _ := decodeHeader(h)
	assert.False(t, hasExc)
	assert.Equal(t, 0, excCount)
}

func TestEncodeDecodeHeader_AllStepBitWidths(t *testing.T) {
	for _, bw := range stepBitWidths {
		h := encodeHeader(128, bw, 0, headerTypeUint32Flag)
		_, gotBW, _, _, _, _, _, _ := decodeHeader(h)
		assert.Equal(t, bw, gotBW, "step bit width %d", bw)
	}
}

func TestEncodeDecodeHeader_AllCounts(t *testing.T) {
	for count := 0; count <= 128; count++ {
		h := encodeHeader(count, 8, 0, headerTypeUint32Flag)
		gotCount, _, _, _, _, _, _, _ := decodeHeader(h)
		assert.Equal(t, count, gotCount, "count %d", count)
	}
}

func TestDecodeHeader_ZigzagWithoutDelta(t *testing.T) {
	flags := headerTypeUint32Flag | headerZigZagFlag
	h := encodeHeader(128, 8, 0, flags)
	_, _, _, _, _, hasDelta, hasZZ, _ := decodeHeader(h)
	assert.False(t, hasDelta)
	assert.True(t, hasZZ)
}

func TestEncodeDecodeHeader_Uint16Type(t *testing.T) {
	h := encodeHeader(64, 8, 0, headerTypeUint16Flag)
	_, _, intType, _, _, _, _, _ := decodeHeader(h)
	assert.Equal(t, IntTypeUint16, intType)
}

func TestDecodeHeader_ReservedBitsZero(t *testing.T) {
	h := encodeHeader(128, 8, 0, headerTypeUint32Flag)
	reserved := h & headerReservedMask
	assert.Equal(t, uint32(0), reserved, "bits 15-17 and 19-21 must be zero in current impl")
}

func TestDecodeHeader_FORFlag(t *testing.T) {
	h := encodeHeader(128, 8, 0, headerTypeUint32Flag|headerFORFlag)
	_, _, _, _, _, _, _, hasFOR := decodeHeader(h)
	assert.True(t, hasFOR)
}

func TestDecodeHeader_FORFlagBitPosition(t *testing.T) {
	assert.Equal(t, uint32(1<<21), headerFORFlag, "FOR flag must be at bit 21")
}

func TestDecodeHeader_SpecialFlagBitPosition(t *testing.T) {
	assert.Equal(t, uint32(1<<18), headerSpecialFlag, "SPECIAL flag must be at bit 18")
}

func TestDecodeHeader_CombineFlagBitPosition(t *testing.T) {
	assert.Equal(t, uint32(1<<19), headerCombineFlag, "Combine flag must be at bit 19")
}

func TestDecodeHeader_Block256FlagBitPosition(t *testing.T) {
	assert.Equal(t, uint32(1<<20), headerBlock256Flag, "Block256 flag must be at bit 20")
}

func TestDecodeHeader_DeltaFlagBitPosition(t *testing.T) {
	assert.Equal(t, uint32(1<<22), headerDeltaFlag, "Delta flag must be at bit 22")
}

func TestDecodeHeader_ZigZagFlagBitPosition(t *testing.T) {
	assert.Equal(t, uint32(1<<23), headerZigZagFlag, "ZigZag flag must be at bit 23")
}

func TestDecodeHeader_EncodingFlagsContiguous(t *testing.T) {
	assert.Equal(t, headerFORFlag<<1, headerDeltaFlag, "FOR and Delta must be adjacent")
	assert.Equal(t, headerDeltaFlag<<1, headerZigZagFlag, "Delta and ZigZag must be adjacent")
}

func TestEncodeDecodeHeader_AllFlagCombinations(t *testing.T) {
	flagSets := []struct {
		name  string
		flags uint32
	}{
		{"none", headerTypeUint32Flag},
		{"delta", headerTypeUint32Flag | headerDeltaFlag},
		{"zigzag", headerTypeUint32Flag | headerZigZagFlag},
		{"delta+zigzag", headerTypeUint32Flag | headerDeltaFlag | headerZigZagFlag},
		{"FOR", headerTypeUint32Flag | headerFORFlag},
		{"FOR+delta", headerTypeUint32Flag | headerFORFlag | headerDeltaFlag},
		{"FOR+delta+zigzag", headerTypeUint32Flag | headerFORFlag | headerDeltaFlag | headerZigZagFlag},
	}
	for _, fs := range flagSets {
		t.Run(fs.name, func(t *testing.T) {
			h := encodeHeader(128, 8, 0, fs.flags)
			_, _, _, _, _, hasDelta, hasZZ, hasFOR := decodeHeader(h)
			assert.Equal(t, fs.flags&headerDeltaFlag != 0, hasDelta)
			assert.Equal(t, fs.flags&headerZigZagFlag != 0, hasZZ)
			assert.Equal(t, fs.flags&headerFORFlag != 0, hasFOR)
		})
	}
}

func TestPayloadOffset(t *testing.T) {
	tests := []struct {
		name          string
		hasFOR        bool
		hasExceptions bool
		want          int
	}{
		{"no_for_no_exc", false, false, 4},
		{"for_no_exc", true, false, 8},
		{"no_for_exc", false, true, 6},
		{"for_exc", true, true, 10},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := payloadOffset(tt.hasFOR, tt.hasExceptions)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestUtlPayloadBytesLUT(t *testing.T) {
	for bw := 0; bw <= 32; bw++ {
		expected := 0
		if bw > 0 {
			expected = ((bw + 3) / 4) * 64
		}
		assert.Equal(t, expected, utlPayloadBytesLUT[bw],
			"payload size mismatch for bw=%d", bw)
	}
}

func TestEncodeDecodeHeader_4BitBitwidthRoundTrip(t *testing.T) {
	for _, bw := range stepBitWidths {
		for _, ec := range []int{0, 1, 16, 128} {
			h := encodeHeader(128, bw, ec, headerTypeUint32Flag)
			gotCount, gotBW, gotType, gotEC, _, _, _, _ := decodeHeader(h)
			assert.Equal(t, 128, gotCount, "count for bw=%d ec=%d", bw, ec)
			assert.Equal(t, bw, gotBW, "bitWidth for bw=%d ec=%d", bw, ec)
			assert.Equal(t, IntTypeUint32, gotType, "intType for bw=%d ec=%d", bw, ec)
			assert.Equal(t, ec, gotEC, "excCount for bw=%d ec=%d", bw, ec)
		}
	}
}

func TestEncodeDecodeHeader_Bit12_ZeroIn128Mode(t *testing.T) {
	for _, bw := range stepBitWidths {
		h := encodeHeader(128, bw, 0, headerTypeUint32Flag)
		bit12 := (h >> 12) & 0x01
		assert.Equal(t, uint32(0), bit12,
			"bit 12 (5th bw bit) must be zero in 128-block mode for bw=%d", bw)
	}
}

func TestEncodeDecodeHeader_IntTypeAtBits13_14(t *testing.T) {
	h := encodeHeader(128, 8, 0, headerTypeUint32Flag)
	gotType := int((h >> headerTypeShift) & headerTypeMask)
	assert.Equal(t, IntTypeUint32, gotType,
		"intType must be at bits 13-14")
}

func TestEncodeDecodeHeader_ReservedBits15_17_AreZero(t *testing.T) {
	for _, bw := range stepBitWidths {
		h := encodeHeader(128, bw, 0, headerTypeUint32Flag)
		bits1517 := (h >> 15) & 0x07
		assert.Equal(t, uint32(0), bits1517,
			"bits 15-17 must be zero for bw=%d", bw)
	}
}
