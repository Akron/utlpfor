package utlpfor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEncodeDecodeHeader_Plain(t *testing.T) {
	h := encodeHeader(128, 8, 0, headerTypeUint32Flag)
	count, bw, intType, excCount, forWidth, hasExc, hasDelta, hasZZ := decodeHeader(h)
	assert.Equal(t, 128, count)
	assert.Equal(t, 8, bw)
	assert.Equal(t, IntTypeUint32, intType)
	assert.Equal(t, 0, excCount)
	assert.Equal(t, forWidthNone, forWidth)
	assert.False(t, hasExc)
	assert.False(t, hasDelta)
	assert.False(t, hasZZ)
}

func TestEncodeDecodeHeader_DeltaZigzag(t *testing.T) {
	flags := headerTypeUint32Flag | headerDeltaFlag | headerZigZagFlag
	h := encodeHeader(100, 12, 0, flags)
	count, bw, _, _, _, _, hasDelta, hasZZ := decodeHeader(h)
	assert.Equal(t, 100, count)
	assert.Equal(t, 12, bw)
	assert.True(t, hasDelta)
	assert.True(t, hasZZ)
}

func TestEncodeDecodeHeader_WithExceptions(t *testing.T) {
	h := encodeHeader(128, 4, 10, headerTypeUint32Flag)
	_, _, _, excCount, _, hasExc, _, _ := decodeHeader(h)
	assert.True(t, hasExc)
	assert.Equal(t, 10, excCount)
}

func TestEncodeDecodeHeader_AllExceptionCounts(t *testing.T) {
	for ec := 1; ec <= 128; ec++ {
		h := encodeHeader(128, 8, ec, headerTypeUint32Flag)
		_, _, _, gotEC, _, hasExc, _, _ := decodeHeader(h)
		assert.True(t, hasExc)
		assert.Equal(t, ec, gotEC, "excCount %d", ec)
	}
}

func TestEncodeDecodeHeader_NoExceptionsExcCountZero(t *testing.T) {
	h := encodeHeader(128, 8, 0, headerTypeUint32Flag)
	_, _, _, excCount, _, hasExc, _, _ := decodeHeader(h)
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
	_, _, _, _, _, _, hasDelta, hasZZ := decodeHeader(h)
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
	assert.Equal(t, uint32(0), reserved, "bits 17-18 and 19-20 must be zero in current impl")
}

func TestDecodeHeader_FORWidth(t *testing.T) {
	tests := []struct {
		name     string
		forWidth int
	}{
		{"none", forWidthNone},
		{"u8", forWidthU8},
		{"u16", forWidthU16},
		{"u32", forWidthU32},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags := headerTypeUint32Flag | uint32(tt.forWidth<<forWidthShift)
			h := encodeHeader(128, 8, 0, flags)
			_, _, _, _, gotWidth, _, _, _ := decodeHeader(h)
			assert.Equal(t, tt.forWidth, gotWidth)
		})
	}
}

func TestDecodeHeader_FORWidthBitPosition(t *testing.T) {
	assert.Equal(t, 15, forWidthShift, "FOR width must start at bit 15")
}

func TestDecodeHeader_Block256AllExcFlagBitPosition(t *testing.T) {
	assert.Equal(t, uint32(1<<21), headerBlock256AllExcFlag, "Block 256 all exceptions flag must be at bit 21")
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

func TestDecodeHeader_DeltaZigZagContiguous(t *testing.T) {
	assert.Equal(t, headerDeltaFlag<<1, headerZigZagFlag, "Delta and ZigZag must be adjacent")
}

func TestEncodeDecodeHeader_AllFlagCombinations(t *testing.T) {
	flagSets := []struct {
		name     string
		flags    uint32
		forWidth int
	}{
		{"none", headerTypeUint32Flag, forWidthNone},
		{"delta", headerTypeUint32Flag | headerDeltaFlag, forWidthNone},
		{"zigzag", headerTypeUint32Flag | headerZigZagFlag, forWidthNone},
		{"delta+zigzag", headerTypeUint32Flag | headerDeltaFlag | headerZigZagFlag, forWidthNone},
		{"FOR_u8", headerTypeUint32Flag | uint32(forWidthU8<<forWidthShift), forWidthU8},
		{"FOR_u16", headerTypeUint32Flag | uint32(forWidthU16<<forWidthShift), forWidthU16},
		{"FOR_u32", headerTypeUint32Flag | uint32(forWidthU32<<forWidthShift), forWidthU32},
		{"FOR_u32+delta", headerTypeUint32Flag | uint32(forWidthU32<<forWidthShift) | headerDeltaFlag, forWidthU32},
		{"FOR_u32+delta+zigzag", headerTypeUint32Flag | uint32(forWidthU32<<forWidthShift) | headerDeltaFlag | headerZigZagFlag, forWidthU32},
	}
	for _, fs := range flagSets {
		t.Run(fs.name, func(t *testing.T) {
			h := encodeHeader(128, 8, 0, fs.flags)
			_, _, _, _, gotForWidth, _, hasDelta, hasZZ := decodeHeader(h)
			assert.Equal(t, fs.flags&headerDeltaFlag != 0, hasDelta)
			assert.Equal(t, fs.flags&headerZigZagFlag != 0, hasZZ)
			assert.Equal(t, fs.forWidth, gotForWidth)
		})
	}
}

func TestPayloadOffset(t *testing.T) {
	tests := []struct {
		name          string
		forBaseBytes  int
		hasExceptions bool
		want          int
	}{
		{"no_for_no_exc", 0, false, 4},
		{"u8_for_no_exc", 1, false, 5},
		{"u16_for_no_exc", 2, false, 6},
		{"u32_for_no_exc", 4, false, 8},
		{"no_for_exc", 0, true, 6},
		{"u8_for_exc", 1, true, 7},
		{"u16_for_exc", 2, true, 8},
		{"u32_for_exc", 4, true, 10},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := payloadOffset(tt.forBaseBytes, tt.hasExceptions)
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
		assert.Equal(t, expected, utlPayloadBytes(bw),
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

func TestEncodeDecodeHeader_ReservedBits17_18_AreZero(t *testing.T) {
	for _, bw := range stepBitWidths {
		h := encodeHeader(128, bw, 0, headerTypeUint32Flag)
		bits1718 := (h >> 17) & 0x03
		assert.Equal(t, uint32(0), bits1718,
			"bits 17-18 must be zero for bw=%d", bw)
	}
}

func TestEncodeDecodeHeader_FORWidthAtBits15_16(t *testing.T) {
	for _, fw := range []int{forWidthNone, forWidthU8, forWidthU16, forWidthU32} {
		flags := headerTypeUint32Flag | uint32(fw<<forWidthShift)
		h := encodeHeader(128, 8, 0, flags)
		gotFW := int((h >> forWidthShift) & forWidthMask)
		assert.Equal(t, fw, gotFW, "FOR width must be at bits 15-16 for fw=%d", fw)
	}
}
