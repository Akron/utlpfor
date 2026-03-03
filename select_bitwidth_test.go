package utlpfor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSelectBitWidth_AllSameWidth(t *testing.T) {
	for bw := 0; bw <= 32; bw++ {
		values := make([]uint32, 128)
		mask := uint32((1 << bw) - 1)
		if bw == 32 {
			mask = 0xFFFFFFFF
		}
		for i := range values {
			values[i] = mask
		}
		width, excCount := selectBitWidth(values)
		assert.Equal(t, bw, width, "bw=%d", bw)
		assert.Equal(t, 0, excCount, "bw=%d", bw)
	}
}

func TestSelectBitWidth_AllZeros(t *testing.T) {
	values := make([]uint32, 128)
	width, excCount := selectBitWidth(values)
	assert.Equal(t, 0, width)
	assert.Equal(t, 0, excCount)
}

func TestSelectBitWidth_FewExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[0] = 0xFFFFFFFF
	width, excCount := selectBitWidth(values)
	assert.Less(t, width, 32, "should choose lower width with exceptions")
	assert.Greater(t, excCount, 0)
}

func TestSelectBitWidth_ExceptionsVsFullWidth(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = 0x10000000 + uint32(i)
	}
	width, _ := selectBitWidth(values)
	assert.True(t, width >= 28, "all-large values should use high bit width")
}

func TestSelectBitWidth_UTLQuantizationAware(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i) & 0xFF
	}
	values[0] = 0x100
	width, excCount := selectBitWidth(values)
	t.Logf("width=%d, excCount=%d", width, excCount)
}

func TestSelectBitWidth_ZeroPlusExceptions(t *testing.T) {
	values := make([]uint32, 128)
	values[10] = 0xFFFF
	values[50] = 0xFFFFF
	width, excCount := selectBitWidth(values)
	assert.Less(t, width, 20,
		"mostly zeros with few exceptions should choose low width")
	assert.Greater(t, excCount, 0)
}

func TestSelectBitWidth_MatchesMaxBitWidthForUniform(t *testing.T) {
	for bw := 0; bw <= 32; bw++ {
		values := make([]uint32, 128)
		mask := uint32((1 << bw) - 1)
		if bw == 32 {
			mask = 0xFFFFFFFF
		}
		for i := range values {
			values[i] = uint32(i) & mask
		}
		width, excCount := selectBitWidth(values)
		maxBW := maxBitWidth(values)
		assert.Equal(t, maxBW, width,
			"uniform data should match maxBitWidth (bw=%d)", bw)
		assert.Equal(t, 0, excCount, "bw=%d", bw)
	}
}

func TestSelectBitWidth_SingleException(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[42] = 0x100000
	width, excCount := selectBitWidth(values)
	assert.Less(t, width, 21, "should pick narrower width with single outlier")
	assert.Equal(t, 1, excCount)
}
