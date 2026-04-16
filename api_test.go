package utlpfor_test

import (
	"testing"

	utlpfor "github.com/Akron/utlpfor"
	"github.com/stretchr/testify/assert"
)

func TestAPISignatures(t *testing.T) {
	var buf []byte
	var dst []uint32
	var scratch []uint32

	_, _ = utlpfor.BlockLength(buf)
	_, _, _ = utlpfor.UnpackUint32(buf, dst, scratch)
	_, _ = utlpfor.GetUint32(0, buf, scratch)
	_, _ = utlpfor.PackUint32(0, []uint32{}, buf, nil)
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
	assert.Equal(t, byte(1), utlpfor.Delta)
}

func TestNoFORFlagConstant(t *testing.T) {
	assert.Equal(t, byte(2), utlpfor.NoFOR)
	assert.Equal(t, byte(0), utlpfor.Delta&utlpfor.NoFOR, "flags must not overlap")
}

func TestNoPatchFlagConstant(t *testing.T) {
	assert.Equal(t, byte(4), utlpfor.NoPatch)
	assert.Equal(t, byte(0), utlpfor.Delta&utlpfor.NoPatch, "flags must not overlap")
	assert.Equal(t, byte(0), utlpfor.NoFOR&utlpfor.NoPatch, "flags must not overlap")
}
