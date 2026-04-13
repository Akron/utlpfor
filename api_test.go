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
	_, _, _ = utlpfor.UnpackUint32(dst, scratch, buf)
	_, _ = utlpfor.GetUint32(0, buf)
	_, _ = utlpfor.PackUint32(0, buf, nil, []uint32{})
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
	_, err := utlpfor.GetUint32(0, nil)
	assert.Error(t, err)
}

func TestPackUint32_ReturnsError(t *testing.T) {
	_, err := utlpfor.PackUint32(0, nil, nil, []uint32{})
	assert.Error(t, err)
}

func TestDeltaFlagConstant(t *testing.T) {
	assert.Equal(t, byte(1), utlpfor.Delta)
}
