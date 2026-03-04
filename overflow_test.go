package utlpfor

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestErrOverflow_Type(t *testing.T) {
	var err error = &ErrOverflow{Position: 5}
	var overflow *ErrOverflow
	assert.True(t, errors.As(err, &overflow))
	assert.Equal(t, 5, overflow.Position)
}

func TestErrOverflow_Message(t *testing.T) {
	err := &ErrOverflow{Position: 42}
	assert.Contains(t, err.Error(), "42")
}

func TestErrOverflow_IsNotErrInvalidBuffer(t *testing.T) {
	var err error = &ErrOverflow{Position: 1}
	assert.False(t, errors.Is(err, ErrInvalidBuffer))
}

func TestErrOverflow_MessagePrefix(t *testing.T) {
	err := &ErrOverflow{Position: 0}
	assert.Contains(t, err.Error(), "UTLpfor:")
}

func TestErrOverflow_CorruptDeltaBlock(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	expected := append([]uint32(nil), values...)
	packed, err := PackUint32(Delta, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestErrOverflow_DescendingDelta_NoOverflow(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(10000 - i*50)
	}
	expected := append([]uint32(nil), values...)
	packed, err := PackUint32(Delta, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}
