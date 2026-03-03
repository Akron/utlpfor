package utlpfor

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
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
