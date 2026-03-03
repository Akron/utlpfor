package utlpfor

import (
	"errors"
	"fmt"
)

// ErrInvalidBuffer is returned when the input buffer is nil, empty, or truncated.
var ErrInvalidBuffer = errors.New("utl-pfor: invalid buffer")

// ErrPositionOutOfRange is returned when the requested position exceeds the block count.
var ErrPositionOutOfRange = errors.New("utl-pfor: position out of range")

// ErrInvalidBlockLength is returned when the block length in the header is invalid.
var ErrInvalidBlockLength = errors.New("utl-pfor: invalid block length")

// ErrInvalidFlags is returned when the header flags contain an invalid combination.
var ErrInvalidFlags = errors.New("utl-pfor: invalid header flags")

// ErrNotImplemented is returned by stub functions not yet implemented.
var ErrNotImplemented = errors.New("utl-pfor: not implemented")

// ErrUnsupportedType is returned when the header integer type is not supported.
var ErrUnsupportedType = errors.New("utl-pfor: unsupported integer type")

// ErrOverflow is returned when delta decoding overflows uint32.
type ErrOverflow struct {
	Position int
}

// Error returns the overflow error message.
func (e *ErrOverflow) Error() string {
	return fmt.Sprintf("utl-pfor: delta decode overflow at index %d", e.Position)
}
