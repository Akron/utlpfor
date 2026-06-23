// Package utlpfor implements a UTL-PFOR integer compression codec optimized
// for Go 1.26+ native SIMD.
//
// The codec operates on fixed blocks of up to 128 unsigned 32-bit integers
// using the Unified Transposed Layout (UTL, aka FastLanes) for SIMD-width
// independence.
//
// Four public functions are provided:
//   - PackUint32: encode uint32 values
//   - UnpackUint32: decode values with scratch buffer and consumed-length return
//   - GetUint32: extract a single value by position
//   - BlockLength: compute encoded block size without decoding
//
// The library automatically selects the best SIMD path at startup:
// AVX-512 -> AVX2 -> SSE2 -> scalar fallback. No build-time configuration
// is required.
//
// Reader/SlimReader APIs are intentionally not provided in this package.
//
//go:generate bash -c "go run ./internal/gen | gofmt > simd_spec_amd64.go"
package utlpfor

import (
	"errors"
	"fmt"
)

// Flag controls encoding options passed to PackUint32.
type Flag byte

// Delta indicates that the values should be delta-encoded before packing.
const Delta Flag = 1 << 0

// NoFOR disables Frame-of-Reference analysis during packing.
// When set, the encoder skips the min/max scan and FOR cost evaluation,
// going directly to bitpacking + patching. This improves packing speed
// when it is known in advance that FOR will not be beneficial.
const NoFOR Flag = 1 << 1

// NoPatch disables exception analysis and patching during packing.
// The encoder uses the minimum step bitwidth that fits all values,
// producing zero exceptions. This yields faster packing at the cost
// of potentially larger output when a few outlier values inflate the
// bitwidth. Ideal for dictionary-compressed data where all values
// share a known maximum range. NoPatch also benefits
// random access by guaranteeing no exception data needs to be decoded.
const NoPatch Flag = 1 << 2

// Special bit written in header during packing.
// The current decoder does not interpret this bit yet.
const Special Flag = 1 << 3

// Append makes PackUint32 append the packed block after the existing
// content of dst (starting at len(dst)) instead of overwriting from
// index 0. The returned slice includes the preserved prefix.
// This is a pack-time control flag only, never stored on disk.
//
//	dst = dst[:0]
//	dst, _ = PackUint32(Delta|Append, block1, dst, scratch)
//	dst, _ = PackUint32(Delta|Append, block2, dst, scratch)
//	// dst now contains both blocks concatenated
const Append Flag = 1 << 4

// ErrInvalidBuffer is returned when the input buffer is nil, empty, or truncated.
var ErrInvalidBuffer = errors.New("UTLpfor: invalid buffer")

// ErrPositionOutOfRange is returned when the requested position exceeds the block count.
var ErrPositionOutOfRange = errors.New("UTLpfor: position out of range")

// ErrInvalidBlockLength is returned when the block length in the header is invalid.
var ErrInvalidBlockLength = errors.New("UTLpfor: invalid block length")

// ErrInvalidFlags is returned when the header flags contain an invalid combination.
var ErrInvalidFlags = errors.New("UTLpfor: invalid header flags")

// ErrNotImplemented is returned by stub functions not yet implemented.
var ErrNotImplemented = errors.New("UTLpfor: not implemented")

// ErrUnsupportedType is returned when the header integer type is not supported.
var ErrUnsupportedType = errors.New("UTLpfor: unsupported integer type")

// ErrOverflow is returned when delta decoding overflows uint32.
type ErrOverflow struct {
	Position int
}

// Error returns the overflow error message.
func (e *ErrOverflow) Error() string {
	return fmt.Sprintf("UTLpfor: delta decode overflow at index %d", e.Position)
}
