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
package utlpfor
