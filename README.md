# UTL-PFOR

UTL-PFOR is an integer compression library for Go using native SIMD support.

It is the successor to [fastpfor-go](https://github.com/Akron/fastpfor-go),
which uses PFOR [1] based on the [FastPFOR](https://github.com/fast-pack/FastPFOR) [2]
library with SSE2 assembly.

UTL-PFOR replaces the SSE2-only lane layout for bitpacking with the
[FastLanes](https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf) [4]
*Unified Transposed Layout* (UTL): 16 lanes x 8 values in 64-byte super-words.
This single wire format works across SSE2, AVX2, and AVX-512 without
data transposition.

In addition to the `fastpfor` compression scheme, UTL-PFOR uses real *Frame-of-Reference* (the `for` in `pfor`) encoding,
storing a minimum value per block and compressing only the residuals, if beneficial.
Outliers, that would harm bitpacking, are stored as exceptions, that are patched on decompression (the `p` in `pfor`).
Exceptions are encoded using [StreamVByte](https://github.com/mhr3/streamvbyte) [3],
a variable-byte encoding scheme optimized for SIMD (specifically SSE2).
*Delta-Encoding* allows to only store the difference between values.
*Zigzag-Encoding* is used to encode negative values after Delta-Encoding.

## API

```go
func PackUint32(flag Flag, values []uint32, dst []byte, scratch []uint32) ([]byte, error)
func UnpackUint32(src []byte, values []uint32, scratch []uint32) ([]uint32, int, error)
func GetUint32(pos int, src []byte, scratch []uint32) (uint32, error)
func BlockLength(src []byte) (int, error)
func MaxBlockLength32(flag Flag) int
func Header(src []byte) (count, bitWidth, excCount int, hasDelta, hasFOR, hasZigZag, hasSpecial bool, err error)
```

And for `[]uint64` handling:

```go
func PackUint64(flag Flag, values []uint64, dst []byte, scratch []uint32) ([]byte, error)
func UnpackUint64(src []byte, values []uint64, scratch []uint32) ([]uint64, int, error)
func GetUint64(pos int, src []byte, scratch []uint32) (uint64, error)
func MaxBlockLength64(flag Flag) int
```

The uint64 functions use the same block format and reuse the entire uint32
pipeline internally. 64-bit values are either range-reduced to 32 bits
(via 64-bit *frame of reference*) or split into lower/upper 32-bit halves encoded
as two consecutive uint32 sub-blocks.

### Compressor

The compressor objects own a reusable scratch buffer
and provide a simplified API for compression, decompression,
and random access. Callers no longer need to allocate or
pass scratch buffers.

```go
// uint32
c := utlpfor.NewUint32()
packed, err := c.Compress(utlpfor.Delta, dst, values)
unpacked, consumed, err := c.Decompress(dst, packed)
val, err := c.Get(pos, packed)

// uint64
c64 := utlpfor.NewUint64()
packed, err := c64.Compress(utlpfor.Delta, dst, values)
unpacked, consumed, err := c64.Decompress(dst, packed)
val, err := c64.Get(pos, packed)
```

A single compressor handles both compression and decompression. It is
not safe for concurrent use; each goroutine should create its own.

### Flag Constants

| Flag |  Description |
|------|-------------|
| `Delta` | Delta-encode values before packing |
| `NoFOR` | Skip Frame-of-Reference analysis |
| `NoPatch` | Skip exception analysis (no patching) |
| `Special` | Set the SPECIAL header bit |
| `Append` | Append packed block after existing dst content (implies `NoInPlace`) |
| `NoInPlace` | Guarantee the input values slice is not modified |

Flags can be combined with bitwise OR, e.g. `Delta | NoFOR`.

The `Append` and `NoInPlace` flags are pack-time control flags only and are
never stored in the on-disk block header.

When `Append` is set, the packed block is written after the existing content instead of overwriting from index 0.

By default, `PackUint32` may modify the input values slice in-place during FOR subtraction and delta encoding. When `NoInPlace` is set (or implied
by `Append`), the library uses a scratch work buffer instead, leaving
the original values untouched.

For zero-allocation operation with `NoInPlace`, provide a scratch buffer
with capacity >= `ScratchLenNoInPlace` (256 elements). If scratch is
smaller, the library allocates internally.

## Requirements

This library uses Go's native SIMD support (`simd/archsimd`) introduced
in Go 1.26 (via `GOEXPERIMENT=simd`).

Runtime dispatch selects the best path at startup:
AVX-512 > AVX2 > SSE2 > scalar fallback.

- **Go 1.26+** with `GOEXPERIMENT=simd` for SIMD acceleration
- Scalar fallback works without the experiment flag on any architecture
- **Recommended**: Use `gotip` (Go 1.27-devel) for optimal AVX2/AVX-512
  performance until Go 1.27 is released. Go 1.26.x compilers have a
  known (and [fixed](https://github.com/golang/go/commit/aa80d7a7e6bf97aa27a74cc5056ef270a2a0c2f4))
  issue generating suboptimal code, resulting in performance degradation on AVX2 and AVX-512
  bitpacking kernels.

## Quick Start

```bash
# Scalar path (any Go 1.26+)
gotip test ./...

# With SIMD acceleration
GOEXPERIMENT=simd gotip test ./...

# Benchmarks
GOEXPERIMENT=simd gotip test -bench=. -benchmem -count=5 ./...
```

## Data Format

A [Kaitai Struct](https://kaitai.io/) definition file is part of this repository.


## Disclaimer

This library was developed with AI assistance (Claude Opus 4.6 and Codex 5.3).

## Literature

- [1] Zukowski, M., Heman, S., Nes, N., & Boncz, P. (2006). *Super-Scalar RAM-CPU Cache Compression*.
    22nd International Conference on Data Engineering (ICDE'06), 59-59.
    https://doi.org/10.1109/ICDE.2006.150
- [2] Lemire, D., & Boytsov, L. (2015). *Decoding billions of integers per second through vectorization*.
    Software: Practice and Experience, 45(1), 1-29. https://doi.org/10.1002/spe.2203
- [3] Lemire, D., Kurz, N. & Rupp, C. (2018). *Stream VByte: Faster Byte-Oriented Integer Compression*, Information Processing Letters 130.
- [4] Afroozeh, A. & Muehlbauer, P. (2023). *FastLanes: An SIMD-Friendly Layout for Analytic Databases*.
    Proceedings of the VLDB Endowment, 16(9), 2132-2144.
    https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf
