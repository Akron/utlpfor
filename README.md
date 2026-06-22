# UTL-PFOR

UTL-PFOR is an integer compression library for Go using native SIMD.

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

## Status

Work in progress. The library is functional but not yet final.

## Requirements

This library uses Go's native SIMD support (`simd/archsimd`) introduced
in Go 1.26 (via `GOEXPERIMENT=simd`).

Runtime dispatch selects the best path at startup:
AVX-512 > AVX2 > SSE2 > scalar fallback.

- **Go 1.26+** with `GOEXPERIMENT=simd` for SIMD acceleration
- Scalar fallback works without the experiment flag on any architecture
- **Recommended**: Use `gotip` (Go 1.27-devel) for optimal AVx2/AVx-512
  performance until Go 1.27 is released. Go 1.26.x compilers have a
  known (and [fixed](https://github.com/golang/go/commit/aa80d7a7e6bf97aa27a74cc5056ef270a2a0c2f4))
  issue generating suboptimal code, resulting in performance degradation on AVX2 and AVX-512
  bitpacking kernels.

## API

```go
func PackUint32(flag PackFlag, values []uint32, dst []byte, scratch []uint32) ([]byte, error)
func UnpackUint32(src []byte, values []uint32, scratch []uint32) ([]uint32, int, error)
func GetUint32(pos int, src []byte, scratch []uint32) (uint32, error)
func BlockLength(src []byte) (int, error)
func Header(src []byte) (count, bitWidth, excCount int, hasDelta, hasFOR, hasZigZag, hasSpecial bool, err error)
```

### PackFlag Constants

| Flag |  Description |
|------|-------------|
| `Delta` | Delta-encode values before packing |
| `NoFOR` | Skip Frame-of-Reference analysis |
| `NoPatch` | Skip exception analysis (no patching) |
| `Special` | Set the SPECIAL header bit |

Flags can be combined with bitwise OR, e.g. `Delta | NoFOR`.

## Quick Start

```bash
# Scalar path (any Go 1.26+)
gotip test ./...

# With SIMD acceleration
GOEXPERIMENT=simd gotip test ./...

# Benchmarks
GOEXPERIMENT=simd gotip test -bench=. -benchmem -count=5 ./...
```

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
