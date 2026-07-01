package utlpfor

import (
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// benchSink prevents dead-code elimination of benchmark results.
var benchSink uint32
var benchSink64 uint64

// matrixUint64Config defines one uint64 benchmark configuration.
type matrixUint64Config struct {
	method string // "pac64", "unp64", "get64"
	name   string // e.g. "fit32", "two_block", "for64"
	flag   Flag
	values []uint64
}

func (c matrixUint64Config) benchName() string {
	return c.method + "_" + c.name
}

// matrixConfig defines one benchmark configuration in the matrix.
type matrixConfig struct {
	method       string // "pac", "unp", "get", "mix", "len", "hdr"
	bitWidth     int    // 4, 16, 28
	excCount     int    // 0, 8, 16, 32
	forWidth     int    // 0, 1, 2, 4 (byte count: 0=none, 1=uint8, 2=uint16, 4=uint32)
	useDelta     bool
	useZZ        bool
	noForNoPatch bool
}

// name returns the fixed-width benchmark name for this configuration.
// All methods use the same format for consistent parsing by benchfmt.
func (c matrixConfig) name() string {
	deltaStr, zzStr, nfnpStr := "-delta", "-zz", "-nfnp"
	if c.useDelta {
		deltaStr = "+delta"
	}
	if c.useZZ {
		zzStr = "+zz"
	}
	if c.noForNoPatch {
		nfnpStr = "+nfnp"
	}
	forStr := fmt.Sprintf("%dfor", c.forWidth)
	return fmt.Sprintf("%s_%02dbit_%02dexc_%s_%s_%s_%s",
		c.method, c.bitWidth, c.excCount,
		forStr, deltaStr, zzStr, nfnpStr)
}

// forBaseForWidth returns a FOR base value that produces the requested
// FOR byte width (0=none, 1=uint8, 2=uint16, 4=uint32).
func forBaseForWidth(fw int) uint32 {
	switch fw {
	case 1:
		return 100
	case 2:
		return 10000
	case 4:
		return 1_000_000
	default:
		return 0
	}
}

// generateMatrixData creates 128 uint32 values targeting a specific
// combination of bitwidth, exception count, FOR width, and delta/zigzag.
// Returns (values, flag).
func generateMatrixData(cfg matrixConfig) ([]uint32, Flag) {
	values := make([]uint32, blockSize)
	bw := cfg.bitWidth
	forBase := forBaseForWidth(cfg.forWidth)

	if cfg.useDelta && cfg.useZZ {
		var baseR, step uint32
		switch {
		case bw <= 4:
			baseR, step = 2, 3
		case bw <= 16:
			baseR, step = 100, 200
		default:
			baseR, step = 10000, 20000
		}
		for i := range values {
			posInLane := i / utlLaneCount
			if posInLane%2 == 0 {
				values[i] = forBase + baseR
			} else {
				values[i] = forBase + baseR + step
			}
		}
	} else if cfg.useDelta && !cfg.useZZ {
		step := uint32(1)
		if bw >= 16 {
			step = 10
		}
		if bw >= 28 {
			step = 100
		}
		for i := range values {
			posInLane := i / utlLaneCount
			values[i] = forBase + uint32(posInLane)*step
		}
	} else {
		mask := uint32((1 << bw) - 1)
		if bw >= 32 {
			mask = 0xFFFFFFFF
		}
		for i := range values {
			values[i] = forBase + uint32(i*3)&mask
		}
	}

	if cfg.excCount > 0 {
		// Use moderate exception values so FOR can still be beneficial.
		// The residual after FOR subtraction needs more than bitWidth bits
		// but is not astronomically large (unlike 0x10000000).
		excResidual := uint32(1 << min(bw+4, 31))
		step := max(blockSize/cfg.excCount, 1)
		for e := 0; e < cfg.excCount && e < blockSize; e++ {
			idx := e * step
			if idx >= blockSize {
				idx = blockSize - 1
			}
			values[idx] = forBase + excResidual + uint32(e)
		}
	}

	var flag Flag
	if cfg.useDelta {
		flag = Delta
	}
	if cfg.noForNoPatch {
		flag |= NoFOR | NoPatch
	}
	return values, flag
}

// lenMatrixConfigs returns a reduced set of benchmark configs for
// BlockLength. BlockLength only reads the header (4 bytes) and
// optionally svbLen (2 bytes), so most dimensions are irrelevant.
// The meaningful code paths are: no-exceptions vs with-exceptions
// (sorted positions vs bitmap), and with/without FOR.
func lenMatrixConfigs() []matrixConfig {
	return []matrixConfig{
		{method: "len", bitWidth: 16, excCount: 0, forWidth: 0},
		{method: "len", bitWidth: 16, excCount: 8, forWidth: 0},
		{method: "len", bitWidth: 16, excCount: 32, forWidth: 0},
		{method: "len", bitWidth: 16, excCount: 8, forWidth: 2},
	}
}

// hdrMatrixConfigs returns a reduced set of benchmark configs for Header.
// Header only reads the 4-byte header, so performance is independent of
// most dimensions. A few representative flag combinations are included.
func hdrMatrixConfigs() []matrixConfig {
	return []matrixConfig{
		{method: "hdr", bitWidth: 4, excCount: 0, forWidth: 0},
		{method: "hdr", bitWidth: 16, excCount: 0, forWidth: 0},
		{method: "hdr", bitWidth: 16, excCount: 8, forWidth: 0},
		{method: "hdr", bitWidth: 16, excCount: 8, forWidth: 2},
		{method: "hdr", bitWidth: 16, excCount: 0, forWidth: 0, useDelta: true},
	}
}

// allMatrixConfigs generates all benchmark configurations.
// pac/unp: full matrix (144 each). pac nfnp: 9 configs. get: no FOR (36).
// len: reduced set (4). hdr: reduced set (5).
func allMatrixConfigs() []matrixConfig {
	fullMethods := []string{"pac", "unp"}
	bitWidths := []int{4, 16, 28}
	excCounts := []int{0, 8, 16, 32}
	forWidths := []int{0, 1, 2, 4}

	type deltaZZ struct{ delta, zz bool }
	deltaOpts := []deltaZZ{
		{false, false},
		{true, false},
		{true, true},
	}

	var configs []matrixConfig
	for _, m := range fullMethods {
		for _, bw := range bitWidths {
			for _, exc := range excCounts {
				for _, fw := range forWidths {
					for _, dz := range deltaOpts {
						configs = append(configs, matrixConfig{
							method:   m,
							bitWidth: bw,
							excCount: exc,
							forWidth: fw,
							useDelta: dz.delta,
							useZZ:    dz.zz,
						})
					}
				}
			}
		}
	}
	// Pack configs with combined NoFOR|NoPatch
	for _, bw := range bitWidths {
		for _, dz := range deltaOpts {
			configs = append(configs, matrixConfig{
				method:       "pac",
				bitWidth:     bw,
				noForNoPatch: true,
				useDelta:     dz.delta,
				useZZ:        dz.zz,
			})
		}
	}

	for _, bw := range bitWidths {
		for _, exc := range excCounts {
			for _, dz := range deltaOpts {
				configs = append(configs, matrixConfig{
					method:   "get",
					bitWidth: bw,
					excCount: exc,
					forWidth: 0,
					useDelta: dz.delta,
					useZZ:    dz.zz,
				})
			}
		}
	}
	configs = append(configs, lenMatrixConfigs()...)
	configs = append(configs, hdrMatrixConfigs()...)
	return configs
}

func BenchmarkMatrix(b *testing.B) {
	for _, cfg := range allMatrixConfigs() {
		b.Run(cfg.name(), func(b *testing.B) {
			values, flag := generateMatrixData(cfg)
			original := slices.Clone(values)

			switch cfg.method {
			case "pac":
				dst := make([]byte, 0, 1024)
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(values, original)
					dst, _ = PackUint32(flag, values, dst[:0], nil)
				}

			case "unp":
				work := slices.Clone(original)
				packed, err := PackUint32(flag, work, nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				dst := make([]uint32, blockSize)
				scratch := make([]uint32, blockSize)
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					UnpackUint32(packed, dst, scratch)
				}

			case "get":
				work := slices.Clone(original)
				packed, err := PackUint32(flag, work, nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				var sink uint32
				scratch := make([]uint32, ScratchLen)
				b.ReportAllocs()
				b.SetBytes(4)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for pos := range blockSize {
						v, _ := GetUint32(pos, packed, scratch)
						sink += v
					}
				}
				b.ReportMetric(float64(b.Elapsed())/float64(time.Nanosecond)/float64(b.N*blockSize), "ns/get")
				benchSink = sink

			case "len":
				work := slices.Clone(original)
				packed, err := PackUint32(flag, work, nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					BlockLength(packed)
				}

			case "hdr":
				work := slices.Clone(original)
				packed, err := PackUint32(flag, work, nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					Header(packed)
				}
			}
		})
	}
}

// allMatrixUint64Configs generates uint64 benchmark configurations
// covering the three encoding modes (fit32, two_block, for64) with
// and without delta encoding.
func allMatrixUint64Configs() []matrixUint64Config {
	genFit32 := func() []uint64 {
		v := make([]uint64, blockSize)
		for i := range v {
			v[i] = uint64(i * 7)
		}
		return v
	}
	genTwoBlock := func() []uint64 {
		v := make([]uint64, blockSize)
		for i := range v {
			v[i] = 0x1_0000_0000 + uint64(i*1000)
		}
		return v
	}
	genFOR64 := func() []uint64 {
		v := make([]uint64, blockSize)
		base := uint64(1_700_000_000_000)
		for i := range v {
			v[i] = base + uint64(i*1000)
		}
		return v
	}

	methods := []string{"pac64", "unp64", "get64"}
	type scenario struct {
		name string
		gen  func() []uint64
	}
	scenarios := []scenario{
		{"fit32", genFit32},
		{"two_block", genTwoBlock},
		{"for64", genFOR64},
	}
	flags := []struct {
		suffix string
		flag   Flag
	}{
		{"", 0},
		{"+delta", Delta},
	}

	var configs []matrixUint64Config
	for _, m := range methods {
		for _, s := range scenarios {
			for _, f := range flags {
				configs = append(configs, matrixUint64Config{
					method: m,
					name:   s.name + f.suffix,
					flag:   f.flag,
					values: s.gen(),
				})
			}
		}
	}
	return configs
}

// BenchmarkMatrixUint64 benchmarks uint64 Pack/Unpack/Get across all
// encoding modes (fit32, two_block, for64) with and without delta.
func BenchmarkMatrixUint64(b *testing.B) {
	for _, cfg := range allMatrixUint64Configs() {
		cfg := cfg
		b.Run(cfg.benchName(), func(b *testing.B) {
			original := slices.Clone(cfg.values)
			values := slices.Clone(cfg.values)

			switch cfg.method {
			case "pac64":
				dst := make([]byte, 0, MaxBlockLength64(cfg.flag))
				scratch := make([]uint32, ScratchLen64)
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 8))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(values, original)
					dst, _ = PackUint64(cfg.flag, values, dst[:0], scratch)
				}

			case "unp64":
				packed, err := PackUint64(cfg.flag, slices.Clone(original), nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				dst := make([]uint64, blockSize)
				scratch := make([]uint32, ScratchLen64)
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 8))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					UnpackUint64(packed, dst, scratch)
				}

			case "get64":
				packed, err := PackUint64(cfg.flag, slices.Clone(original), nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				var sink uint64
				scratch := make([]uint32, ScratchLen64)
				b.ReportAllocs()
				b.SetBytes(8)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for pos := range blockSize {
						v, _ := GetUint64(pos, packed, scratch)
						sink += v
					}
				}
				b.ReportMetric(float64(b.Elapsed())/float64(time.Nanosecond)/float64(b.N*blockSize), "ns/get")
				benchSink64 = sink
			}
		})
	}
}

// quickCompareConfigs returns ~20 representative benchmark configurations
// for rapid cross-SIMD-level comparison. Covers pack and unpack across
// different bitwidths, exception counts, FOR widths, and delta/zigzag.
// Includes lightweight random-access probes and mixed pipeline cases.
func quickCompareConfigs() []matrixConfig {
	return []matrixConfig{
		// Pure pack: varying bitwidth, no frills
		{method: "pac", bitWidth: 4, excCount: 0, forWidth: 0, useDelta: false, useZZ: false},
		{method: "pac", bitWidth: 16, excCount: 0, forWidth: 0, useDelta: false, useZZ: false},
		{method: "pac", bitWidth: 28, excCount: 0, forWidth: 0, useDelta: false, useZZ: false},
		// Pack with delta
		{method: "pac", bitWidth: 4, excCount: 0, forWidth: 0, useDelta: true, useZZ: false},
		// Pack with delta + zigzag
		{method: "pac", bitWidth: 4, excCount: 0, forWidth: 0, useDelta: true, useZZ: true},
		// Pack with exceptions
		{method: "pac", bitWidth: 4, excCount: 8, forWidth: 0, useDelta: false, useZZ: false},
		{method: "pac", bitWidth: 16, excCount: 32, forWidth: 0, useDelta: false, useZZ: false},
		// Pack with FOR
		{method: "pac", bitWidth: 16, excCount: 0, forWidth: 2, useDelta: false, useZZ: false},
		// Full pipeline: exceptions + FOR + delta + zigzag
		{method: "pac", bitWidth: 4, excCount: 8, forWidth: 1, useDelta: true, useZZ: false},
		{method: "pac", bitWidth: 16, excCount: 16, forWidth: 4, useDelta: true, useZZ: true},

		// Pure unpack: varying bitwidth
		{method: "unp", bitWidth: 4, excCount: 0, forWidth: 0, useDelta: false, useZZ: false},
		{method: "unp", bitWidth: 16, excCount: 0, forWidth: 0, useDelta: false, useZZ: false},
		{method: "unp", bitWidth: 28, excCount: 0, forWidth: 0, useDelta: false, useZZ: false},
		// Unpack with delta
		{method: "unp", bitWidth: 4, excCount: 0, forWidth: 0, useDelta: true, useZZ: false},
		// Unpack with delta + zigzag
		{method: "unp", bitWidth: 4, excCount: 0, forWidth: 0, useDelta: true, useZZ: true},
		// Unpack with exceptions
		{method: "unp", bitWidth: 4, excCount: 8, forWidth: 0, useDelta: false, useZZ: false},
		{method: "unp", bitWidth: 16, excCount: 32, forWidth: 0, useDelta: false, useZZ: false},
		// Unpack with FOR
		{method: "unp", bitWidth: 16, excCount: 0, forWidth: 2, useDelta: false, useZZ: false},
		// Full pipeline: exceptions + FOR + delta + zigzag
		{method: "unp", bitWidth: 4, excCount: 8, forWidth: 1, useDelta: true, useZZ: false},
		{method: "unp", bitWidth: 16, excCount: 16, forWidth: 4, useDelta: true, useZZ: true},

		// Get-only probes over full block (report includes per-get metric)
		{method: "get", bitWidth: 4, excCount: 0, forWidth: 0, useDelta: false, useZZ: false},
		{method: "get", bitWidth: 16, excCount: 16, forWidth: 0, useDelta: false, useZZ: false},
		{method: "get", bitWidth: 4, excCount: 0, forWidth: 0, useDelta: true, useZZ: false},
		{method: "get", bitWidth: 4, excCount: 8, forWidth: 0, useDelta: true, useZZ: true},

		// Mixed pack+unpack+get on same block (pipeline realism)
		{method: "mix", bitWidth: 16, excCount: 0, forWidth: 0, useDelta: false, useZZ: false},
		{method: "mix", bitWidth: 4, excCount: 8, forWidth: 1, useDelta: true, useZZ: false},

		// NoFOR+NoPatch pack
		{method: "pac", bitWidth: 4, excCount: 0, forWidth: 0, noForNoPatch: true},
		{method: "pac", bitWidth: 16, excCount: 0, forWidth: 0, useDelta: true, useZZ: true, noForNoPatch: true},

		// Header (scalar-only)
		{method: "hdr", bitWidth: 16, excCount: 0, forWidth: 0},
		{method: "hdr", bitWidth: 16, excCount: 8, forWidth: 0},
	}
}

// BenchmarkQuickCompare runs ~20 representative benchmarks for rapid
// cross-SIMD-level comparison. Usage:
//
//	for level in scalar sse2 avx2 avx512; do
//	  echo "=== $level ===";
//	  GOEXPERIMENT=simd UTL_SIMD_LEVEL=$level taskset -c 0-7 go test \
//	    -bench=BenchmarkQuickCompare -benchmem -count=10 -run='^$' -timeout=300s ./...;
//	done
func BenchmarkQuickCompare(b *testing.B) {
	for _, cfg := range quickCompareConfigs() {
		b.Run(cfg.name(), func(b *testing.B) {
			values, flag := generateMatrixData(cfg)
			original := slices.Clone(values)

			switch cfg.method {
			case "pac":
				dst := make([]byte, 0, 1024)
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(values, original)
					dst, _ = PackUint32(flag, values, dst[:0], nil)
				}

			case "unp":
				work := slices.Clone(original)
				packed, err := PackUint32(flag, work, nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				dst := make([]uint32, blockSize)
				scratch := make([]uint32, blockSize)
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					UnpackUint32(packed, dst, scratch)
				}

			case "get":
				work := slices.Clone(original)
				packed, err := PackUint32(flag, work, nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				var sink uint32
				scratch := make([]uint32, ScratchLen)
				b.ReportAllocs()
				b.SetBytes(4)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for pos := range blockSize {
						v, _ := GetUint32(pos, packed, scratch)
						sink += v
					}
				}
				b.ReportMetric(float64(b.Elapsed())/float64(time.Nanosecond)/float64(b.N*blockSize), "ns/get")
				benchSink = sink

			case "mix":
				positions := [4]int{0, 17, 64, 127}
				packDst := make([]byte, 0, 1024)
				unpackDst := make([]uint32, blockSize)
				scratch := make([]uint32, blockSize)
				var sink uint32
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(values, original)
					packed, _ := PackUint32(flag, values, packDst[:0], nil)
					UnpackUint32(packed, unpackDst, scratch)
					for _, pos := range positions {
						v, _ := GetUint32(pos, packed, scratch)
						sink += v
					}
				}
				benchSink = sink

			case "hdr":
				work := slices.Clone(original)
				packed, err := PackUint32(flag, work, nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					Header(packed)
				}
			}
		})
	}
}

type quickUint64Config struct {
	method string // "pac", "unp", "get", "mix"
	name   string
	flag   Flag
	values []uint64
}

func quickCompareUint64Configs() []quickUint64Config {
	genFit32 := func() []uint64 {
		v := make([]uint64, blockSize)
		for i := range v {
			v[i] = uint64(i * 7)
		}
		return v
	}
	genTwoBlock := func() []uint64 {
		v := make([]uint64, blockSize)
		for i := range v {
			v[i] = 0x1_0000_0000 + uint64(i*1000)
		}
		return v
	}
	genFOR64 := func() []uint64 {
		v := make([]uint64, blockSize)
		base := uint64(1_700_000_000_000)
		for i := range v {
			v[i] = base + uint64(i*1000)
		}
		return v
	}

	return []quickUint64Config{
		// Pack-only
		{method: "pac", name: "fit32", flag: 0, values: genFit32()},
		{method: "pac", name: "two_block", flag: 0, values: genTwoBlock()},
		{method: "pac", name: "for64", flag: 0, values: genFOR64()},
		{method: "pac", name: "fit32+delta", flag: Delta, values: genFit32()},

		// Pure unpack
		{method: "unp", name: "fit32", flag: 0, values: genFit32()},
		{method: "unp", name: "two_block", flag: 0, values: genTwoBlock()},
		{method: "unp", name: "for64", flag: 0, values: genFOR64()},
		{method: "unp", name: "fit32+delta", flag: Delta, values: genFit32()},

		// Get-only probes over full block (report includes per-get metric)
		{method: "get", name: "fit32", flag: 0, values: genFit32()},
		{method: "get", name: "two_block", flag: 0, values: genTwoBlock()},
		{method: "get", name: "for64", flag: 0, values: genFOR64()},
		{method: "get", name: "fit32+delta", flag: Delta, values: genFit32()},

		// Mixed pack+unpack+get on same block (pipeline realism)
		{method: "mix", name: "fit32", flag: 0, values: genFit32()},
		{method: "mix", name: "for64", flag: 0, values: genFOR64()},
	}
}

// BenchmarkQuickCompareUint64 runs a small representative set of uint64
// Pack/Unpack/Get benchmarks intended to be included in `make bench-quick`.
func BenchmarkQuickCompareUint64(b *testing.B) {
	for _, cfg := range quickCompareUint64Configs() {
		cfg := cfg
		b.Run(cfg.method+"_"+cfg.name, func(b *testing.B) {
			original := slices.Clone(cfg.values)
			values := slices.Clone(cfg.values)

			switch cfg.method {
			case "pac":
				dst := make([]byte, 0, MaxBlockLength64(cfg.flag))
				scratch := make([]uint32, ScratchLen64)
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 8))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(values, original)
					dst, _ = PackUint64(cfg.flag, values, dst[:0], scratch)
				}

			case "unp":
				packed, err := PackUint64(cfg.flag, slices.Clone(original), nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				dst := make([]uint64, blockSize)
				scratch := make([]uint32, ScratchLen64)
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 8))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					UnpackUint64(packed, dst, scratch)
				}

			case "get":
				packed, err := PackUint64(cfg.flag, slices.Clone(original), nil, nil)
				if err != nil {
					b.Fatalf("pack failed: %v", err)
				}
				var sink uint64
				scratch := make([]uint32, ScratchLen64)
				b.ReportAllocs()
				b.SetBytes(8)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for pos := range blockSize {
						v, _ := GetUint64(pos, packed, scratch)
						sink += v
					}
				}
				b.ReportMetric(float64(b.Elapsed())/float64(time.Nanosecond)/float64(b.N*blockSize), "ns/get")
				benchSink64 = sink

			case "mix":
				positions := [4]int{0, 17, 64, 127}
				packDst := make([]byte, 0, MaxBlockLength64(cfg.flag))
				unpackDst := make([]uint64, blockSize)
				scratch := make([]uint32, ScratchLen64)
				var sink uint64
				b.ReportAllocs()
				b.SetBytes(int64(blockSize * 8))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(values, original)
					packed, _ := PackUint64(cfg.flag, values, packDst[:0], scratch)
					UnpackUint64(packed, unpackDst, scratch)
					for _, pos := range positions {
						v, _ := GetUint64(pos, packed, scratch)
						sink += v
					}
				}
				benchSink64 = sink
			}
		})
	}
}

// TestMatrixDataGeneration validates that the data generator produces
// packed blocks with the expected header flags.
func TestMatrixDataGeneration(t *testing.T) {
	bitWidths := []int{4, 16, 28}
	excCounts := []int{0, 8, 16, 32}
	forWidths := []int{0, 1, 2, 4}

	for _, bw := range bitWidths {
		for _, exc := range excCounts {
			for _, fw := range forWidths {
				for _, useDelta := range []bool{false, true} {
					for _, useZZ := range []bool{false, true} {
						if useZZ && !useDelta {
							continue
						}
						cfg := matrixConfig{
							method:   "pac",
							bitWidth: bw,
							excCount: exc,
							forWidth: fw,
							useDelta: useDelta,
							useZZ:    useZZ,
						}
						t.Run(cfg.name(), func(t *testing.T) {
							values, flag := generateMatrixData(cfg)
							work := slices.Clone(values)
							packed, err := PackUint32(flag, work, nil, nil)
							require.NoError(t, err)
							require.GreaterOrEqual(t, len(packed), headerBytes)

							header := bo.Uint32(packed)
							_, _, _, hExc, hFORWidth, _, hDelta, hZZ, _, _ := decodeHeader(header)

							if useDelta {
								assert.True(t, hDelta, "expected delta flag")
							}

							if fw > 0 {
								if hFORWidth == 0 {
									t.Logf("note: target forWidth=%d bytes, but FOR was not selected", fw)
								}
							} else {
								if hFORWidth > 0 {
									t.Logf("note: no FOR requested, but cost model selected forWidth=%d", hFORWidth)
								}
							}

							if useZZ {
								assert.True(t, hZZ,
									"zigzag expected but not triggered; saw-like data did not produce negative deltas")
							}

							if hExc != exc {
								t.Logf("note: target exc=%d, actual=%d "+
									"(cost model chose differently)", exc, hExc)
							}
						})
					}
				}
			}
		}
	}

	// NoFOR+NoPatch configurations
	for _, bw := range bitWidths {
		for _, useDelta := range []bool{false, true} {
			for _, useZZ := range []bool{false, true} {
				if useZZ && !useDelta {
					continue
				}
				cfg := matrixConfig{
					method:       "pac",
					bitWidth:     bw,
					noForNoPatch: true,
					useDelta:     useDelta,
					useZZ:        useZZ,
				}
				t.Run(cfg.name(), func(t *testing.T) {
					values, flag := generateMatrixData(cfg)
					work := slices.Clone(values)
					packed, err := PackUint32(flag, work, nil, nil)
					require.NoError(t, err)
					require.GreaterOrEqual(t, len(packed), headerBytes)

					header := bo.Uint32(packed)
					_, _, _, hExc, hFORWidth, _, hDelta, hZZ, _, _ := decodeHeader(header)

					assert.Equal(t, 0, hFORWidth,
						"NoFOR+NoPatch must produce no FOR")
					assert.Equal(t, 0, hExc,
						"NoFOR+NoPatch must produce no exceptions")
					if useDelta {
						assert.True(t, hDelta, "expected delta flag")
					}
					if useZZ {
						assert.True(t, hZZ, "expected zigzag flag")
					}
				})
			}
		}
	}

	// Header configurations
	for _, cfg := range hdrMatrixConfigs() {
		t.Run(cfg.name(), func(t *testing.T) {
			values, flag := generateMatrixData(cfg)
			work := slices.Clone(values)
			packed, err := PackUint32(flag, work, nil, nil)
			require.NoError(t, err)
			require.GreaterOrEqual(t, len(packed), headerBytes)

			count, bitWidth, _, hasDelta, _, _, _, err := Header(packed)
			require.NoError(t, err)
			assert.Equal(t, blockSize, count)
			assert.Greater(t, bitWidth, 0)
			if cfg.useDelta {
				assert.True(t, hasDelta, "expected delta flag")
			}
		})
	}
}
