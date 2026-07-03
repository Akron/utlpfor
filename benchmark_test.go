package utlpfor

import (
	"fmt"
	"slices"
	"testing"
)

func BenchmarkPackUint32(b *testing.B) {
	for _, bw := range []int{4, 8, 12, 16, 24, 32} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			dst := make([]byte, 0, 1024)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dst, _ = PackUint32(0, values, dst[:0], nil)
			}
		})
	}
}

func BenchmarkUnpackUint32(b *testing.B) {
	for _, bw := range []int{4, 8, 12, 16, 24, 32} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			packed, _ := PackUint32(0, values, nil, nil)
			dst := make([]uint32, blockSize)
			scratch := make([]uint32, blockSize)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				UnpackUint32(packed, dst, scratch)
			}
		})
	}
}

func BenchmarkGetUint32(b *testing.B) {
	for _, bw := range []int{4, 8, 16, 32} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			packed, _ := PackUint32(0, values, nil, nil)

			scratch := make([]uint32, ScratchLen)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GetUint32(i%blockSize, packed, scratch)
			}
		})
	}
}

func BenchmarkGetUint64(b *testing.B) {
	cases := []struct {
		name   string
		flag   Flag
		values []uint64
	}{
		{"fit32", 0, func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(i * 7)
			}
			return v
		}()},
		{"two_block", 0, func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0x1_0000_0000 + uint64(i*1000)
			}
			return v
		}()},
		{"for64", 0, func() []uint64 {
			v := make([]uint64, blockSize)
			base := uint64(1_700_000_000_000)
			for i := range v {
				v[i] = base + uint64(i*1000)
			}
			return v
		}()},
		{"fit32+delta", Delta, func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(i * 7)
			}
			return v
		}()},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			packed, _ := PackUint64(tc.flag, slices.Clone(tc.values), nil, nil)
			scratch := make([]uint32, ScratchLen64)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GetUint64(i%blockSize, packed, scratch)
			}
		})
	}
}

func BenchmarkBlockLength(b *testing.B) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, _ := PackUint32(0, values, nil, nil)

	b.ReportAllocs()

	for b.Loop() {
		BlockLength(packed)
	}
}

func BenchmarkPackWithExceptions(b *testing.B) {
	for _, excCount := range []int{1, 5, 10, 30, 64, 128} {
		b.Run(fmt.Sprintf("exc%d", excCount), func(b *testing.B) {
			values := make([]uint32, blockSize)
			for i := range values {
				values[i] = uint32(i)
			}
			step := max(blockSize/excCount, 1)
			for i := 0; i < excCount && i < blockSize; i++ {
				idx := i * step
				if idx >= blockSize {
					idx = blockSize - 1
				}
				values[idx] = 0x10000000 + uint32(i)
			}
			dst := make([]byte, 0, 1024)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dst, _ = PackUint32(0, values, dst[:0], nil)
			}
		})
	}
}

func BenchmarkUnpackWithExceptions(b *testing.B) {
	for _, excCount := range []int{1, 5, 10, 30, 64, 128} {
		b.Run(fmt.Sprintf("exc%d", excCount), func(b *testing.B) {
			values := make([]uint32, blockSize)
			for i := range values {
				values[i] = uint32(i)
			}
			step := max(blockSize/excCount, 1)
			for i := 0; i < excCount && i < blockSize; i++ {
				idx := i * step
				if idx >= blockSize {
					idx = blockSize - 1
				}
				values[idx] = 0x10000000 + uint32(i)
			}
			packed, _ := PackUint32(0, values, nil, nil)
			dst := make([]uint32, blockSize)
			scratch := make([]uint32, blockSize)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				UnpackUint32(packed, dst, scratch)
			}
		})
	}
}

func BenchmarkPackDeltaUint32(b *testing.B) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	dst := make([]byte, 0, 1024)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		clone := slices.Clone(values)
		dst, _ = PackUint32(Delta, clone, dst[:0], nil)
	}
}

func BenchmarkUnpackDeltaUint32(b *testing.B) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	clone := slices.Clone(values)
	packed, _ := PackUint32(Delta, clone, nil, nil)
	dst := make([]uint32, blockSize)
	scratch := make([]uint32, blockSize)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

// --- Cross-Repo Comparison Benchmarks ---
// These use the same data patterns as fastpfor-go benchmarks for direct comparison.

func BenchmarkPackSequential(b *testing.B) {
	data := genSequential(blockSize)
	dst := make([]byte, 0, 1024)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		dst, _ = PackUint32(0, data, dst[:0], nil)
	}
}

func BenchmarkUnpackSequential(b *testing.B) {
	packed, _ := PackUint32(0, genSequential(blockSize), nil, nil)
	dst := make([]uint32, blockSize)
	scratch := make([]uint32, blockSize)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		_, _, _ = UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkPackDeltaMonotonic(b *testing.B) {
	source := genMonotonic(blockSize)
	data := make([]uint32, blockSize)
	dst := make([]byte, 0, 1024)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		copy(data, source)
		dst, _ = PackUint32(Delta, data, dst[:0], nil)
	}
}

func BenchmarkUnpackDeltaMonotonic(b *testing.B) {
	source := slices.Clone(genMonotonic(blockSize))
	packed, _ := PackUint32(Delta, source, nil, nil)
	dst := make([]uint32, blockSize)
	scratch := make([]uint32, blockSize)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkPackDeltaMixed(b *testing.B) {
	source := genMixed(blockSize)
	data := make([]uint32, blockSize)
	dst := make([]byte, 0, 1024)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		copy(data, source)
		dst, _ = PackUint32(Delta, data, dst[:0], nil)
	}
}

func BenchmarkUnpackDeltaMixed(b *testing.B) {
	source := slices.Clone(genMixed(blockSize))
	packed, _ := PackUint32(Delta, source, nil, nil)
	dst := make([]uint32, blockSize)
	scratch := make([]uint32, blockSize)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkPackWithSmallExceptions(b *testing.B) {
	data := genDataWithSmallExceptions()
	dst := make([]byte, 0, 1024)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		dst, _ = PackUint32(0, data, dst[:0], nil)
	}
}

func BenchmarkUnpackWithSmallExceptions(b *testing.B) {
	packed, _ := PackUint32(0, genDataWithSmallExceptions(), nil, nil)
	dst := make([]uint32, blockSize)
	scratch := make([]uint32, blockSize)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkPackWithLargeExceptions(b *testing.B) {
	data := genDataWithLargeExceptions()
	dst := make([]byte, 0, 1024)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		dst, _ = PackUint32(0, data, dst[:0], nil)
	}
}

func BenchmarkUnpackWithLargeExceptions(b *testing.B) {
	packed, _ := PackUint32(0, genDataWithLargeExceptions(), nil, nil)
	dst := make([]uint32, blockSize)
	scratch := make([]uint32, blockSize)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))

	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

// --- Exception Penalty Calibration ---

func BenchmarkExceptionPenaltyCalibration(b *testing.B) {
	for _, excCount := range []int{0, 1, 2, 4, 8, 16, 32, 64, 128} {
		b.Run(fmt.Sprintf("exc%d", excCount), func(b *testing.B) {
			values := make([]uint32, blockSize)
			for i := range values {
				values[i] = uint32(i)
			}
			step := blockSize
			if excCount > 0 {
				step = blockSize / excCount
			}
			if step < 1 {
				step = 1
			}
			for i := 0; i < excCount && i < blockSize; i++ {
				idx := i * step
				if idx >= blockSize {
					idx = blockSize - 1
				}
				values[idx] = 0x10000000 + uint32(i)
			}
			packed, _ := PackUint32(0, values, nil, nil)
			dst := make([]uint32, blockSize)
			scratch := make([]uint32, blockSize)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ReportMetric(float64(len(packed)), "packed_bytes")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				UnpackUint32(packed, dst, scratch)
			}
		})
	}
}

// --- Compression Ratio Benchmarks ---

func BenchmarkCompressionRatio(b *testing.B) {
	patterns := []struct {
		name string
		gen  func() []uint32
	}{
		{"sequential", func() []uint32 { return genSequential(blockSize) }},
		{"small_values", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = uint32(i % 16)
			}
			return v
		}},
		{"medium_values", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = uint32(i * 1000)
			}
			return v
		}},
		{"large_values", func() []uint32 {
			v := make([]uint32, blockSize)
			for i := range v {
				v[i] = 0x10000000 + uint32(i)
			}
			return v
		}},
		{"monotonic_delta", func() []uint32 { return genMonotonic(blockSize) }},
		{"mixed_delta", func() []uint32 { return genMixed(blockSize) }},
		{"small_exceptions", genDataWithSmallExceptions},
		{"large_exceptions", genDataWithLargeExceptions},
	}
	for _, p := range patterns {
		b.Run(p.name, func(b *testing.B) {
			values := p.gen()
			packed, _ := PackUint32(0, values, nil, nil)
			rawSize := len(values) * 4
			ratio := float64(len(packed)) / float64(rawSize) * 100
			b.ReportMetric(ratio, "ratio%")
			b.ReportMetric(float64(len(packed)), "bytes")
		})
	}

	b.Run("monotonic_delta_compressed", func(b *testing.B) {
		values := genMonotonic(blockSize)
		clone := slices.Clone(values)
		packed, _ := PackUint32(Delta, clone, nil, nil)
		rawSize := len(values) * 4
		ratio := float64(len(packed)) / float64(rawSize) * 100
		b.ReportMetric(ratio, "ratio%")
		b.ReportMetric(float64(len(packed)), "bytes")
	})

	b.Run("mixed_delta_compressed", func(b *testing.B) {
		values := genMixed(blockSize)
		clone := slices.Clone(values)
		packed, _ := PackUint32(Delta, clone, nil, nil)
		rawSize := len(values) * 4
		ratio := float64(len(packed)) / float64(rawSize) * 100
		b.ReportMetric(ratio, "ratio%")
		b.ReportMetric(float64(len(packed)), "bytes")
	})
}

func BenchmarkGetUint32WithExceptions(b *testing.B) {
	values := genDataWithSmallExceptions()
	packed, _ := PackUint32(0, values, nil, nil)

	scratch := make([]uint32, ScratchLen)
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		GetUint32(i%blockSize, packed, scratch)
	}
}

func BenchmarkGetUint32Delta(b *testing.B) {
	values := genMonotonic(blockSize)
	clone := slices.Clone(values)
	packed, _ := PackUint32(Delta, clone, nil, nil)

	scratch := make([]uint32, ScratchLen)
	b.ReportAllocs()

	for i := 0; b.Loop(); i++ {
		GetUint32(i%blockSize, packed, scratch)
	}
}

func BenchmarkCollectAndWriteExceptions(b *testing.B) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i)
	}
	for i := range 48 {
		values[(i*11)%blockSize] = 0x200000 + uint32(i)
	}

	const bitWidth = 12
	const excCount = 48
	var excIdx [16]byte
	var highBits [blockSize]uint32

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		collectAndWriteExceptions(values, bitWidth, excIdx[:], excCount, highBits[:])
	}
}

func BenchmarkPackUint64(b *testing.B) {
	cases := []struct {
		name   string
		values []uint64
	}{
		{"fit32", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(i * 7)
			}
			return v
		}()},
		{"two_block", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0x1_0000_0000 + uint64(i*1000)
			}
			return v
		}()},
		{"for64", func() []uint64 {
			v := make([]uint64, blockSize)
			base := uint64(1_700_000_000_000)
			for i := range v {
				v[i] = base + uint64(i*1000)
			}
			return v
		}()},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			dst := make([]byte, 0, MaxBlockLength64(0))
			scratch := make([]uint32, ScratchLen64)
			input := slices.Clone(tc.values)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 8))
			b.ResetTimer()
			for b.Loop() {
				copy(input, tc.values)
				dst, _ = PackUint64(0, input, dst[:0], scratch)
			}
		})
	}
}

func BenchmarkUnpackUint64(b *testing.B) {
	cases := []struct {
		name   string
		values []uint64
	}{
		{"fit32", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(i * 7)
			}
			return v
		}()},
		{"two_block", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0x1_0000_0000 + uint64(i*1000)
			}
			return v
		}()},
		{"for64", func() []uint64 {
			v := make([]uint64, blockSize)
			base := uint64(1_700_000_000_000)
			for i := range v {
				v[i] = base + uint64(i*1000)
			}
			return v
		}()},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			packed, _ := PackUint64(0, slices.Clone(tc.values), nil, nil)
			dst := make([]uint64, blockSize)
			scratch := make([]uint32, ScratchLen64)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 8))
			b.ResetTimer()
			for b.Loop() {
				UnpackUint64(packed, dst, scratch)
			}
		})
	}
}
