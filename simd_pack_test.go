//go:build goexperiment.simd && amd64

package utlpfor

import (
	"fmt"
	"math/rand"
	"simd/archsimd"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPackUTL_SIMDMatchesScalar_AllBitWidths(t *testing.T) {
	for bw := 1; bw <= 32; bw++ {
		t.Run(fmt.Sprintf("bw%d", bw), func(t *testing.T) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range values {
				values[i] = uint32(i*7+3) & mask
			}

			payloadLen := utlPayloadBytesLUT[bw]
			scalarOut := make([]byte, payloadLen)
			packLanesUTLScalar(scalarOut, values, bw)

			if simdLevel >= simdLevelSSE2 {
				simdOut := make([]byte, payloadLen)
				packLanesUTLSSE2(simdOut, values, bw)
				assert.Equal(t, scalarOut, simdOut, "SSE2 pack mismatch at bw=%d", bw)
			}

			if simdLevel >= simdLevelAVX2 {
				simdOut := make([]byte, payloadLen)
				packLanesUTLAVX2(simdOut, values, bw)
				assert.Equal(t, scalarOut, simdOut, "AVX2 pack mismatch at bw=%d", bw)
			}

			if simdLevel >= simdLevelAVX512 {
				simdOut := make([]byte, payloadLen)
				packLanesUTLAVX512(simdOut, values, bw)
				assert.Equal(t, scalarOut, simdOut, "AVX-512 pack mismatch at bw=%d", bw)
			}
		})
	}
}

func TestUnpackUTL_SIMDMatchesScalar_AllBitWidths(t *testing.T) {
	for bw := 1; bw <= 32; bw++ {
		t.Run(fmt.Sprintf("bw%d", bw), func(t *testing.T) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
			for i := range values {
				values[i] = uint32(i*7+3) & mask
			}

			payloadLen := utlPayloadBytesLUT[bw]
			payload := make([]byte, payloadLen)
			packLanesUTLScalar(payload, values, bw)

			scalarDst := make([]uint32, blockSize)
			unpackLanesUTLScalar(scalarDst, payload, blockSize, bw)

			if simdLevel >= simdLevelSSE2 {
				simdDst := make([]uint32, blockSize)
				unpackLanesUTLSSE2(simdDst, payload, blockSize, bw)
				assert.Equal(t, scalarDst, simdDst, "SSE2 unpack mismatch at bw=%d", bw)
			}

			if simdLevel >= simdLevelAVX2 {
				simdDst := make([]uint32, blockSize)
				unpackLanesUTLAVX2(simdDst, payload, blockSize, bw)
				assert.Equal(t, scalarDst, simdDst, "AVX2 unpack mismatch at bw=%d", bw)
			}

			if simdLevel >= simdLevelAVX512 {
				simdDst := make([]uint32, blockSize)
				unpackLanesUTLAVX512(simdDst, payload, blockSize, bw)
				assert.Equal(t, scalarDst, simdDst, "AVX-512 unpack mismatch at bw=%d", bw)
			}
		})
	}
}

func TestPackUnpack_SIMDScalarDifferential_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	for seed := int64(0); seed < 1000; seed++ {
		rng.Seed(seed)
		count := rng.Intn(blockSize) + 1
		values := make([]uint32, count)
		for i := range values {
			values[i] = rng.Uint32()
		}

		packed, err := PackUint32(0, nil, values)
		require.NoError(t, err, "seed=%d", seed)

		unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
		require.NoError(t, err, "seed=%d", seed)
		assert.Equal(t, values, unpacked, "seed=%d", seed)
	}
}

func TestGetUint32_SIMDMatchesScalar(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i*13 + 7)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	for pos := range blockSize {
		got, err := GetUint32(pos, packed)
		require.NoError(t, err, "pos=%d", pos)
		assert.Equal(t, values[pos], got, "pos=%d", pos)
	}
}

// --- Kernel-only micro-benchmarks for SIMD investigation ---

func BenchmarkKernelUnpackScalar(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytesLUT[bw]
			payload := make([]byte, payloadLen)
			packLanesUTLScalar(payload, values, bw)
			dst := make([]uint32, blockSize)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				unpackLanesUTLScalar(dst, payload, blockSize, bw)
			}
		})
	}
}

func BenchmarkKernelUnpackSSE2(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytesLUT[bw]
			payload := make([]byte, payloadLen)
			packLanesUTLScalar(payload, values, bw)
			dst := make([]uint32, blockSize)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				unpackLanesUTLSSE2(dst, payload, blockSize, bw)
			}
		})
	}
}

func BenchmarkKernelUnpackAVX2(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytesLUT[bw]
			payload := make([]byte, payloadLen)
			packLanesUTLScalar(payload, values, bw)
			dst := make([]uint32, blockSize)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				unpackLanesUTLAVX2(dst, payload, blockSize, bw)
			}
		})
	}
}

func BenchmarkKernelUnpackAVX2_VZ(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytesLUT[bw]
			payload := make([]byte, payloadLen)
			packLanesUTLScalar(payload, values, bw)
			dst := make([]uint32, blockSize)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				unpackLanesUTLAVX2(dst, payload, blockSize, bw)
				archsimd.ClearAVXUpperBits()
			}
		})
	}
}

func BenchmarkKernelPackAVX2_VZ(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytesLUT[bw]
			dst := make([]byte, payloadLen)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				packLanesUTLAVX2(dst, values, bw)
				archsimd.ClearAVXUpperBits()
			}
		})
	}
}

func BenchmarkKernelPackScalar(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytesLUT[bw]
			dst := make([]byte, payloadLen)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				packLanesUTLScalar(dst, values, bw)
			}
		})
	}
}

func BenchmarkKernelPackSSE2(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytesLUT[bw]
			dst := make([]byte, payloadLen)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				packLanesUTLSSE2(dst, values, bw)
			}
		})
	}
}

func BenchmarkKernelPackAVX2(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytesLUT[bw]
			dst := make([]byte, payloadLen)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				packLanesUTLAVX2(dst, values, bw)
			}
		})
	}
}

func BenchmarkRawLoadStoreSSE2(b *testing.B) {
	src := make([]uint32, blockSize)
	dst := make([]uint32, blockSize)
	for i := range src {
		src[i] = uint32(i)
	}
	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < blockSize; j += 4 {
			v := archsimd.LoadUint32x4Slice(src[j : j+4])
			v.StoreSlice(dst[j : j+4])
		}
	}
}

func BenchmarkRawLoadStoreAVX2(b *testing.B) {
	src := make([]uint32, blockSize)
	dst := make([]uint32, blockSize)
	for i := range src {
		src[i] = uint32(i)
	}
	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < blockSize; j += 8 {
			v := archsimd.LoadUint32x8Slice(src[j : j+8])
			v.StoreSlice(dst[j : j+8])
		}
	}
}

func BenchmarkRawLoadStoreAVX2_Ptr(b *testing.B) {
	src := make([]uint32, blockSize)
	dst := make([]uint32, blockSize)
	for i := range src {
		src[i] = uint32(i)
	}
	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < blockSize; j += 8 {
			v := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&src[j])))
			v.Store((*[8]uint32)(unsafe.Pointer(&dst[j])))
		}
	}
}

func BenchmarkRawShiftAndSSE2(b *testing.B) {
	src := make([]uint32, blockSize)
	dst := make([]uint32, blockSize)
	mask := archsimd.BroadcastUint32x4(0xFF)
	for i := range src {
		src[i] = uint32(i * 7)
	}
	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < blockSize; j += 4 {
			v := archsimd.LoadUint32x4Slice(src[j : j+4])
			r := v.ShiftAllRight(3).And(mask)
			r.StoreSlice(dst[j : j+4])
		}
	}
}

func BenchmarkRawShiftAndAVX2(b *testing.B) {
	src := make([]uint32, blockSize)
	dst := make([]uint32, blockSize)
	mask := archsimd.BroadcastUint32x8(0xFF)
	for i := range src {
		src[i] = uint32(i * 7)
	}
	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < blockSize; j += 8 {
			v := archsimd.LoadUint32x8Slice(src[j : j+8])
			r := v.ShiftAllRight(3).And(mask)
			r.StoreSlice(dst[j : j+8])
		}
	}
}

func BenchmarkRawShiftAndAVX2_Ptr(b *testing.B) {
	src := make([]uint32, blockSize)
	dst := make([]uint32, blockSize)
	mask := archsimd.BroadcastUint32x8(0xFF)
	for i := range src {
		src[i] = uint32(i * 7)
	}
	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < blockSize; j += 8 {
			v := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&src[j])))
			r := v.ShiftAllRight(3).And(mask)
			r.Store((*[8]uint32)(unsafe.Pointer(&dst[j])))
		}
	}
}

func unpackAVX2_constBW8(dst []uint32, payload []byte) {
	const bw = 8
	mask := archsimd.BroadcastUint32x8(0xFF)
	bitOffset := 0
	for v := 0; v < utlValuesPerLane; v++ {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		lo := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&payload[base])))
		hi := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&payload[base+32])))
		rLo := lo.ShiftAllRight(shift).And(mask)
		rHi := hi.ShiftAllRight(shift).And(mask)

		if int(shift)+bw > 32 {
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			nextLo := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&payload[nextBase])))
			nextHi := archsimd.LoadUint32x8((*[8]uint32)(unsafe.Pointer(&payload[nextBase+32])))
			leftShift := uint64(32) - shift
			rLo = rLo.Or(nextLo.ShiftAllLeft(leftShift).And(mask))
			rHi = rHi.Or(nextHi.ShiftAllLeft(leftShift).And(mask))
		}

		outBase := v * utlLaneCount
		rLo.StoreSlice(dst[outBase : outBase+8])
		rHi.StoreSlice(dst[outBase+8 : outBase+16])
		bitOffset += bw
	}
}

func BenchmarkKernelUnpackAVX2_ConstBW8(b *testing.B) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i*7) & 0xFF
	}
	payloadLen := utlPayloadBytesLUT[8]
	payload := make([]byte, payloadLen)
	packLanesUTLScalar(payload, values, 8)
	dst := make([]uint32, blockSize)

	b.ReportAllocs()
	b.SetBytes(int64(blockSize * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		unpackAVX2_constBW8(dst, payload)
	}
}

func TestPackUnpack_SIMDEdgeBitWidths(t *testing.T) {
	for _, bw := range []int{0, 1, 31, 32} {
		t.Run(fmt.Sprintf("bw%d", bw), func(t *testing.T) {
			values := make([]uint32, blockSize)
			if bw > 0 {
				mask := uint32((1 << bw) - 1)
				if bw == 32 {
					mask = 0xFFFFFFFF
				}
				for i := range values {
					values[i] = uint32(i) & mask
				}
			}

			packed, err := PackUint32(0, nil, values)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
			require.NoError(t, err)
			assert.Equal(t, values, unpacked)
		})
	}
}
