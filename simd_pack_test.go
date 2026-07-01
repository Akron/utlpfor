//go:build goexperiment.simd && amd64

package utlpfor

import (
	"fmt"
	"math/rand"
	"simd/archsimd"
	"slices"
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

			payloadLen := utlPayloadBytes(bw)
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

			payloadLen := utlPayloadBytes(bw)
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
	for seed := range int64(1000) {
		rng.Seed(seed)
		count := rng.Intn(blockSize) + 1
		values := make([]uint32, count)
		for i := range values {
			values[i] = rng.Uint32()
		}
		original := slices.Clone(values)

		packed, err := PackUint32(0, values, nil, nil)
		require.NoError(t, err, "seed=%d", seed)

		unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
		require.NoError(t, err, "seed=%d", seed)
		assert.Equal(t, original, unpacked, "seed=%d", seed)
	}
}

func TestGetUint32_SIMDMatchesScalar(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i*13 + 7)
	}
	packed, err := PackUint32(0, values, nil, nil)
	require.NoError(t, err)

	scratch := make([]uint32, ScratchLen)
	for pos := range blockSize {
		got, err := GetUint32(pos, packed, scratch)
		require.NoError(t, err, "pos=%d", pos)
		assert.Equal(t, values[pos], got, "pos=%d", pos)
	}
}

func BenchmarkKernelUnpackScalar(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytes(bw)
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
			payloadLen := utlPayloadBytes(bw)
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
			payloadLen := utlPayloadBytes(bw)
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
			payloadLen := utlPayloadBytes(bw)
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
			payloadLen := utlPayloadBytes(bw)
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
			payloadLen := utlPayloadBytes(bw)
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
			payloadLen := utlPayloadBytes(bw)
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
			payloadLen := utlPayloadBytes(bw)
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
			v := archsimd.LoadUint32x4(src[j : j+4])
			v.Store(dst[j : j+4])
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
			v := archsimd.LoadUint32x8(src[j : j+8])
			v.Store(dst[j : j+8])
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
			v := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&src[j])))
			v.StoreArray((*[8]uint32)(unsafe.Pointer(&dst[j])))
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
			v := archsimd.LoadUint32x4(src[j : j+4])
			r := v.ShiftAllRight(3).And(mask)
			r.Store(dst[j : j+4])
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
			v := archsimd.LoadUint32x8(src[j : j+8])
			r := v.ShiftAllRight(3).And(mask)
			r.Store(dst[j : j+8])
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
			v := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&src[j])))
			r := v.ShiftAllRight(3).And(mask)
			r.StoreArray((*[8]uint32)(unsafe.Pointer(&dst[j])))
		}
	}
}

func unpackAVX2_constBW8(dst []uint32, payload []byte) {
	const bw = 8
	mask := archsimd.BroadcastUint32x8(0xFF)
	bitOffset := 0
	for v := range utlValuesPerLane {
		wordIdx := bitOffset / 32
		shift := uint64(bitOffset % 32)
		base := wordIdx * utlSuperWordBytes

		lo := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&payload[base])))
		hi := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&payload[base+32])))
		rLo := lo.ShiftAllRight(shift).And(mask)
		rHi := hi.ShiftAllRight(shift).And(mask)

		if int(shift)+bw > 32 {
			nextBase := (wordIdx + 1) * utlSuperWordBytes
			nextLo := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&payload[nextBase])))
			nextHi := archsimd.LoadUint32x8Array((*[8]uint32)(unsafe.Pointer(&payload[nextBase+32])))
			leftShift := uint64(32) - shift
			rLo = rLo.Or(nextLo.ShiftAllLeft(leftShift).And(mask))
			rHi = rHi.Or(nextHi.ShiftAllLeft(leftShift).And(mask))
		}

		outBase := v * utlLaneCount
		rLo.Store(dst[outBase : outBase+8])
		rHi.Store(dst[outBase+8 : outBase+16])
		bitOffset += bw
	}
}

func BenchmarkKernelUnpackAVX2_ConstBW8(b *testing.B) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i*7) & 0xFF
	}
	payloadLen := utlPayloadBytes(8)
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

func BenchmarkKernelUnpackAVX2_Direct(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytes(bw)
			payload := make([]byte, payloadLen)
			packLanesUTLScalar(payload, values, bw)
			dst := make([]uint32, blockSize)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			switch bw {
			case 8:
				for i := 0; i < b.N; i++ {
					unpackAVX2BW8(&dst[0], &payload[0])
				}
			case 16:
				for i := 0; i < b.N; i++ {
					unpackAVX2BW16(&dst[0], &payload[0])
				}
			}
		})
	}
}

func BenchmarkKernelUnpackAVX512(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytes(bw)
			payload := make([]byte, payloadLen)
			packLanesUTLScalar(payload, values, bw)
			dst := make([]uint32, blockSize)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				unpackLanesUTLAVX512(dst, payload, blockSize, bw)
			}
		})
	}
}

func BenchmarkKernelPackAVX512(b *testing.B) {
	for _, bw := range []int{8, 16} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			for i := range values {
				values[i] = uint32(i*7) & mask
			}
			payloadLen := utlPayloadBytes(bw)
			dst := make([]byte, payloadLen)

			b.ReportAllocs()
			b.SetBytes(int64(blockSize * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				packLanesUTLAVX512(dst, values, bw)
			}
		})
	}
}

// simdPackUnpackUint64 packs with SIMD and unpacks with SIMD, comparing
// against a scalar roundtrip. It tests each available SIMD level directly.
func simdPackUnpackUint64(t *testing.T, flag Flag, values []uint64, label string) {
	t.Helper()

	scratchScalar := make([]uint32, ScratchLen64)
	scalarPacked, err := packUint64Scalar(flag, slices.Clone(values), nil, scratchScalar)
	require.NoError(t, err, "%s: scalar pack", label)

	scalarDst := make([]uint64, blockSize)
	scalarUnpacked, scalarConsumed, err := unpackUint64Scalar(scalarDst, scratchScalar, scalarPacked)
	require.NoError(t, err, "%s: scalar unpack", label)
	require.Equal(t, values, scalarUnpacked, "%s: scalar roundtrip", label)

	if simdLevel >= simdLevelSSE2 {
		scratchSSE := make([]uint32, ScratchLen64)
		ssePacked, err := packUint64SSE2(flag, slices.Clone(values), nil, scratchSSE)
		require.NoError(t, err, "%s: SSE2 pack", label)
		assert.Equal(t, scalarPacked, ssePacked, "%s: SSE2 packed output differs from scalar", label)

		sseDst := make([]uint64, blockSize)
		sseUnpacked, sseConsumed, err := unpackUint64SSE2(sseDst, scratchSSE, ssePacked)
		require.NoError(t, err, "%s: SSE2 unpack", label)
		assert.Equal(t, scalarConsumed, sseConsumed, "%s: SSE2 consumed differs", label)
		assert.Equal(t, values, sseUnpacked, "%s: SSE2 roundtrip", label)
	}

	if simdLevel >= simdLevelAVX2 {
		scratchAVX := make([]uint32, ScratchLen64)
		avxPacked, err := packUint64AVX2(flag, slices.Clone(values), nil, scratchAVX)
		require.NoError(t, err, "%s: AVX2 pack", label)
		assert.Equal(t, scalarPacked, avxPacked, "%s: AVX2 packed output differs from scalar", label)

		avxDst := make([]uint64, blockSize)
		avxUnpacked, avxConsumed, err := unpackUint64AVX2(avxDst, scratchAVX, avxPacked)
		require.NoError(t, err, "%s: AVX2 unpack", label)
		assert.Equal(t, scalarConsumed, avxConsumed, "%s: AVX2 consumed differs", label)
		assert.Equal(t, values, avxUnpacked, "%s: AVX2 roundtrip", label)
	}

	if simdLevel >= simdLevelAVX512 {
		scratchAVX5 := make([]uint32, ScratchLen64)
		avx5Packed, err := packUint64AVX512(flag, slices.Clone(values), nil, scratchAVX5)
		require.NoError(t, err, "%s: AVX512 pack", label)

		avx5Dst := make([]uint64, blockSize)
		avx5Unpacked, avx5Consumed, err := unpackUint64AVX512(avx5Dst, scratchAVX5, avx5Packed)
		require.NoError(t, err, "%s: AVX512 unpack", label)
		assert.Equal(t, scalarConsumed, avx5Consumed, "%s: AVX512 consumed differs", label)
		assert.Equal(t, values, avx5Unpacked, "%s: AVX512 roundtrip", label)
	}
}

func TestPackUnpackUint64_SIMDMatchesScalar_AllZero(t *testing.T) {
	values := make([]uint64, blockSize)
	simdPackUnpackUint64(t, 0, values, "all-zero")
}

func TestPackUnpackUint64_SIMDMatchesScalar_AllFitIn32(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i*13 + 7)
	}
	simdPackUnpackUint64(t, 0, values, "fit-in-32")
}

func TestPackUnpackUint64_SIMDMatchesScalar_TwoBlock(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i)*1000 + (1 << 40)
	}
	simdPackUnpackUint64(t, 0, values, "two-block")
}

func TestPackUnpackUint64_SIMDMatchesScalar_FOR64(t *testing.T) {
	base := uint64(1) << 50
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = base + uint64(i*3)
	}
	simdPackUnpackUint64(t, 0, values, "for64")
}

func TestPackUnpackUint64_SIMDMatchesScalar_Partial(t *testing.T) {
	for _, count := range []int{1, 7, 63, 100} {
		t.Run(fmt.Sprintf("count%d", count), func(t *testing.T) {
			values := make([]uint64, count)
			for i := range values {
				values[i] = uint64(i)*12345 + (1 << 33)
			}
			simdPackUnpackUint64(t, 0, values, fmt.Sprintf("partial-%d", count))
		})
	}
}

func TestPackUnpackUint64_SIMDMatchesScalar_WithDelta(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(1000000 + i*100)
	}
	simdPackUnpackUint64(t, Delta, values, "delta")
}

func TestPackUnpackUint64_SIMDMatchesScalar_WithNoPatch(t *testing.T) {
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = uint64(i & 0xFF)
	}
	simdPackUnpackUint64(t, NoPatch, values, "no-patch")
}

func TestPackUnpackUint64_SIMDMatchesScalar_WithNoFOR(t *testing.T) {
	base := uint64(1) << 50
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = base + uint64(i*3)
	}
	simdPackUnpackUint64(t, NoFOR, values, "no-for")
}

func TestPackUnpackUint64_SIMDMatchesScalar_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for seed := range int64(200) {
		rng.Seed(seed)
		count := rng.Intn(blockSize) + 1
		values := make([]uint64, count)
		for i := range values {
			values[i] = uint64(rng.Int63())
		}
		original := slices.Clone(values)
		simdPackUnpackUint64(t, 0, original, fmt.Sprintf("random-seed%d", seed))
	}
}

func TestPackUnpackUint64_SIMDMatchesScalar_FOR64_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for trial := range 50 {
		base := uint64(rng.Int63n(1<<62)) + (1 << 32)
		count := rng.Intn(blockSize) + 1
		values := make([]uint64, count)
		for i := range values {
			values[i] = base + uint64(rng.Intn(1<<20))
		}
		original := slices.Clone(values)
		simdPackUnpackUint64(t, 0, original, fmt.Sprintf("for64-random-%d", trial))
	}
}

func TestPackUnpackUint64_SIMDMatchesScalar_CrossUnpack(t *testing.T) {
	base := uint64(1) << 50
	values := make([]uint64, blockSize)
	for i := range values {
		values[i] = base + uint64(i*3)
	}

	scratchScalar := make([]uint32, ScratchLen64)
	scalarPacked, err := packUint64Scalar(0, slices.Clone(values), nil, scratchScalar)
	require.NoError(t, err)

	if simdLevel >= simdLevelSSE2 {
		sseDst := make([]uint64, blockSize)
		sseScratch := make([]uint32, ScratchLen64)
		sseUnpacked, _, err := unpackUint64SSE2(sseDst, sseScratch, scalarPacked)
		require.NoError(t, err, "SSE2 unpack of scalar-packed")
		assert.Equal(t, values, sseUnpacked, "SSE2 cross-unpack")
	}

	if simdLevel >= simdLevelSSE2 {
		sseScratch := make([]uint32, ScratchLen64)
		ssePacked, err := packUint64SSE2(0, slices.Clone(values), nil, sseScratch)
		require.NoError(t, err, "SSE2 pack")

		scalarDst := make([]uint64, blockSize)
		scalarUnpacked, _, err := unpackUint64Scalar(scalarDst, scratchScalar, ssePacked)
		require.NoError(t, err, "scalar unpack of SSE2-packed")
		assert.Equal(t, values, scalarUnpacked, "scalar cross-unpack of SSE2")
	}
}

// TestAnalyzeUint64_SIMDMatchesScalar verifies that the fused analyze
// functions (analyzeUint64SSE2, analyzeUint64AVX2, analyzeUint64AVX512)
// produce the same min/max/acc as the scalar analyzeUint64.
func TestAnalyzeUint64_SIMDMatchesScalar(t *testing.T) {
	patterns := []struct {
		name string
		gen  func() []uint64
	}{
		{"sequential", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(i * 7)
			}
			return v
		}},
		{"for64_timestamps", func() []uint64 {
			v := make([]uint64, blockSize)
			base := uint64(1_700_000_000_000)
			for i := range v {
				v[i] = base + uint64(i*1000)
			}
			return v
		}},
		{"boundary_crossing", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0xFFFFFFC0 + uint64(i)
			}
			return v
		}},
		{"above_32bit", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0x1_0000_0000 + uint64(i*1000)
			}
			return v
		}},
		{"max_uint64", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = ^uint64(0) - uint64(i)
			}
			return v
		}},
		{"single_value", func() []uint64 {
			return []uint64{42}
		}},
		{"two_values", func() []uint64 {
			return []uint64{100, 200}
		}},
		{"three_values", func() []uint64 {
			return []uint64{0x1_0000_0000, 0x2_0000_0000, 0x3_0000_0000}
		}},
		{"random", func() []uint64 {
			rng := rand.New(rand.NewSource(123))
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(rng.Int63())
			}
			return v
		}},
	}

	for _, p := range patterns {
		t.Run(p.name, func(t *testing.T) {
			values := p.gen()
			scalarMin, scalarMax, scalarAcc := analyzeUint64(values)

			if simdLevel >= simdLevelSSE2 {
				sseMin, sseMax, sseAcc := analyzeUint64SSE2(values)
				assert.Equal(t, scalarMin, sseMin, "SSE2 min mismatch")
				assert.Equal(t, scalarMax, sseMax, "SSE2 max mismatch")
				assert.Equal(t, scalarAcc, sseAcc, "SSE2 acc mismatch")
			}

			if simdLevel >= simdLevelAVX2 {
				avxMin, avxMax, avxAcc := analyzeUint64AVX2(values)
				assert.Equal(t, scalarMin, avxMin, "AVX2 min mismatch")
				assert.Equal(t, scalarMax, avxMax, "AVX2 max mismatch")
				assert.Equal(t, scalarAcc, avxAcc, "AVX2 acc mismatch")
			}

			if simdLevel >= simdLevelAVX512 {
				a512Min, a512Max, a512Acc := analyzeUint64AVX512(values)
				assert.Equal(t, scalarMin, a512Min, "AVX512 min mismatch")
				assert.Equal(t, scalarMax, a512Max, "AVX512 max mismatch")
				assert.Equal(t, scalarAcc, a512Acc, "AVX512 acc mismatch")
			}
		})
	}
}

// TestSplitUint64Only_SIMDMatchesScalar verifies that splitUint64OnlyAVX2
// produces the same lower/upper halves as the scalar splitUint64Only.
func TestSplitUint64Only_SIMDMatchesScalar(t *testing.T) {
	patterns := []struct {
		name string
		gen  func() []uint64
	}{
		{"above_32bit", func() []uint64 {
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = 0x1_0000_0000 + uint64(i*1000)
			}
			return v
		}},
		{"random", func() []uint64 {
			rng := rand.New(rand.NewSource(456))
			v := make([]uint64, blockSize)
			for i := range v {
				v[i] = uint64(rng.Int63())
			}
			return v
		}},
		{"small_count", func() []uint64 {
			return []uint64{0x1_0000_0000, 0x2_0000_0000, 0x3_0000_0000}
		}},
	}

	for _, p := range patterns {
		t.Run(p.name, func(t *testing.T) {
			values := p.gen()
			n := len(values)

			sLower := make([]uint32, n)
			sUpper := make([]uint32, n)
			splitUint64Only(sLower, sUpper, values)

			if simdLevel >= simdLevelAVX2 {
				aLower := make([]uint32, n)
				aUpper := make([]uint32, n)
				splitUint64OnlyAVX2(aLower, aUpper, values)
				assert.Equal(t, sLower, aLower, "AVX2 lower mismatch")
				assert.Equal(t, sUpper, aUpper, "AVX2 upper mismatch")
			}
		})
	}
}

// TestNarrowToUint32 verifies that narrowToUint32 and its SIMD variants
// produce correct output matching scalar truncation.
func TestNarrowToUint32(t *testing.T) {
	for _, tc := range []struct {
		name string
		n    int
	}{
		{"full_block", blockSize},
		{"small", 3},
		{"one", 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := make([]uint64, tc.n)
			for i := range values {
				values[i] = uint64(i*13 + 7)
			}

			expected := make([]uint32, tc.n)
			narrowToUint32(expected, values, tc.n)
			for i := range tc.n {
				assert.Equal(t, uint32(values[i]), expected[i], "scalar pos %d", i)
			}

			if simdLevel >= simdLevelAVX2 {
				dst := make([]uint32, tc.n)
				narrowToUint32AVX2(dst, values, tc.n)
				assert.Equal(t, expected, dst, "AVX2 mismatch")
			}

			if simdLevel >= simdLevelAVX512 {
				dst := make([]uint32, tc.n)
				narrowToUint32AVX512(dst, values, tc.n)
				assert.Equal(t, expected, dst, "AVX512 mismatch")
			}
		})
	}
}

// BenchmarkAnalyzeUint64_FOR64 benchmarks the fused analysis function
// used by the FOR64 pack path at each SIMD level.
func BenchmarkAnalyzeUint64_FOR64(b *testing.B) {
	values := make([]uint64, blockSize)
	base := uint64(1_700_000_000_000)
	for i := range values {
		values[i] = base + uint64(i*1000)
	}

	b.Run("analyze_scalar", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			analyzeUint64(values)
		}
	})

	if simdLevel >= simdLevelSSE2 {
		b.Run("analyze_SSE2", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				analyzeUint64SSE2(values)
			}
		})
	}

	if simdLevel >= simdLevelAVX2 {
		b.Run("analyze_AVX2", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				analyzeUint64AVX2(values)
			}
		})
	}

	if simdLevel >= simdLevelAVX512 {
		b.Run("analyze_AVX512", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				analyzeUint64AVX512(values)
			}
		})
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

			packed, err := PackUint32(0, values, nil, nil)
			require.NoError(t, err)
			unpacked, _, err := UnpackUint32(packed, nil, make([]uint32, blockSize))
			require.NoError(t, err)
			assert.Equal(t, values, unpacked)
		})
	}
}
