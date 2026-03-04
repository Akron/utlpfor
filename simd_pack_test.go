//go:build goexperiment.simd && amd64

package utlpfor

import (
	"fmt"
	"math/rand"
	"testing"

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
