package utlpfor

import (
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestErrOverflow_Type(t *testing.T) {
	var err error = &ErrOverflow{Position: 5}
	var overflow *ErrOverflow
	assert.True(t, errors.As(err, &overflow))
	assert.Equal(t, 5, overflow.Position)
}

func TestErrOverflow_Message(t *testing.T) {
	err := &ErrOverflow{Position: 42}
	assert.Contains(t, err.Error(), "42")
}

func TestErrOverflow_IsNotErrInvalidBuffer(t *testing.T) {
	var err error = &ErrOverflow{Position: 1}
	assert.False(t, errors.Is(err, ErrInvalidBuffer))
}

func TestErrOverflow_MessagePrefix(t *testing.T) {
	err := &ErrOverflow{Position: 0}
	assert.Contains(t, err.Error(), "UTLpfor:")
}

func TestErrOverflow_CorruptDeltaBlock(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	expected := append([]uint32(nil), values...)
	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestErrOverflow_DescendingDelta_NoOverflow(t *testing.T) {
	values := make([]uint32, blockSize)
	for i := range values {
		values[i] = uint32(10000 - i*50)
	}
	expected := append([]uint32(nil), values...)
	packed, err := PackUint32(Delta, nil, nil, values)
	require.NoError(t, err)
	unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
	require.NoError(t, err)
	assert.Equal(t, expected, unpacked)
}

func TestPackUnpackScalar64RoundTrip_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for seed := range int64(1000) {
		rng.Seed(seed)
		count := rng.Intn(blockSize) + 1
		values := make([]uint32, count)
		for i := range values {
			values[i] = rng.Uint32()
		}
		original := slices.Clone(values)

		packed, err := PackUint32(0, nil, nil, values)
		require.NoError(t, err, "seed=%d", seed)

		unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
		require.NoError(t, err, "seed=%d", seed)
		assert.Equal(t, original, unpacked, "seed=%d count=%d", seed, count)
	}
}

func TestPackScalar64_PartialBlock(t *testing.T) {
	for _, count := range []int{1, 2, 3, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128} {
		for _, bw := range []int{4, 8, 12, 16, 20, 24, 28, 32} {
			t.Run(fmt.Sprintf("count%d_bw%d", count, bw), func(t *testing.T) {
				values := make([]uint32, count)
				mask := uint32((1 << bw) - 1)
				if bw == 32 {
					mask = 0xFFFFFFFF
				}
				for i := range values {
					values[i] = uint32(i*13+5) & mask
				}
				original := slices.Clone(values)

				packed, err := PackUint32(0, nil, nil, values)
				require.NoError(t, err)

				unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
				require.NoError(t, err)
				assert.Equal(t, original, unpacked)
			})
		}
	}
}

func BenchmarkKernelPackScalar64(b *testing.B) {
	for _, bw := range []int{4, 8, 12, 16, 20, 24, 28, 32} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
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

func BenchmarkKernelUnpackScalar64(b *testing.B) {
	for _, bw := range []int{4, 8, 12, 16, 20, 24, 28, 32} {
		b.Run(fmt.Sprintf("bw%d", bw), func(b *testing.B) {
			values := make([]uint32, blockSize)
			mask := uint32((1 << bw) - 1)
			if bw == 32 {
				mask = 0xFFFFFFFF
			}
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
