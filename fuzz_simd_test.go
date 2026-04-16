//go:build goexperiment.simd && amd64

package utlpfor

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func FuzzSIMDScalarConsistency(f *testing.F) {
	f.Add(encodeValuesSeed([]uint32{1, 2, 3, 4}))
	f.Add(encodeValuesSeed([]uint32{0, 0xFF, 0xFFFF, 0xFFFFFFFF}))
	f.Add(encodeValuesSeed(make([]uint32, 128)))

	f.Fuzz(func(t *testing.T, data []byte) {
		values := decodeValuesSeed(data)
		if len(values) == 0 || len(values) > blockSize {
			return
		}

		scratch := make([]uint32, blockSize)
		scalarValues := make([]uint32, len(values))
		copy(scalarValues, values)
		simdValues := make([]uint32, len(values))
		copy(simdValues, values)

		scalarPacked, err := packUint32Scalar(0, nil, scratch, scalarValues)
		if err != nil {
			return
		}
		simdPacked, err := PackUint32(0, simdValues, nil, nil)
		if err != nil {
			return
		}
		require.Equal(t, scalarPacked, simdPacked,
			"SIMD and scalar must produce identical bytes")
	})
}
