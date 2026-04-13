package utlpfor

import (
	"math/bits"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func FuzzPackUnpackUint32RoundTrip(f *testing.F) {
	f.Add(encodeValuesSeed([]uint32{0, 1, 2, 3}))
	f.Add(encodeValuesSeed([]uint32{0xFFFFFFFF}))
	f.Add(encodeValuesSeed(make([]uint32, 128)))
	f.Add(encodeValuesSeed([]uint32{0, 255, 256, 65535, 65536, 0xFFFFFF, 0x1000000, 0xFFFFFFFF}))

	f.Fuzz(func(t *testing.T, data []byte) {
		values := decodeValuesSeed(data)
		if len(values) == 0 || len(values) > blockSize {
			return
		}

		original := slices.Clone(values)
		packed, err := PackUint32(0, nil, nil, values)
		require.NoError(t, err)

		unpacked, consumed, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
		require.NoError(t, err)
		require.Equal(t, len(packed), consumed)
		require.Equal(t, original, unpacked)
	})
}

func FuzzPackDeltaUint32RoundTrip(f *testing.F) {
	f.Add(encodeValuesSeed([]uint32{10, 20, 30, 40, 50}))
	f.Add(encodeValuesSeed([]uint32{100, 50, 200, 10}))
	f.Add(encodeValuesSeed([]uint32{0, 0, 0, 0}))

	f.Fuzz(func(t *testing.T, data []byte) {
		values := decodeValuesSeed(data)
		if len(values) == 0 || len(values) > blockSize {
			return
		}

		original := slices.Clone(values)
		packed, err := PackUint32(Delta, nil, nil, values)
		require.NoError(t, err)

		unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
		require.NoError(t, err)
		require.Equal(t, original, unpacked)
	})
}

func FuzzGetUint32MatchesUnpack(f *testing.F) {
	f.Add(encodeValuesSeed([]uint32{1, 2, 3, 4, 5}))

	f.Fuzz(func(t *testing.T, data []byte) {
		values := decodeValuesSeed(data)
		if len(values) == 0 || len(values) > blockSize {
			return
		}

		packed, err := PackUint32(0, nil, nil, values)
		require.NoError(t, err)

		unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
		require.NoError(t, err)

		for pos := range unpacked {
			got, err := GetUint32(pos, packed)
			require.NoError(t, err)
			require.Equal(t, unpacked[pos], got, "pos=%d", pos)
		}
	})
}

func FuzzBlockLengthNeverPanics(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0x00, 0x00, 0x00, 0x00})
	f.Add([]byte{0xFF, 0xFF, 0xFF, 0xFF})

	f.Fuzz(func(t *testing.T, data []byte) {
		_, _ = BlockLength(data)
	})
}

func FuzzCorruptDeltaOverflow(f *testing.F) {
	f.Add(encodeValuesSeed([]uint32{0xFFFFFFFF, 0xFFFFFFFF, 1}))

	f.Fuzz(func(t *testing.T, data []byte) {
		values := decodeValuesSeed(data)
		if len(values) < 2 || len(values) > blockSize {
			return
		}

		packed, err := PackUint32(Delta, nil, nil, values)
		if err != nil {
			return
		}

		_, _, err = UnpackUint32(nil, make([]uint32, blockSize), packed)
		assert.NoError(t, err)
	})
}

func FuzzDeltaWithExceptions(f *testing.F) {
	f.Add(encodeValuesSeed([]uint32{100, 200, 0xFFFFFF, 400, 500}))
	f.Add(encodeValuesSeed([]uint32{0, 0, 0x10000, 0, 0, 0, 0, 0x1000000}))

	f.Fuzz(func(t *testing.T, data []byte) {
		values := decodeValuesSeed(data)
		if len(values) == 0 || len(values) > blockSize {
			return
		}

		original := slices.Clone(values)
		packed, err := PackUint32(Delta, nil, nil, values)
		if err != nil {
			return
		}

		unpacked, _, err := UnpackUint32(nil, make([]uint32, blockSize), packed)
		require.NoError(t, err)
		require.Equal(t, original, unpacked,
			"delta+exceptions round-trip must preserve values")
	})
}

func FuzzCompressionRatio(f *testing.F) {
	f.Add(encodeValuesSeed(make([]uint32, 128)))
	f.Add(encodeValuesSeed([]uint32{1, 2, 3, 4, 5, 6, 7, 8}))

	f.Fuzz(func(t *testing.T, data []byte) {
		values := decodeValuesSeed(data)
		if len(values) < 16 {
			return
		}

		maxBW := 0
		for _, v := range values {
			bw := bits.Len32(v)
			if bw > maxBW {
				maxBW = bw
			}
		}

		packed, err := PackUint32(0, nil, nil, values)
		if err != nil {
			return
		}

		theoreticalPacked := headerBytes + utlPayloadBytesLUT[maxBW]
		rawSize := len(values) * 4
		if maxBW < 32 && theoreticalPacked < rawSize {
			assert.LessOrEqual(t, len(packed), rawSize,
				"blocks with sub-32-bit values should compress (bw=%d, count=%d)", maxBW, len(values))
		}
	})
}
