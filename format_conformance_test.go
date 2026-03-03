package utlpfor

import (
	"bytes"
	"errors"
	"math/bits"
	"os"
	"sort"
	"testing"

	"github.com/kaitai-io/kaitai_struct_go_runtime/kaitai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// kaitaiBlock holds parsed fields from a UTL-PFOR block using the Kaitai runtime.
// Field names and extraction logic correspond to the utl_pfor.ksy definition.
type kaitaiBlock struct {
	raw           uint32
	count         int
	bitWidth      int
	intType       int
	hasDelta      bool
	hasZigZag     bool
	excCount      int
	hasExceptions bool
	payloadSize   int
	svbLength     uint16
	payload       []byte
	excIndex      []byte
	svbData       []byte
}

// parseKaitaiBlock reads a packed block using the Kaitai stream, mirroring
// the utl_pfor.ksy schema. Returns parsed fields for conformance checks.
func parseKaitaiBlock(t *testing.T, data []byte) kaitaiBlock {
	t.Helper()
	s := kaitai.NewStream(bytes.NewReader(data))

	raw, err := s.ReadU4le()
	require.NoError(t, err, "kaitai: read header u4le")

	b := kaitaiBlock{raw: raw}
	b.count = int(raw & 0xFF)
	b.bitWidth = int((raw >> 8) & 0x3F)
	b.intType = int((raw >> 14) & 0x03)
	b.hasDelta = (raw & 0x00400000) != 0
	b.hasZigZag = (raw & 0x00800000) != 0
	b.excCount = int((raw >> 24) & 0xFF)
	b.hasExceptions = b.excCount > 0

	if b.bitWidth == 0 {
		b.payloadSize = 0
	} else {
		b.payloadSize = ((b.bitWidth + 3) / 4) * 64
	}

	if b.hasExceptions {
		svbLen, err := s.ReadU2le()
		require.NoError(t, err, "kaitai: read svb_length u2le")
		b.svbLength = svbLen
	}

	b.payload, err = s.ReadBytes(b.payloadSize)
	require.NoError(t, err, "kaitai: read payload")

	if b.hasExceptions {
		if b.excCount > 16 {
			b.excIndex, err = s.ReadBytes(16)
		} else {
			b.excIndex, err = s.ReadBytes(b.excCount)
		}
		require.NoError(t, err, "kaitai: read exception index")

		b.svbData, err = s.ReadBytes(int(b.svbLength))
		require.NoError(t, err, "kaitai: read svb_data")
	}

	eof, err := s.EOF()
	require.NoError(t, err, "kaitai: check EOF")
	assert.True(t, eof, "kaitai: stream should be fully consumed")

	return b
}

func TestFormatConformance_HeaderLayout(t *testing.T) {
	tests := []struct {
		name     string
		count    int
		bitWidth int
		excCount int
		flags    uint32
	}{
		{"plain_128_bw8", 128, 8, 0, headerTypeUint32Flag},
		{"plain_50_bw4", 50, 4, 0, headerTypeUint32Flag},
		{"delta_128_bw12", 128, 12, 0, headerTypeUint32Flag | headerDeltaFlag},
		{"delta_zz_128", 128, 16, 0, headerTypeUint32Flag | headerDeltaFlag | headerZigZagFlag},
		{"exceptions_10", 128, 4, 10, headerTypeUint32Flag},
		{"exceptions_128", 128, 4, 128, headerTypeUint32Flag},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := encodeHeader(tt.count, tt.bitWidth, tt.excCount, tt.flags)
			buf := make([]byte, 4)
			bo.PutUint32(buf, h)

			// Byte 0: count (bits 0-7).
			gotCount := int(buf[0])
			assert.Equal(t, tt.count&0xFF, gotCount, "count mismatch")

			// Byte 1 bits 0-5: bitWidth (bits 8-13).
			gotBW := int(buf[1]) & 0x3F
			assert.Equal(t, tt.bitWidth, gotBW, "bit width mismatch")

			// Byte 1 bits 6-7: intType (bits 14-15).
			gotType := int(buf[1]>>6) & 0x03
			assert.Equal(t, IntTypeUint32, gotType, "int type mismatch")

			// Byte 2 bit 6: delta flag (bit 22).
			gotDelta := buf[2]&(1<<6) != 0
			wantDelta := tt.flags&headerDeltaFlag != 0
			assert.Equal(t, wantDelta, gotDelta, "delta flag mismatch")

			// Byte 2 bit 7: zigzag flag (bit 23).
			gotZZ := buf[2]&(1<<7) != 0
			wantZZ := tt.flags&headerZigZagFlag != 0
			assert.Equal(t, wantZZ, gotZZ, "zigzag flag mismatch")

			// Byte 3: excCount (bits 24-31).
			gotExcCount := int(buf[3])
			assert.Equal(t, tt.excCount, gotExcCount, "excCount mismatch")

			gotExc := gotExcCount > 0
			wantExc := tt.excCount > 0
			assert.Equal(t, wantExc, gotExc, "exception flag mismatch")
		})
	}
}

func TestFormatConformance_HeaderBitPositions(t *testing.T) {
	// Verify exact bit positions match the Kaitai .ksy definition.
	h := encodeHeader(0x80, 0x3F, 0xFF, headerTypeUint32Flag|headerDeltaFlag|headerZigZagFlag)

	assert.Equal(t, 0x80, int(h&0xFF), "count at bits 0-7")
	assert.Equal(t, 0x3F, int((h>>8)&0x3F), "bitWidth at bits 8-13")
	assert.Equal(t, IntTypeUint32, int((h>>14)&0x03), "intType at bits 14-15")
	assert.True(t, h&headerDeltaFlag != 0, "delta at bit 22")
	assert.True(t, h&headerZigZagFlag != 0, "zigzag at bit 23")
	assert.Equal(t, 0xFF, int((h>>24)&0xFF), "excCount at bits 24-31")
}

func TestFormatConformance_PayloadSize(t *testing.T) {
	// Verify utlPayloadBytesLUT matches the formula:
	// bitWidth == 0 -> 0, else ceil(bitWidth/4) * 64.
	for bw := 0; bw <= 32; bw++ {
		expected := 0
		if bw > 0 {
			expected = ((bw + 3) / 4) * 64
		}
		assert.Equal(t, expected, utlPayloadBytesLUT[bw],
			"payload size mismatch for bw=%d", bw)
	}
}

func TestFormatConformance_PayloadSizeSteps(t *testing.T) {
	// Verify payload sizes increase in 64-byte steps (one super-word per 4 bit-widths).
	for bw := 1; bw <= 32; bw++ {
		size := utlPayloadBytesLUT[bw]
		assert.Equal(t, 0, size%64, "payload must be multiple of 64 for bw=%d", bw)

		if bw > 1 && bw%4 == 1 {
			prev := utlPayloadBytesLUT[bw-1]
			assert.Equal(t, 64, size-prev,
				"payload should increase by 64 at bw=%d boundary", bw)
		}
	}
}

func TestFormatConformance_ExceptionTableLayout_SortedPositions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[10] = 0x10000
	values[50] = 0x1000000

	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	_, bw, _, excCount, hasExc, _, _, _ := decodeHeader(header)
	require.True(t, hasExc, "expected exceptions")
	require.LessOrEqual(t, excCount, excBitmapThreshold,
		"2 exceptions should use sorted-positions format")

	svbLen := int(bo.Uint16(packed[headerBytes:]))
	payloadEnd := headerBytes + svbLenBytes + utlPayloadBytesLUT[bw]
	positions := packed[payloadEnd : payloadEnd+excCount]

	assert.Equal(t, 2, excCount)
	assert.Greater(t, svbLen, 0)
	assert.True(t, sort.SliceIsSorted(positions, func(i, j int) bool {
		return positions[i] < positions[j]
	}), "exception positions must be sorted ascending")
}

func TestFormatConformance_ExceptionTableLayout_Bitmap(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	// Set 20 exceptions (> 16 threshold) to trigger bitmap format.
	for i := range 20 {
		values[i*6] = 0x10000000 + uint32(i)
	}

	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	_, bw, _, excCount, hasExc, _, _, _ := decodeHeader(header)
	require.True(t, hasExc, "expected exceptions")

	if excCount > excBitmapThreshold {
		payloadEnd := headerBytes + svbLenBytes + utlPayloadBytesLUT[bw]
		bitmap := packed[payloadEnd : payloadEnd+16]

		// Verify bitmap has the correct bits set for exception positions.
		setBits := 0
		for b := range 16 {
			for bit := range 8 {
				if bitmap[b]&(1<<bit) != 0 {
					setBits++
				}
			}
		}
		assert.Equal(t, excCount, setBits, "bitmap set bits must match excCount")
	}
}

func TestFormatConformance_BlockLengthMatchesActual(t *testing.T) {
	for bw := 0; bw <= 32; bw++ {
		values := make([]uint32, 128)
		mask := uint32((1 << bw) - 1)
		if bw == 32 {
			mask = 0xFFFFFFFF
		}
		for i := range values {
			values[i] = uint32(i) & mask
		}

		packed, err := PackUint32(0, nil, values)
		require.NoError(t, err)

		blockLen, err := BlockLength(packed)
		require.NoError(t, err)
		assert.Equal(t, len(packed), blockLen,
			"BlockLength must match actual packed size for bw=%d", bw)
	}
}

func TestFormatConformance_BlockLengthMatchesActual_WithExceptions(t *testing.T) {
	for excTarget := 1; excTarget <= 128; excTarget += 17 {
		values := make([]uint32, 128)
		for i := range values {
			values[i] = uint32(i)
		}
		for i := 0; i < excTarget && i < 128; i++ {
			idx := (i * 128 / excTarget) % 128
			values[idx] = 0x10000000 + uint32(i)
		}

		packed, err := PackUint32(0, nil, values)
		require.NoError(t, err)

		blockLen, err := BlockLength(packed)
		require.NoError(t, err)
		assert.Equal(t, len(packed), blockLen,
			"BlockLength must match actual for excTarget=%d", excTarget)
	}
}

func TestFormatConformance_BlockLengthMatchesActual_WithDelta(t *testing.T) {
	for bw := 0; bw <= 32; bw++ {
		values := make([]uint32, 128)
		mask := uint32((1 << bw) - 1)
		if bw == 32 {
			mask = 0xFFFFFFFF
		}
		for i := range values {
			values[i] = uint32(i) & mask
		}

		work := append([]uint32(nil), values...)
		packed, err := PackUint32(Delta, nil, work)
		require.NoError(t, err)

		blockLen, err := BlockLength(packed)
		require.NoError(t, err)
		assert.Equal(t, len(packed), blockLen,
			"BlockLength must match actual for delta bw=%d", bw)
	}
}

func TestFormatConformance_WireLayout_NoExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	_, bw, _, _, hasExc, _, _, _ := decodeHeader(header)
	require.False(t, hasExc)

	// Layout: [header:4][payload:N]
	expectedLen := headerBytes + utlPayloadBytesLUT[bw]
	assert.Equal(t, expectedLen, len(packed))
}

func TestFormatConformance_WireLayout_WithExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[10] = 0x10000
	values[50] = 0x1000000

	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	header := bo.Uint32(packed)
	_, bw, _, excCount, hasExc, _, _, _ := decodeHeader(header)
	require.True(t, hasExc)

	svbLen := int(bo.Uint16(packed[headerBytes:]))
	payloadBytes := utlPayloadBytesLUT[bw]
	excIndexSize := excCount
	if excCount > excBitmapThreshold {
		excIndexSize = 16
	}

	// Layout: [header:4][svbLen:2][payload:N][excIndex:M][svbData:svbLen]
	expectedLen := headerBytes + svbLenBytes + payloadBytes + excIndexSize + svbLen
	assert.Equal(t, expectedLen, len(packed))
}

func TestFormatConformance_ReservedBitsZero(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}

	for _, flag := range []byte{0, Delta} {
		work := append([]uint32(nil), values...)
		packed, err := PackUint32(flag, nil, work)
		require.NoError(t, err)

		header := bo.Uint32(packed)
		reserved := header & headerReservedMask
		assert.Equal(t, uint32(0), reserved,
			"bits 19-21 must be zero in current implementation (flag=%d)", flag)
	}
}

func TestKaitai_HeaderFields_Plain(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(128 + i)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	b := parseKaitaiBlock(t, packed)
	assert.Equal(t, 128, b.count)
	assert.Equal(t, IntTypeUint32, b.intType)
	assert.False(t, b.hasDelta)
	assert.False(t, b.hasZigZag)
	assert.False(t, b.hasExceptions)
	assert.Equal(t, 0, b.excCount)
	assert.Equal(t, utlPayloadBytesLUT[b.bitWidth], b.payloadSize)
}

func TestKaitai_HeaderFields_Delta(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	work := append([]uint32(nil), values...)
	packed, err := PackUint32(Delta, nil, work)
	require.NoError(t, err)

	b := parseKaitaiBlock(t, packed)
	assert.Equal(t, 128, b.count)
	assert.True(t, b.hasDelta)
	assert.Equal(t, IntTypeUint32, b.intType)
	assert.False(t, b.hasExceptions)
}

func TestKaitai_HeaderFields_WithExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	values[10] = 0x10000
	values[50] = 0x1000000
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	b := parseKaitaiBlock(t, packed)
	assert.Equal(t, 128, b.count)
	assert.True(t, b.hasExceptions)
	assert.Equal(t, 2, b.excCount)
	assert.Greater(t, int(b.svbLength), 0)

	assert.True(t, sort.SliceIsSorted(b.excIndex, func(i, j int) bool {
		return b.excIndex[i] < b.excIndex[j]
	}), "kaitai: exception positions must be sorted")
}

func TestKaitai_BitmapExceptions(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	for i := range 20 {
		values[i*6] = 0x10000000 + uint32(i)
	}
	packed, err := PackUint32(0, nil, values)
	require.NoError(t, err)

	b := parseKaitaiBlock(t, packed)
	if b.excCount > excBitmapThreshold {
		assert.Len(t, b.excIndex, 16, "kaitai: bitmap must be 16 bytes")

		setBits := 0
		for _, byt := range b.excIndex {
			setBits += bits.OnesCount8(byt)
		}
		assert.Equal(t, b.excCount, setBits, "kaitai: bitmap set bits must match excCount")
	}
}

func TestKaitai_PayloadSizeAllBitWidths(t *testing.T) {
	for bw := 0; bw <= 32; bw++ {
		values := make([]uint32, 128)
		mask := uint32((1 << bw) - 1)
		if bw == 32 {
			mask = 0xFFFFFFFF
		}
		for i := range values {
			values[i] = uint32(i) & mask
		}
		packed, err := PackUint32(0, nil, values)
		require.NoError(t, err, "bw=%d", bw)

		b := parseKaitaiBlock(t, packed)
		expected := 0
		if b.bitWidth > 0 {
			expected = ((b.bitWidth + 3) / 4) * 64
		}
		assert.Equal(t, expected, len(b.payload),
			"kaitai: payload size mismatch for bw=%d (actual bitWidth=%d)", bw, b.bitWidth)
	}
}

func TestKaitai_ReservedBitsZero(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	for _, flag := range []byte{0, Delta} {
		work := append([]uint32(nil), values...)
		packed, err := PackUint32(flag, nil, work)
		require.NoError(t, err)

		b := parseKaitaiBlock(t, packed)
		reserved := b.raw & headerReservedMask
		assert.Equal(t, uint32(0), reserved,
			"kaitai: reserved bits 19-21 must be zero (flag=%d)", flag)
	}
}

func TestKaitai_WireLayoutMatchesBlockLength(t *testing.T) {
	datasets := []struct {
		name   string
		values []uint32
		flag   byte
	}{
		{"plain_small", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i)
			}
			return v
		}(), 0},
		{"delta_sorted", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i * 100)
			}
			return v
		}(), Delta},
		{"with_2_exceptions", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i)
			}
			v[10] = 0x10000
			v[50] = 0x1000000
			return v
		}(), 0},
		{"with_20_exceptions", func() []uint32 {
			v := make([]uint32, 128)
			for i := range v {
				v[i] = uint32(i)
			}
			for i := range 20 {
				v[i*6] = 0x10000000 + uint32(i)
			}
			return v
		}(), 0},
	}
	for _, ds := range datasets {
		t.Run(ds.name, func(t *testing.T) {
			work := append([]uint32(nil), ds.values...)
			packed, err := PackUint32(ds.flag, nil, work)
			require.NoError(t, err)

			b := parseKaitaiBlock(t, packed)
			blockLen, err := BlockLength(packed)
			require.NoError(t, err)

			kaitaiLen := headerBytes + b.payloadSize
			if b.hasExceptions {
				kaitaiLen = headerBytes + svbLenBytes + b.payloadSize + len(b.excIndex) + int(b.svbLength)
			}
			assert.Equal(t, blockLen, kaitaiLen,
				"kaitai-computed length must match BlockLength")
			assert.Equal(t, len(packed), kaitaiLen,
				"kaitai-computed length must match actual packed size")
		})
	}
}

func TestKaitai_GoldenFilesParse(t *testing.T) {
	tests := []struct {
		name      string
		file      string
		wantCount int
		wantDelta bool
		wantExc   bool
	}{
		{"plain_128_bw8", "testdata/plain_128_bw8.bin", 128, false, false},
		{"plain_128_bw0", "testdata/plain_128_bw0.bin", 128, false, false},
		{"delta_128", "testdata/delta_128.bin", 128, true, false},
		{"exceptions_128", "testdata/exceptions_128.bin", 128, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := os.ReadFile(tt.file)
			if errors.Is(err, os.ErrNotExist) {
				t.Skipf("golden file %s not found", tt.file)
			}
			require.NoError(t, err)

			b := parseKaitaiBlock(t, data)
			assert.Equal(t, tt.wantCount, b.count, "count")
			assert.Equal(t, tt.wantDelta, b.hasDelta, "delta flag")
			assert.Equal(t, tt.wantExc, b.hasExceptions, "exceptions flag")
			assert.Equal(t, IntTypeUint32, b.intType, "int type")
		})
	}
}

func goldenBW8Values() []uint32 {
	v := make([]uint32, 128)
	for i := range v {
		v[i] = uint32(128 + i)
	}
	return v
}

func goldenBW0Values() []uint32 {
	return make([]uint32, 128)
}

func goldenDeltaValues() []uint32 {
	v := make([]uint32, 128)
	for i := range v {
		v[i] = uint32(i * 100)
	}
	return v
}

func goldenExceptionValues() []uint32 {
	v := make([]uint32, 128)
	for i := range v {
		v[i] = uint32(i)
	}
	excPositions := []int{5, 15, 25, 35, 45, 55, 65, 75, 85, 95}
	for _, p := range excPositions {
		v[p] = 0x10000000 + uint32(p)
	}
	return v
}

func TestGoldenVectors(t *testing.T) {
	tests := []struct {
		name   string
		file   string
		values []uint32
		flag   byte
	}{
		{"plain_128_bw8", "testdata/plain_128_bw8.bin", goldenBW8Values(), 0},
		{"plain_128_bw0", "testdata/plain_128_bw0.bin", goldenBW0Values(), 0},
		{"delta_128", "testdata/delta_128.bin", goldenDeltaValues(), Delta},
		{"exceptions_128", "testdata/exceptions_128.bin", goldenExceptionValues(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			work := append([]uint32(nil), tt.values...)
			packed, err := PackUint32(tt.flag, nil, work)
			require.NoError(t, err)

			golden, err := os.ReadFile(tt.file)
			if errors.Is(err, os.ErrNotExist) {
				require.NoError(t, os.WriteFile(tt.file, packed, 0644))
				t.Skipf("golden file %s created; re-run to validate", tt.file)
			}
			require.NoError(t, err)
			assert.Equal(t, golden, packed, "output diverged from golden file %s", tt.file)
		})
	}
}

func TestGoldenVectors_RoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		values []uint32
		flag   byte
	}{
		{"plain_128_bw8", goldenBW8Values(), 0},
		{"plain_128_bw0", goldenBW0Values(), 0},
		{"delta_128", goldenDeltaValues(), Delta},
		{"exceptions_128", goldenExceptionValues(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			work := append([]uint32(nil), tt.values...)
			packed, err := PackUint32(tt.flag, nil, work)
			require.NoError(t, err)

			unpacked, consumed, err := UnpackUint32(nil, make([]uint32, 128), packed)
			require.NoError(t, err)
			assert.Equal(t, tt.values, unpacked)

			blockLen, err := BlockLength(packed)
			require.NoError(t, err)
			assert.Equal(t, consumed, blockLen, "consumed must match BlockLength")
			assert.Equal(t, len(packed), blockLen, "packed length must match BlockLength")

			for pos, want := range unpacked {
				got, err := GetUint32(pos, packed)
				require.NoError(t, err)
				assert.Equal(t, want, got, "GetUint32 mismatch at pos=%d", pos)
			}
		})
	}
}
