package utlpfor

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMain(m *testing.M) {
	if env := os.Getenv("UTL_SIMD_LEVEL"); env != "" {
		detected := simdLevel
		switch env {
		case "scalar":
			simdLevel = simdLevelScalar
		case "sse2":
			if detected < simdLevelSSE2 {
				os.Exit(0)
			}
			simdLevel = simdLevelSSE2
		case "avx2":
			if detected < simdLevelAVX2 {
				os.Exit(0)
			}
			simdLevel = simdLevelAVX2
		case "avx512":
			if detected < simdLevelAVX512 {
				os.Exit(0)
			}
			simdLevel = simdLevelAVX512
		}
	}
	os.Exit(m.Run())
}

func simdLevelName(level int) string {
	switch level {
	case simdLevelScalar:
		return "Scalar"
	case simdLevelSSE2:
		return "SSE2"
	case simdLevelAVX2:
		return "AVX2"
	case simdLevelAVX512:
		return "AVX-512"
	case simdLevelAVX512VBMI:
		return "AVX-512 VBMI"
	default:
		return "Unknown"
	}
}

func TestDispatch_SimdLevelIsSet(t *testing.T) {
	assert.GreaterOrEqual(t, simdLevel, simdLevelScalar)
	assert.LessOrEqual(t, simdLevel, simdLevelAVX512VBMI)

	t.Logf("Active SIMD level: %s (%d)", simdLevelName(simdLevel), simdLevel)
	t.Logf("SIMD levels testable on this system:")
	t.Logf("  Scalar:       always available")
	for _, lvl := range []int{simdLevelSSE2, simdLevelAVX2, simdLevelAVX512, simdLevelAVX512VBMI} {
		if simdLevel >= lvl {
			t.Logf("  %-14s available", simdLevelName(lvl)+":")
		} else {
			t.Logf("  %-14s not available", simdLevelName(lvl)+":")
		}
	}
}

func TestDispatch_ScalarAlwaysAvailable(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	scratch := make([]uint32, blockSize)

	packed, err := packUint32Scalar(0, nil, scratch, values)
	require.NoError(t, err)

	unpacked, _, err := unpackUint32Scalar(nil, scratch, packed)
	require.NoError(t, err)
	assert.Equal(t, values, unpacked)
}

func TestDispatch_AllLevelsProduceIdenticalOutput(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i*7 + 3)
	}
	scratch := make([]uint32, blockSize)

	scalarPacked, err := packUint32Scalar(0, nil, scratch, values)
	require.NoError(t, err)

	apiPacked, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)
	assert.Equal(t, scalarPacked, apiPacked,
		"API output must match scalar output")
}

func TestDispatch_AllLevelsProduceIdenticalOutput_Delta(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 100)
	}
	scratch := make([]uint32, blockSize)
	scalarWork := append([]uint32(nil), values...)
	scalarPacked, err := packUint32Scalar(Delta, nil, scratch, scalarWork)
	require.NoError(t, err)

	apiWork := append([]uint32(nil), values...)
	apiPacked, err := PackUint32(Delta, nil, nil, apiWork)
	require.NoError(t, err)
	assert.Equal(t, scalarPacked, apiPacked,
		"API delta output must match scalar delta output")
}

func TestDispatch_UnpackMatchesScalar(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 7)
	}
	values[10] = 0x100000

	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)

	scalarOut, scalarConsumed, err := unpackUint32Scalar(nil, make([]uint32, 128), packed)
	require.NoError(t, err)

	apiOut, apiConsumed, err := UnpackUint32(nil, make([]uint32, 128), packed)
	require.NoError(t, err)

	assert.Equal(t, scalarOut, apiOut)
	assert.Equal(t, scalarConsumed, apiConsumed)
}

func TestDispatch_GetMatchesScalar(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 7)
	}
	values[10] = 0x100000

	packed, err := PackUint32(0, nil, nil, values)
	require.NoError(t, err)

	for pos := range values {
		scalarVal, err := getUint32Scalar(pos, packed)
		require.NoError(t, err)

		apiVal, err := GetUint32(pos, packed)
		require.NoError(t, err)

		assert.Equal(t, scalarVal, apiVal, "pos=%d", pos)
	}
}

func TestSelectSimdLevel_AVX512VBMI(t *testing.T) {
	assert.Equal(t, simdLevelAVX512VBMI, selectSimdLevel(true, true, true, true))
}

func TestSelectSimdLevel_AVX512(t *testing.T) {
	assert.Equal(t, simdLevelAVX512, selectSimdLevel(false, true, true, true))
}

func TestSelectSimdLevel_AVX2(t *testing.T) {
	assert.Equal(t, simdLevelAVX2, selectSimdLevel(false, false, true, true))
}

func TestSelectSimdLevel_SSE2(t *testing.T) {
	assert.Equal(t, simdLevelSSE2, selectSimdLevel(false, false, false, true))
}

func TestSelectSimdLevel_Scalar(t *testing.T) {
	assert.Equal(t, simdLevelScalar, selectSimdLevel(false, false, false, false))
}

func TestSelectSimdLevel_PreferHighest(t *testing.T) {
	assert.Equal(t, simdLevelAVX512VBMI,
		selectSimdLevel(true, true, true, true),
		"must prefer AVX512VBMI when all features are available")
	assert.Equal(t, simdLevelAVX512,
		selectSimdLevel(false, true, true, true),
		"must prefer AVX512 when VBMI is missing")
}

func BenchmarkDispatchOverhead(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	dst := make([]byte, 0, 1024)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst, _ = PackUint32(0, dst[:0], nil, values)
	}
}

func BenchmarkScalarDirect(b *testing.B) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i)
	}
	dst := make([]byte, 0, 1024)
	scratch := make([]uint32, blockSize)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst, _ = packUint32Scalar(0, dst[:0], scratch, values)
	}
}
