package utlpfor

import (
	"os"
	"strings"
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
		case "avx", "sse2": // "avx" is an alias of "sse2"
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
		return "SSE2 (128-bit kernels, needs AVX)"
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

	unpacked, _, err := unpackUint32Scalar(nil, scratch, packed, false)
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

	apiPacked, err := PackUint32(0, values, nil, nil)
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
	apiPacked, err := PackUint32(Delta, apiWork, nil, nil)
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

	packed, err := PackUint32(0, values, nil, nil)
	require.NoError(t, err)

	scalarOut, scalarConsumed, err := unpackUint32Scalar(nil, make([]uint32, 128), packed, false)
	require.NoError(t, err)

	apiOut, apiConsumed, err := UnpackUint32(packed, nil, make([]uint32, 128))
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

	packed, err := PackUint32(0, values, nil, nil)
	require.NoError(t, err)

	scratch := make([]uint32, ScratchLen)
	for pos := range values {
		scalarVal, err := getUint32Scalar(pos, packed, scratch, false)
		require.NoError(t, err)

		apiVal, err := GetUint32(pos, packed, scratch)
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

func TestSelectSimdLevel_AVX(t *testing.T) {
	// AVX without AVX2 (Sandy Bridge, Ivy Bridge, Bulldozer ...): the 128-bit
	// kernels, which must not need AVX2.
	assert.Equal(t, simdLevelSSE2, selectSimdLevel(false, false, false, true))
}

func TestSelectSimdLevel_Scalar(t *testing.T) {
	// SSE-only CPU (no AVX, e.g. Nehalem or a qemu64 guest): archsimd cannot
	// run there, so the scalar level is selected - silently.
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
		dst, _ = PackUint32(0, values, dst[:0], nil)
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

// TestDispatch_HonoursGODEBUGCPU checks the real dispatch against a simulated
// CPU (make test-cpu-avx-only / test-cpu-sse-only):
//
//	GODEBUG=cpu.avx2=off            -> at most the 128-bit (AVX) level
//	GODEBUG=cpu.avx=off             -> scalar (SSE-only CPU), no crash
func TestDispatch_HonoursGODEBUGCPU(t *testing.T) {
	if os.Getenv("UTL_SIMD_LEVEL") != "" {
		t.Skip("level is forced by UTL_SIMD_LEVEL")
	}
	godebug := "," + os.Getenv("GODEBUG") + ","
	switch {
	case strings.Contains(godebug, ",cpu.avx=off,"):
		assert.Equal(t, simdLevelScalar, simdLevel,
			"a CPU without AVX must run the scalar code")
	case strings.Contains(godebug, ",cpu.avx2=off,"):
		assert.LessOrEqual(t, simdLevel, simdLevelSSE2,
			"a CPU without AVX2 must not use the AVX2 kernels")
	default:
		t.Skip("no simulated CPU in GODEBUG")
	}
}

// TestDispatch_ExpectedLevel is the proof for a run on a real machine (see
// "make build-remote-test"). With UTL_EXPECT_SIMD_LEVEL set it FAILS unless the
// level selected by the real CPU detection (and not forced by UTL_SIMD_LEVEL)
// is the expected one:
//
//	UTL_EXPECT_SIMD_LEVEL=scalar   CPU without AVX (SSE only), or a build without GOEXPERIMENT=simd
//	UTL_EXPECT_SIMD_LEVEL=sse2     128-bit kernels (CPU with AVX, without AVX2); "avx" is an alias
//	UTL_EXPECT_SIMD_LEVEL=avx2     AVX2 kernels
//	UTL_EXPECT_SIMD_LEVEL=avx512   AVX-512 kernels (VBMI included)
//
// The CPU flags found in /proc/cpuinfo are logged as evidence.
func TestDispatch_ExpectedLevel(t *testing.T) {
	want := os.Getenv("UTL_EXPECT_SIMD_LEVEL")
	if want == "" {
		t.Skip("UTL_EXPECT_SIMD_LEVEL not set")
	}
	if os.Getenv("UTL_SIMD_LEVEL") != "" {
		t.Fatal("UTL_SIMD_LEVEL forces the level; unset it to test the real detection")
	}
	if data, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "flags") {
				var found []string
				have := " " + line + " "
				for _, f := range []string{"sse2", "sse4_2", "avx", "avx2", "avx512f", "avx512vbmi"} {
					if strings.Contains(have, " "+f+" ") {
						found = append(found, f)
					}
				}
				t.Logf("cpu flags (subset): %s", strings.Join(found, " "))
				break
			}
		}
	}
	t.Logf("detected SIMD level: %s (%d)", simdLevelName(simdLevel), simdLevel)

	switch want {
	case "scalar":
		assert.Equal(t, simdLevelScalar, simdLevel)
	case "sse2", "avx":
		assert.Equal(t, simdLevelSSE2, simdLevel,
			"expected the 128-bit kernels (this fails on a CPU without AVX or with AVX2)")
	case "avx2":
		assert.Equal(t, simdLevelAVX2, simdLevel)
	case "avx512":
		assert.GreaterOrEqual(t, simdLevel, simdLevelAVX512)
	default:
		t.Fatalf("unknown UTL_EXPECT_SIMD_LEVEL %q", want)
	}
}

// TestDispatch_ActiveLevelRoundTrip packs and unpacks at every level that this
// CPU (or the simulated CPU) supports.
func TestDispatch_ActiveLevelRoundTrip(t *testing.T) {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i*2654435761) >> (i % 29)
	}
	packed, err := PackUint32(0, values, nil, nil)
	require.NoError(t, err)
	got, _, err := UnpackUint32(packed, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, values, got, "level %s", simdLevelName(simdLevel))
}
