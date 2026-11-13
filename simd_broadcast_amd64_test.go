//go:build goexperiment.simd && amd64

package utlpfor

import (
	"simd/archsimd"
	"testing"
)

// TestBroadcastHelpers checks that every lane of the AVX-only broadcast helpers holds x.
func TestBroadcastHelpers(t *testing.T) {
	if !archsimd.X86.AVX() {
		t.Skip("CPU lacks AVX")
	}
	vals32 := []uint32{0, 1, 0xFF, 0x80000000, 0xDEADBEEF, 0xFFFFFFFF}
	for _, x := range vals32 {
		var got [4]uint32
		broadcastUint32x4(x).StoreArray(&got)
		for i, g := range got {
			if g != x {
				t.Errorf("broadcastUint32x4(%#x): lane %d = %#x", x, i, g)
			}
		}
	}
	vals64 := []uint64{0, 1, 0xFF, 1 << 32, 0x8000000000000000, 0xDEADBEEFCAFEF00D, ^uint64(0)}
	for _, x := range vals64 {
		var got [2]uint64
		broadcastUint64x2(x).StoreArray(&got)
		for i, g := range got {
			if g != x {
				t.Errorf("broadcastUint64x2(%#x): lane %d = %#x", x, i, g)
			}
		}
	}
}

var (
	sinkBroadcast32 [4]uint32
	sinkBroadcast64 [2]uint64
)

// broadcastUint32x4ViaArray is the old helper (four 4-byte stores, one 16-byte load: a
// store-forwarding stall). It is kept only as the baseline of BenchmarkBroadcast.
func broadcastUint32x4ViaArray(x uint32) archsimd.Uint32x4 {
	v := [4]uint32{x, x, x, x}
	return archsimd.LoadUint32x4Array(&v)
}

func broadcastUint64x2ViaArray(x uint64) archsimd.Uint64x2 {
	v := [2]uint64{x, x}
	return archsimd.LoadUint64x2Array(&v)
}

// BenchmarkBroadcast compares the helpers (shuffle) with the old array-load version. On Sandy
// Bridge the array variant is several times slower (store-forwarding stall).
func BenchmarkBroadcast(b *testing.B) {
	if !archsimd.X86.AVX() {
		b.Skip("CPU lacks AVX")
	}
	b.Run("u32/shuffle", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			broadcastUint32x4(uint32(i)).StoreArray(&sinkBroadcast32)
		}
	})
	b.Run("u32/array", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			broadcastUint32x4ViaArray(uint32(i)).StoreArray(&sinkBroadcast32)
		}
	})
	b.Run("u64/shuffle", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			broadcastUint64x2(uint64(i)).StoreArray(&sinkBroadcast64)
		}
	})
	b.Run("u64/array", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			broadcastUint64x2ViaArray(uint64(i)).StoreArray(&sinkBroadcast64)
		}
	})
}
