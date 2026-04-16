package utlpfor

import (
	"slices"
	"testing"

	fastpfor "github.com/Akron/fastpfor-go"
)

func makeCompPlainValues() []uint32 {
	v := make([]uint32, 128)
	for i := range v {
		v[i] = uint32(i % 200)
	}
	return v
}

func makeCompDeltaValues() []uint32 {
	v := make([]uint32, 128)
	for i := range v {
		v[i] = 1000 + uint32(i*3)
	}
	return v
}

func makeCompExceptionValues() []uint32 {
	v := make([]uint32, 128)
	for i := range v {
		v[i] = uint32(i % 10)
	}
	v[10] = 0x10000000
	v[50] = 0x20000000
	v[99] = 0xFFFFFF
	return v
}

func makeCompMixedValues() []uint32 {
	v := make([]uint32, 128)
	for i := range v {
		switch i % 8 {
		case 0:
			v[i] = uint32(i % 256)
		case 1:
			v[i] = uint32(i*257) % 65536
		case 2:
			v[i] = uint32(i*65537) % 16777216
		case 3:
			v[i] = uint32(i) * 16843009
		case 4:
			v[i] = 0
		case 5:
			v[i] = uint32(i * 100)
		case 6:
			v[i] = uint32(i*i) % 1000000
		case 7:
			v[i] = 0xFFFFFFFF - uint32(i)
		}
	}
	return v
}

func BenchmarkCompare_Pack_Plain_UTL(b *testing.B) {
	values := makeCompPlainValues()
	scratch := make([]uint32, ScratchLen)
	dst := make([]byte, 0, 1024)
	b.SetBytes(int64(len(values) * 4))
	b.ReportAllocs()
	for b.Loop() {
		dst, _ = PackUint32(0, values, dst[:0], scratch)
	}
}

func BenchmarkCompare_Pack_Plain_BP128(b *testing.B) {
	values := makeCompPlainValues()
	b.SetBytes(int64(len(values) * 4))
	for b.Loop() {
		fastpfor.PackUint32(nil, values)
	}
}

func BenchmarkCompare_Pack_Delta_UTL(b *testing.B) {
	source := makeCompDeltaValues()
	data := make([]uint32, len(source))
	scratch := make([]uint32, ScratchLen)
	dst := make([]byte, 0, 1024)
	b.SetBytes(int64(len(source) * 4))
	b.ReportAllocs()
	for b.Loop() {
		copy(data, source)
		dst, _ = PackUint32(Delta, data, dst[:0], scratch)
	}
}

func BenchmarkCompare_Pack_Delta_BP128(b *testing.B) {
	source := makeCompDeltaValues()
	data := make([]uint32, len(source))
	b.SetBytes(int64(len(source) * 4))
	for b.Loop() {
		copy(data, source)
		fastpfor.PackDeltaUint32(nil, data)
	}
}

func BenchmarkCompare_Pack_Exceptions_UTL(b *testing.B) {
	values := makeCompExceptionValues()
	scratch := make([]uint32, ScratchLen)
	dst := make([]byte, 0, 1024)
	b.SetBytes(int64(len(values) * 4))
	b.ReportAllocs()
	for b.Loop() {
		dst, _ = PackUint32(0, values, dst[:0], scratch)
	}
}

func BenchmarkCompare_Pack_Exceptions_BP128(b *testing.B) {
	values := makeCompExceptionValues()
	b.SetBytes(int64(len(values) * 4))
	for b.Loop() {
		fastpfor.PackUint32(nil, values)
	}
}

func BenchmarkCompare_Pack_Mixed_UTL(b *testing.B) {
	values := makeCompMixedValues()
	scratch := make([]uint32, ScratchLen)
	dst := make([]byte, 0, 1024)
	b.SetBytes(int64(len(values) * 4))
	b.ReportAllocs()
	for b.Loop() {
		dst, _ = PackUint32(0, values, dst[:0], scratch)
	}
}

func BenchmarkCompare_Pack_Mixed_BP128(b *testing.B) {
	values := makeCompMixedValues()
	b.SetBytes(int64(len(values) * 4))
	for b.Loop() {
		fastpfor.PackUint32(nil, values)
	}
}

func BenchmarkCompare_Unpack_Plain_UTL(b *testing.B) {
	values := makeCompPlainValues()
	packed, _ := PackUint32(0, values, nil, nil)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(values) * 4))
	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkCompare_Unpack_Plain_BP128(b *testing.B) {
	values := makeCompPlainValues()
	packed := fastpfor.PackUint32(nil, values)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(values) * 4))
	for b.Loop() {
		fastpfor.UnpackUint32WithBufferAndLength(dst, scratch, packed)
	}
}

func BenchmarkCompare_Unpack_Delta_UTL(b *testing.B) {
	source := makeCompDeltaValues()
	data := slices.Clone(source)
	packed, _ := PackUint32(Delta, data, nil, nil)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(source) * 4))
	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkCompare_Unpack_Delta_BP128(b *testing.B) {
	source := makeCompDeltaValues()
	data := slices.Clone(source)
	packed := fastpfor.PackDeltaUint32(nil, data)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(source) * 4))
	for b.Loop() {
		fastpfor.UnpackUint32WithBufferAndLength(dst, scratch, packed)
	}
}

func makeCompDeltaZigzagValues() []uint32 {
	v := make([]uint32, 128)
	for i := range v {
		v[i] = 5000 + uint32(i*7) - uint32((i%5)*3)
	}
	return v
}

func BenchmarkCompare_Unpack_DeltaZigzag_UTL(b *testing.B) {
	source := makeCompDeltaZigzagValues()
	data := slices.Clone(source)
	packed, _ := PackUint32(Delta, data, nil, nil)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(source) * 4))
	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkCompare_Unpack_DeltaZigzag_BP128(b *testing.B) {
	source := makeCompDeltaZigzagValues()
	data := slices.Clone(source)
	packed := fastpfor.PackDeltaUint32(nil, data)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(source) * 4))
	for b.Loop() {
		fastpfor.UnpackUint32WithBufferAndLength(dst, scratch, packed)
	}
}

func BenchmarkCompare_Unpack_Exceptions_UTL(b *testing.B) {
	values := makeCompExceptionValues()
	packed, _ := PackUint32(0, values, nil, nil)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(values) * 4))
	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkCompare_Unpack_Exceptions_BP128(b *testing.B) {
	values := makeCompExceptionValues()
	packed := fastpfor.PackUint32(nil, values)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(values) * 4))
	for b.Loop() {
		fastpfor.UnpackUint32WithBufferAndLength(dst, scratch, packed)
	}
}

func BenchmarkCompare_Unpack_Mixed_UTL(b *testing.B) {
	values := makeCompMixedValues()
	packed, _ := PackUint32(0, values, nil, nil)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(values) * 4))
	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkCompare_Unpack_Mixed_BP128(b *testing.B) {
	values := makeCompMixedValues()
	packed := fastpfor.PackUint32(nil, values)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.SetBytes(int64(len(values) * 4))
	for b.Loop() {
		fastpfor.UnpackUint32WithBufferAndLength(dst, scratch, packed)
	}
}

func BenchmarkCompare_Get_Plain_UTL(b *testing.B) {
	values := makeCompPlainValues()
	packed, _ := PackUint32(0, values, nil, nil)
	scratch := make([]uint32, ScratchLen)
	for b.Loop() {
		GetUint32(64, packed, scratch)
	}
}

func BenchmarkCompare_Get_Plain_BP128(b *testing.B) {
	values := makeCompPlainValues()
	packed := fastpfor.PackUint32(nil, values)
	for b.Loop() {
		fastpfor.GetUint32(64, packed)
	}
}

func BenchmarkCompare_Get_Delta_UTL(b *testing.B) {
	source := makeCompDeltaValues()
	data := slices.Clone(source)
	packed, _ := PackUint32(Delta, data, nil, nil)
	scratch := make([]uint32, ScratchLen)
	for b.Loop() {
		GetUint32(64, packed, scratch)
	}
}

func BenchmarkCompare_Get_Delta_BP128(b *testing.B) {
	source := makeCompDeltaValues()
	data := slices.Clone(source)
	packed := fastpfor.PackDeltaUint32(nil, data)
	for b.Loop() {
		fastpfor.GetUint32(64, packed)
	}
}

func BenchmarkCompare_Get_Exceptions_UTL(b *testing.B) {
	values := makeCompExceptionValues()
	packed, _ := PackUint32(0, values, nil, nil)
	scratch := make([]uint32, ScratchLen)
	for b.Loop() {
		GetUint32(50, packed, scratch)
	}
}

func BenchmarkCompare_Get_Exceptions_BP128(b *testing.B) {
	values := makeCompExceptionValues()
	packed := fastpfor.PackUint32(nil, values)
	for b.Loop() {
		fastpfor.GetUint32(50, packed)
	}
}

func BenchmarkCompare_BlockLength_UTL(b *testing.B) {
	values := makeCompPlainValues()
	packed, _ := PackUint32(0, values, nil, nil)
	for b.Loop() {
		BlockLength(packed)
	}
}

func BenchmarkCompare_BlockLength_BP128(b *testing.B) {
	values := makeCompPlainValues()
	packed := fastpfor.PackUint32(nil, values)
	for b.Loop() {
		fastpfor.BlockLength(packed)
	}
}

func BenchmarkCompare_Allocs_Pack_UTL(b *testing.B) {
	values := makeCompMixedValues()
	b.ReportAllocs()
	for b.Loop() {
		PackUint32(0, values, nil, nil)
	}
}

func BenchmarkCompare_Allocs_Pack_UTL_Scratch(b *testing.B) {
	values := makeCompMixedValues()
	scratch := make([]uint32, ScratchLen)
	dst := make([]byte, 0, 1024)
	b.ReportAllocs()
	for b.Loop() {
		dst, _ = PackUint32(0, values, dst[:0], scratch)
	}
}

func BenchmarkCompare_Allocs_Pack_BP128(b *testing.B) {
	values := makeCompMixedValues()
	b.ReportAllocs()
	for b.Loop() {
		fastpfor.PackUint32(nil, values)
	}
}

func BenchmarkCompare_Allocs_Unpack_UTL(b *testing.B) {
	values := makeCompMixedValues()
	packed, _ := PackUint32(0, values, nil, nil)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.ReportAllocs()
	for b.Loop() {
		UnpackUint32(packed, dst, scratch)
	}
}

func BenchmarkCompare_Allocs_Unpack_BP128(b *testing.B) {
	values := makeCompMixedValues()
	packed := fastpfor.PackUint32(nil, values)
	dst := make([]uint32, 128)
	scratch := make([]uint32, 128)
	b.ReportAllocs()
	for b.Loop() {
		fastpfor.UnpackUint32WithBufferAndLength(dst, scratch, packed)
	}
}
