//go:build goexperiment.simd && amd64

package utlpfor

// AVX-512 stubs

func packUint32AVX512(flag byte, dst []byte, values []uint32) ([]byte, error) {
	return packUint32Scalar(flag, dst, values)
}

func unpackUint32AVX512(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackUint32Scalar(dst, scratch, buf)
}

func getUint32AVX512(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
