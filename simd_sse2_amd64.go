//go:build goexperiment.simd && amd64

package utlpfor

// SSE2 stubs

func packUint32SSE2(flag byte, dst []byte, values []uint32) ([]byte, error) {
	return packUint32Scalar(flag, dst, values)
}

func unpackUint32SSE2(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	return unpackUint32Scalar(dst, scratch, buf)
}

func getUint32SSE2(pos int, buf []byte) (uint32, error) {
	return getUint32Scalar(pos, buf)
}
