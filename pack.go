package utlpfor

// PackUint32 encodes uint32 values into a packed block appended to dst.
// The flag byte controls encoding options (e.g. Delta for delta encoding).
func PackUint32(flag byte, dst []byte, values []uint32) ([]byte, error) {
	return nil, ErrNotImplemented
}
