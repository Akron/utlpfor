package utlpfor

// GetUint32 extracts a single value at the given position from the packed block.
func GetUint32(pos int, buf []byte) (uint32, error) {
	if len(buf) < 4 {
		return 0, ErrInvalidBuffer
	}
	return 0, ErrNotImplemented
}
