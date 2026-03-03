package utlpfor

// UnpackUint32 decodes a packed block into dst, using scratch as workspace.
// Returns the populated dst slice, the number of bytes consumed, and any error.
func UnpackUint32(dst []uint32, scratch []uint32, buf []byte) ([]uint32, int, error) {
	if len(buf) < 4 {
		return nil, 0, ErrInvalidBuffer
	}
	return nil, 0, ErrNotImplemented
}
