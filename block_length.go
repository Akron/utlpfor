package utlpfor

// BlockLength returns the total byte length of the encoded block
// starting at the beginning of buf.
func BlockLength(buf []byte) (int, error) {
	if len(buf) < 4 {
		return 0, ErrInvalidBuffer
	}
	return 0, ErrNotImplemented
}
