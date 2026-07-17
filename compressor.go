package utlpfor

// compressor handles zero-allocation UTL-PFOR compression and
// decompression of uint32 blocks. It owns a reusable scratch buffer
// so callers do not need to manage one.
//
// A single compressor is used for both compression and decompression.
// It is not safe for concurrent use; each goroutine should create its own.
//
// Common flag combinations for [compressor.Compress]:
//
//	c.Compress(0, dst, values)                        // raw
//	c.Compress(Delta, dst, values)                    // delta-encoded
//	c.Compress(Delta|Append, dst, values)             // delta, appended to dst
//	c.Compress(NoPatch|NoFOR, dst, values)            // dictionary data
type compressor struct {
	scratch []uint32
}

// NewUint32 creates a [compressor] with a pre-allocated scratch buffer
// for uint32 block operations. The scratch buffer is sized to support
// NoInPlace and Append without internal allocation.
func NewUint32() *compressor {
	return &compressor{
		scratch: make([]uint32, ScratchLenNoInPlace),
	}
}

// Compress encodes values into a packed block.
// If dst has sufficient capacity, it is reused; otherwise a new slice
// is allocated.
// When flags includes [Append], the block is written after the existing
// content of dst; otherwise dst is overwritten from index 0.
// When flags includes [Append] or [NoInPlace], the values slice is
// guaranteed unmodified after the call. Without these flags, values
// may be modified in-place (FOR subtraction, delta encoding).
func (c *compressor) Compress(flags Flag, dst []byte, values []uint32) ([]byte, error) {
	return PackUint32(flags, values, dst, c.scratch)
}

// Decompress decodes a packed block from src into dst.
// If dst has sufficient capacity it is reused; otherwise a new slice
// is allocated. Pass dst as nil or dst[:0] to let the codec set the
// length.
// Returns the populated slice, the number of source bytes consumed,
// and any error.
func (c *compressor) Decompress(dst []uint32, src []byte) ([]uint32, int, error) {
	return UnpackUint32(src, dst, c.scratch)
}

// Get extracts a single value at the given position from the packed
// block without full decompression. This is useful for random access
// into compressed data.
func (c *compressor) Get(pos int, src []byte) (uint32, error) {
	return GetUint32(pos, src, c.scratch)
}

// compressor64 handles zero-allocation UTL-PFOR compression and
// decompression of uint64 blocks. It owns a reusable scratch buffer
// so callers do not need to manage one.
//
// A single compressor64 is used for both compression and decompression.
// It is not safe for concurrent use; each goroutine should create its own.
//
// Common flag combinations for [compressor64.Compress]:
//
//	c.Compress(0, dst, values)                        // raw
//	c.Compress(Delta, dst, values)                    // delta-encoded
//	c.Compress(Delta|Append, dst, values)             // delta, appended to dst
type compressor64 struct {
	scratch []uint32
}

// NewUint64 creates a [compressor64] with a pre-allocated scratch buffer
// for uint64 block operations.
func NewUint64() *compressor64 {
	return &compressor64{
		scratch: make([]uint32, ScratchLen64),
	}
}

// Compress encodes values into a packed block.
// If dst has sufficient capacity, it is reused; otherwise a new slice
// is allocated.
// When flags includes [Append], the block is written after the existing
// content of dst; otherwise dst is overwritten from index 0.
// The values slice is read but not modified.
func (c *compressor64) Compress(flags Flag, dst []byte, values []uint64) ([]byte, error) {
	return PackUint64(flags, values, dst, c.scratch)
}

// Decompress decodes a packed uint64 block from src into dst.
// If dst has sufficient capacity it is reused; otherwise a new slice
// is allocated. Pass dst as nil or dst[:0] to let the codec set the
// length.
// Returns the populated slice, the number of source bytes consumed,
// and any error.
func (c *compressor64) Decompress(dst []uint64, src []byte) ([]uint64, int, error) {
	return UnpackUint64(src, dst, c.scratch)
}

// Get extracts a single uint64 value at the given position from the
// packed block without full decompression. This is useful for random
// access into compressed data.
func (c *compressor64) Get(pos int, src []byte) (uint64, error) {
	return GetUint64(pos, src, c.scratch)
}
