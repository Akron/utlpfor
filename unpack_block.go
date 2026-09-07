package utlpfor

// Shared unpack-block prologue.
//
// Every unpack pipeline (scalar, SSE2, AVX2, AVX-512) calls
// decodeUnpackHeader once per block to decode the 4-byte header, validate
// the int-type, compute payload/exception region offsets, and prove that
// the buffer is long enough.  Having a single implementation guarantees
// that all entry points accept exactly the same set of blocks and slice
// the buffer identically.
//
// Implementation note: the helper writes results through an
// unpackHeaderOut pointer instead of returning a struct.  Go cannot
// inline functions that return large structs, and a 96-byte struct return
// measured as a per-block regression (+28 % geomean) in A/B benchmarks.
// The boolean block attributes are packed into one uint8 flags field for
// the same reason (fewer fields = smaller struct = less copy overhead).

// Flag bits in unpackHeaderOut.flags.
const (
	hfExceptions = 1 << iota // block has an SVB exception region
	hfDelta                  // block is delta encoded
	hfZigZag                 // delta values are zigzag encoded (no overflow)
	hfFOR                    // block carries a frame-of-reference base
	hfU64Single              // FOR64 single-block (8-byte base, caller adds base)
	hfEmpty                  // count == 0: caller returns dst[:0], headerBytes
)

// unpackHeaderOut carries the decoded header fields needed by the unpack
// epilogue (bit-unpack, exceptions, delta-decode, FOR-add).
// All offsets are relative to the start of the caller's original buf.
type unpackHeaderOut struct {
	consumed int    // pOff + payloadBytes; also the exception-region start offset
	pOff     int    // payload start offset within buf
	forBase  uint32 // decoded FOR base (0 when hasFOR == false)
	count    int    // number of values in the block
	bitWidth int    // step bitwidth (0..32)
	excCount int    // number of exceptions (0 when none)
	flags    uint8  // hf* bit flags
}

func (h *unpackHeaderOut) hasExceptions() bool { return h.flags&hfExceptions != 0 }
func (h *unpackHeaderOut) hasDelta() bool      { return h.flags&hfDelta != 0 }
func (h *unpackHeaderOut) hasZigZag() bool     { return h.flags&hfZigZag != 0 }
func (h *unpackHeaderOut) hasFOR() bool        { return h.flags&hfFOR != 0 }
func (h *unpackHeaderOut) u64Single() bool     { return h.flags&hfU64Single != 0 }
func (h *unpackHeaderOut) empty() bool         { return h.flags&hfEmpty != 0 }

// decodeUnpackHeader decodes and validates the block header in buf.
//
// Parameters:
//   - forUint64: when true, uses the uint64 int-type validator and enables
//     combine-flag handling (double-block uint64 mode).
//   - out: receives the decoded header fields on success.
//
// On success the returned []byte is buf[:blockEnd], proving that all payload
// and exception-region slices are in bounds.  On error the original buf is
// returned unchanged.
//
// The caller must check out.empty() after a nil error: an empty block
// (count == 0) is valid but contains no payload.
func decodeUnpackHeader(buf []byte, forUint64 bool, out *unpackHeaderOut) ([]byte, error) {
	if len(buf) < headerBytes {
		return buf, ErrInvalidBuffer
	}

	// Decode the 4-byte header to extract all block parameters.
	header := bo.Uint32(buf)
	count, bitWidth, intType, excCount, forWidth, hasExceptions, hasDelta, hasZigZag, _, hasCombine := decodeHeader(header)
	hasFOR := forWidth > 0

	// Validate int type based on caller context (uint32 vs uint64 API).
	if forUint64 {
		if err := validateIntType64(intType); err != nil {
			return buf, err
		}
	} else {
		if err := validateIntType(intType); err != nil {
			return buf, err
		}
		hasCombine = false // combine is never valid for uint32 blocks
	}

	// Range check: valid blocks have 1..128 values and bitWidth 0..32.
	// The unsigned trick (count-1 >= blockSize) catches count==0 cheaply:
	// uint(0-1) wraps to MaxUint, which is >= 128.
	if uint(count-1) >= blockSize || uint(bitWidth) > 32 {
		if count == 0 {
			out.flags = hfEmpty
			return buf, nil
		}
		if count > blockSize {
			return buf, ErrInvalidBlockLength
		}
		return buf, ErrInvalidBuffer
	}

	// FOR64 single-block: forWidth=3 means 8-byte base, handled by uint64 caller.
	u64Single := forUint64 && isFor64SingleBlock(intType, forWidth, hasCombine)

	// Compute payload start: header [+ svbLen] [+ forBase] [+ block2Len].
	pOff := headerBytes
	if hasExceptions {
		pOff += svbLenBytes
	}
	forBaseOff := pOff
	if hasFOR {
		if u64Single {
			pOff += for64BaseSize
		} else {
			pOff += forBaseBytes(forWidth)
		}
	}
	if hasCombine {
		pOff += block2LenBytes
	}

	payloadBytes := utlPayloadBytes(bitWidth)
	excIdxSize := 0
	if hasExceptions {
		excIdxSize = excIndexSize(excCount)
	}
	// Single length check covers the FOR base region, the payload, and the
	// exception index; the reslice proves all later slices in range.
	blockEnd := pOff + payloadBytes + excIdxSize
	if len(buf) < blockEnd {
		return buf, ErrInvalidBuffer
	}
	if hasExceptions {
		// Extend the proven region by the SVB data length read at the
		// fixed offset (guaranteed present, blockEnd >= 6).
		svbLen := int(bo.Uint16(buf[headerBytes:]))
		if blockEnd+svbLen > len(buf) {
			return buf, ErrInvalidBuffer
		}
		blockEnd += svbLen
	}
	buf = buf[:blockEnd]

	// FOR base is read only after the length check covers its region.
	var forBase uint32
	if hasFOR && !u64Single {
		forBase = readFORBase(buf, forBaseOff, forWidth)
	}

	// Fill the out-struct (single flags write instead of six bool fields).
	var flags uint8
	if hasExceptions {
		flags |= hfExceptions
	}
	if hasDelta {
		flags |= hfDelta
	}
	if hasZigZag {
		flags |= hfZigZag
	}
	if hasFOR {
		flags |= hfFOR
	}
	if u64Single {
		flags |= hfU64Single
	}

	out.consumed = pOff + payloadBytes
	out.pOff = pOff
	out.forBase = forBase
	out.count = count
	out.bitWidth = bitWidth
	out.excCount = excCount
	out.flags = flags
	return buf, nil
}

// growUnpackDst grows dst to blockSize (the unpack kernels write the full
// UTL block). Reuses dst when the capacity suffices, matching the previous
// per-pipeline logic.
func growUnpackDst(dst []uint32) []uint32 {
	if cap(dst) < blockSize {
		dst = make([]uint32, blockSize)
	}
	return dst[:blockSize]
}

// payloadAt returns the bit-packed payload region within the blockEnd-limited
// buf returned by decodeUnpackHeader (the prologue proved it in-range).
func (h *unpackHeaderOut) payloadAt(buf []byte) []byte {
	return buf[h.pOff : h.pOff+utlPayloadBytes(h.bitWidth)]
}
