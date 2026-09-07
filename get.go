package utlpfor

import "math/bits"

// deltaFullUnpackThreshold returns the posInLane value above which
// GetUint32 uses full UnpackUint32 instead of the single-lane walk
// for delta-encoded blocks. The threshold depends on both SIMD level
// and bitwidth: for high bitwidths (>=20), lane extraction is expensive
// (crossing word boundaries), so full unpack wins at lower positions.
// For low/medium bitwidths, lane walk is very cheap and preferred.
//
// Thresholds are derived from BenchmarkGetUint32_Approaches measured
// across scalar/SSE2/AVX2 (AVX-512 uses AVX2 values as conservative
// proxy until benchmarked on AVX-512 hardware).
func deltaFullUnpackThreshold(bitWidth int) int {
	if bitWidth >= 20 {
		switch simdLevel {
		case simdLevelAVX512VBMI, simdLevelAVX512:
			return 4
		case simdLevelAVX2:
			return 4
		case simdLevelSSE2:
			return 2
		default:
			return 2
		}
	}
	return utlValuesPerLane
}

// extractPackedValueUTL extracts a single packed value from a UTL payload.
func extractPackedValueUTL(pos int, payload []byte, bitWidth int) uint32 {
	lane := pos % utlLaneCount
	posInLane := pos / utlLaneCount
	bitPos := posInLane * bitWidth
	wordInLane := bitPos / 32
	bitOffset := bitPos % 32

	byteOffset := wordInLane*utlSuperWordBytes + lane*4

	var acc uint64
	if byteOffset+4 <= len(payload) {
		acc = uint64(bo.Uint32(payload[byteOffset:]))
	}
	if bitWidth > 32-bitOffset {
		nextByteOffset := byteOffset + utlSuperWordBytes
		if nextByteOffset+4 <= len(payload) {
			acc |= uint64(bo.Uint32(payload[nextByteOffset:])) << 32
		}
	}

	acc >>= uint(bitOffset)
	var mask uint64
	if bitWidth >= 32 {
		mask = 0xFFFFFFFF
	} else {
		mask = uint64((1 << bitWidth) - 1)
	}
	return uint32(acc & mask)
}

// GetUint32 extracts a single value at the given position from the packed block.
// scratch with capacity >= ScratchLen enables zero-allocation operation for
// delta-encoded blocks; pass nil if zero-alloc is not required.
// Single code path; SIMD acceleration is applied internally via dispatched
// UnpackUint32 (full-unpack fallback for delta blocks with high posInLane).
func GetUint32(pos int, src []byte, scratch []uint32) (uint32, error) {
	return getUint32Scalar(pos, src, scratch, false)
}

// getUint32Scalar is the shared get-single-value implementation for both
// uint32 and uint64 sub-blocks. The forUint64 flag selects the int-type
// validator and enables combine-flag-aware payload offset calculation.
func getUint32Scalar(pos int, src []byte, scratch []uint32, forUint64 bool) (uint32, error) {
	if len(src) < headerBytes {
		return 0, ErrInvalidBuffer
	}

	// Decode header and validate int type for the caller's API context.
	header := bo.Uint32(src)
	count, bitWidth, intType, excCount, forWidth, hasExceptions, hasDelta, hasZigZag, _, hasCombine := decodeHeader(header)
	hasFOR := forWidth > 0

	if forUint64 {
		if err := validateIntType64(intType); err != nil {
			return 0, err
		}
	} else {
		if err := validateIntType(intType); err != nil {
			return 0, err
		}
		hasCombine = false // combine is never valid for uint32 blocks
	}

	if pos < 0 || pos >= count {
		return 0, ErrPositionOutOfRange
	}
	if bitWidth > 32 {
		return 0, ErrInvalidBuffer
	}

	// FOR64 single-block: forWidth=3 means 8-byte base, handled by uint64 caller.
	u64Single := forUint64 && isFor64SingleBlock(intType, forWidth, hasCombine)

	// Phase 1: offset computation: accumulate the payload start offset
	// from the variable-length metadata fields. hasExceptions is checked
	// here to account for the 2-byte svbLen field that precedes the FOR base.
	// Phase 2 (below, after blockEnd is known): re-checks hasExceptions to
	// validate the exception region against the buffer. Merging the two
	// phases would tangle offset arithmetic with validation and hurt
	// readability; checking a register boolean twice is free.
	pOff := headerBytes
	if hasExceptions {
		pOff += svbLenBytes
	}
	forB := 0
	if hasFOR {
		if u64Single {
			pOff += for64BaseSize
		} else {
			forB = forBaseBytes(forWidth)
			pOff += forB
		}
	}
	if hasCombine {
		pOff += block2LenBytes
	}

	payloadBytes := utlPayloadBytes(bitWidth)
	blockEnd := pOff + payloadBytes
	if len(src) < blockEnd {
		return 0, ErrInvalidBuffer
	}
	if hasExceptions {
		// Phase 2: exception region validation (see Phase 1 comment above).
		// More exceptions than values cannot describe a valid block.
		if excCount > count {
			return 0, ErrInvalidBuffer
		}
		// svbLen lives at the fixed offset; present because blockEnd >= 6.
		svbLen := int(bo.Uint16(src[headerBytes:]))
		idxSize := excIndexSize(excCount)
		if svbLen < svbControlByteCount(excCount) || blockEnd+idxSize+svbLen > len(src) {
			return 0, ErrInvalidBuffer
		}
		blockEnd += idxSize + svbLen
	}
	src = src[:blockEnd]

	// FOR base is read only after the length check covers its region.
	var forBase uint32
	if forB != 0 {
		forBase = readFORBase(src, pOff-forB, forWidth)
	}

	payload := src[pOff : pOff+payloadBytes]

	// Non-delta path: extract single value directly from packed payload.
	if !hasDelta {
		value, err := getValueDirect(pos, src, payload, pOff+payloadBytes, bitWidth, count, excCount, hasExceptions)
		if err != nil {
			return 0, err
		}
		return value + forBase, nil
	}

	// Delta path: use full unpack for deep lane positions, lane walk otherwise.
	posInLane := pos / utlLaneCount
	if posInLane > deltaFullUnpackThreshold(bitWidth) {
		if forUint64 {
			return getFullUnpackViaBlock(pos, src, scratch)
		}
		return getUint32FullUnpack(pos, src, scratch)
	}

	value, err := getValueWithDelta(pos, src, payload,
		pOff+payloadBytes, bitWidth, count, excCount,
		hasExceptions, hasZigZag)
	if err != nil {
		return 0, err
	}
	return value + forBase, nil
}

// getFullUnpackViaBlock performs a full block unpack to extract a single
// uint32 value from a uint64-typed block. Uses scratch[0:blockSize] as the
// output buffer and scratch[2*blockSize:3*blockSize] as exception workspace,
// matching the unpack scratch layout.
// Dispatches to the SIMD-appropriate unpack kernel for the delta decode
// and exception application steps.
func getFullUnpackViaBlock(pos int, src []byte, scratch []uint32) (uint32, error) {
	if len(scratch) < ScratchLen64 {
		scratch = make([]uint32, ScratchLen64)
	}
	var unpacked []uint32
	var err error
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		unpacked, _, err = unpackBlockForUint64AVX512(scratch[:0], scratch[2*blockSize:], src)
	case simdLevelAVX2:
		unpacked, _, err = unpackBlockForUint64AVX2(scratch[:0], scratch[2*blockSize:], src)
	case simdLevelSSE2:
		unpacked, _, err = unpackBlockForUint64SSE2(scratch[:0], scratch[2*blockSize:], src)
	default:
		unpacked, _, err = unpackUint32Scalar(scratch[:0], scratch[2*blockSize:], src, true)
	}
	if err != nil {
		return 0, err
	}
	return unpacked[pos], nil
}

// getValueDirect extracts a single value without delta decoding.
// Uses embedded single-value SVB decode instead of bulk decode.
// excStart is the offset of the exception index in buf; the region is
// validated by the caller. The single-value decode clamps all reads into
// the buffer, so corrupt control bytes degrade to wrong values instead
// of panicking. Scalar exception index search and SVB decode are used
// directly; SIMD alternatives have higher setup overhead that exceeds
// their benefit for the typical small exception counts on this path.
func getValueDirect(pos int, buf, payload []byte, excStart, bitWidth, count, excCount int, hasExceptions bool) (uint32, error) {
	var value uint32
	if bitWidth > 0 {
		value = extractPackedValueUTL(pos, payload, bitWidth)
	}

	if hasExceptions {
		excRegion := buf[excStart:]
		excIdxPos := findExceptionIndex(excRegion, excCount, pos)
		if excIdxPos >= 0 {
			highBits := svbDecodeOneInternal(excRegion[excIndexSize(excCount):], excCount, excIdxPos)
			value |= highBits << bitWidth
		}
	}

	return value, nil
}

// getValueWithDelta extracts a value with per-lane delta decoding.
// Reconstructs the lane (up to posInLane) using per-position SVB decode.
// excStart is the offset of the exception index in buf; the region is
// validated by the caller.
func getValueWithDelta(pos int, buf, payload []byte, excStart, bitWidth, count, excCount int, hasExceptions, hasZigZag bool) (uint32, error) {
	lane := pos % utlLaneCount
	posInLane := pos / utlLaneCount

	var laneValues [utlValuesPerLane]uint32
	for v := 0; v <= posInLane; v++ {
		seqIdx := lane + v*utlLaneCount
		if seqIdx >= count {
			break
		}
		if bitWidth > 0 {
			laneValues[v] = extractPackedValueUTL(seqIdx, payload, bitWidth)
		}
	}

	if hasExceptions {
		applyLaneExceptions(laneValues[:], buf[excStart:], excCount, lane, posInLane, count, bitWidth)
	}

	if hasZigZag {
		laneValues[0] = uint32(zigzagDecode32(laneValues[0]))
	}
	for v := 1; v <= posInLane; v++ {
		seqIdx := lane + v*utlLaneCount
		if seqIdx >= count {
			break
		}
		if hasZigZag {
			laneValues[v] = laneValues[v-1] + uint32(zigzagDecode32(laneValues[v]))
		} else {
			laneValues[v] = laneValues[v-1] + laneValues[v]
		}
	}

	return laneValues[posInLane], nil
}

// applyLaneExceptions resolves exception high bits for lane positions
// [0..posInLane] in a single pass over the exception index, calling
// svbDecodeOneInternal only for positions that belong to the target lane.
// excRegion is the pre-sliced exception region in buf (positions or
// bitmap followed by the SVB data); the caller validates it covers the
// whole exception area, while the single-value SVB decode clamps reads
// into the region so corrupt input degrades instead of panicking. The
// single-pass design avoids repeated exception index scans. Scalar SVB
// decode is used directly; SIMD SVB helpers were benchmarked and showed
// higher overhead than scalar for all typical exception counts.
func applyLaneExceptions(laneValues []uint32, excRegion []byte,
	excCount, lane, posInLane, count, bitWidth int) {
	shift := uint(bitWidth)
	// The SVB stream follows the exception index (positions or bitmap).
	svbData := excRegion[excIndexSize(excCount):]
	if excCount <= excBitmapThreshold {
		for i := range excCount {
			p := int(excRegion[i])
			if p%utlLaneCount != lane {
				continue
			}
			v := p / utlLaneCount
			if v > posInLane {
				break
			}
			highBits := svbDecodeOneInternal(svbData, excCount, i)
			laneValues[v] |= highBits << shift
		}
	} else {
		// The bitmap occupies the first 16 bytes of the region.
		bitmap := excRegion[:16]
		for v := 0; v <= posInLane; v++ {
			seqIdx := lane + v*utlLaneCount
			if seqIdx >= count {
				break
			}
			byteIdx := seqIdx / 8
			bitIdx := uint(seqIdx % 8)
			if bitmap[byteIdx]&(1<<bitIdx) == 0 {
				continue
			}
			rank := 0
			for b := range byteIdx {
				rank += bits.OnesCount8(bitmap[b])
			}
			rank += bits.OnesCount8(bitmap[byteIdx] & ((1 << bitIdx) - 1))
			highBits := svbDecodeOneInternal(svbData, excCount, rank)
			laneValues[v] |= highBits << shift
		}
	}
}

// getUint32FullUnpack performs a full block unpack to extract a single value.
// Outlined from getUint32Scalar to keep the common path lean (avoids the
// 512-byte stack allocation when single-lane extraction suffices).
func getUint32FullUnpack(pos int, src []byte, scratch []uint32) (uint32, error) {
	if len(scratch) < blockSize {
		scratch = make([]uint32, blockSize)
	}
	var dst [blockSize]uint32
	unpacked, _, err := UnpackUint32(src, dst[:0], scratch)
	if err != nil {
		return 0, err
	}
	return unpacked[pos], nil
}
