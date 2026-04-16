package utlpfor

import "math/bits"

// deltaFullUnpackThreshold returns the posInLane value above which
// GetUint32 uses full UnpackUint32 instead of the single-lane walk
// for delta-encoded blocks. The threshold depends on both SIMD level
// and bitwidth: for high bitwidths (>=20), lane extraction is expensive
// (crossing word boundaries), so full unpack wins at lower positions.
// For low/medium bitwidths, lane walk is very cheap and preferred.
func deltaFullUnpackThreshold(bitWidth int) int {
	if bitWidth >= 20 {
		switch simdLevel {
		case simdLevelAVX512VBMI, simdLevelAVX512:
			return 0
		case simdLevelAVX2:
			return 1
		case simdLevelSSE2:
			return 2
		default:
			return 3
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
// SIMD acceleration is applied through the dispatched UnpackUint32 on the
// full-unpack path; single-lane extraction is always scalar.
func GetUint32(pos int, buf []byte, scratch []uint32) (uint32, error) {
	return getUint32Scalar(pos, buf, scratch)
}

// getUint32Scalar implements GetUint32. Single-lane extraction is scalar;
// the full-unpack path dispatches through UnpackUint32 for SIMD acceleration.
func getUint32Scalar(pos int, buf []byte, scratch []uint32) (uint32, error) {
	if len(buf) < headerBytes {
		return 0, ErrInvalidBuffer
	}

	header := bo.Uint32(buf)
	count, bitWidth, intType, excCount, forWidth, hasExceptions, hasDelta, hasZigZag := decodeHeader(header)
	hasFOR := forWidth > 0

	if err := validateIntType(intType); err != nil {
		return 0, err
	}

	if pos < 0 || pos >= count {
		return 0, ErrPositionOutOfRange
	}

	if bitWidth > 32 {
		return 0, ErrInvalidBuffer
	}

	pOff := payloadOffset(0, hasExceptions)
	var forBase uint32
	if hasFOR {
		forBase = readFORBase(buf, pOff, forWidth)
		pOff += forBaseBytes(forWidth)
	}

	payloadBytes := utlPayloadBytes(bitWidth)
	if len(buf) < pOff+payloadBytes {
		return 0, ErrInvalidBuffer
	}

	payload := buf[pOff : pOff+payloadBytes]

	if !hasDelta {
		value, err := getValueDirect(pos, buf, payload, pOff+payloadBytes, bitWidth, count, excCount, hasExceptions)
		if err != nil {
			return 0, err
		}
		return value + forBase, nil
	}

	// Delta path: choose between full unpack and single-lane.
	posInLane := pos / utlLaneCount
	if posInLane > deltaFullUnpackThreshold(bitWidth) {
		return getUint32FullUnpack(pos, buf, scratch)
	}

	value, err := getValueWithDelta(pos, buf, payload,
		pOff+payloadBytes, bitWidth, count, excCount,
		hasExceptions, hasZigZag)
	if err != nil {
		return 0, err
	}
	return value + forBase, nil
}

// getValueDirect extracts a single value without delta decoding.
// Uses embedded single-value SVB decode instead of bulk decode.
func getValueDirect(pos int, buf []byte, payload []byte, excStart, bitWidth, count, excCount int, hasExceptions bool) (uint32, error) {
	var value uint32
	if bitWidth > 0 {
		value = extractPackedValueUTL(pos, payload, bitWidth)
	}

	if hasExceptions {
		excIdx := findExceptionIndex(buf, excStart, excCount, pos)
		if excIdx >= 0 {
			svbData := svbDataSlice(buf, excStart, excCount)
			highBits := svbDecodeOneInternal(svbData, excCount, excIdx)
			value |= highBits << bitWidth
		}
	}

	return value, nil
}

// getValueWithDelta extracts a value with per-lane delta decoding.
// Reconstructs the lane (up to posInLane) using per-position SVB decode.
func getValueWithDelta(pos int, buf []byte, payload []byte, excStart, bitWidth, count, excCount int, hasExceptions, hasZigZag bool) (uint32, error) {
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
		svbData := svbDataSlice(buf, excStart, excCount)
		applyLaneExceptions(laneValues[:], buf, excStart, excCount, lane, posInLane, count, bitWidth, svbData)
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
func applyLaneExceptions(laneValues []uint32, buf []byte,
	excStart, excCount, lane, posInLane, count, bitWidth int,
	svbData []byte) {
	shift := uint(bitWidth)
	if excCount <= excBitmapThreshold {
		for excIdx := range excCount {
			p := int(buf[excStart+excIdx])
			if p%utlLaneCount != lane {
				continue
			}
			v := p / utlLaneCount
			if v > posInLane {
				break
			}
			highBits := svbDecodeOneInternal(svbData, excCount, excIdx)
			laneValues[v] |= highBits << shift
		}
	} else {
		bitmap := buf[excStart : excStart+16]
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
func getUint32FullUnpack(pos int, buf []byte, scratch []uint32) (uint32, error) {
	if len(scratch) < blockSize {
		scratch = make([]uint32, blockSize)
	}
	var dst [blockSize]uint32
	unpacked, _, err := UnpackUint32(dst[:0], scratch, buf)
	if err != nil {
		return 0, err
	}
	return unpacked[pos], nil
}
