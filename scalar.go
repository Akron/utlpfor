package utlpfor

// packLanesUTLScalar packs values into UTL lane-interleaved format using
// 64-bit pair-lane processing. Two adjacent lanes are packed simultaneously
// with a single uint64 store per super-word access, halving memory operations.
// This is using the pattern described in the FastLanes paper.
func packLanesUTLScalar(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	for lanePair := 0; lanePair < utlLaneCount; lanePair += 2 {
		packLanePairUTL64(dst, values, lanePair, bitWidth)
	}
}

// packLanePairUTL64 packs values for two adjacent UTL lanes using a pair
// of uint64 accumulators and uint64 stores.
// This is using the pattern described in the FastLanes paper.
func packLanePairUTL64(dst []byte, values []uint32, lane0, bitWidth int) {
	var mask uint64
	if bitWidth >= 32 {
		mask = 0xFFFFFFFF
	} else {
		mask = uint64((1 << bitWidth) - 1)
	}

	var acc0, acc1 uint64
	var bitsInAcc int
	outByteIdx := lane0 * 4

	for v := range utlValuesPerLane {
		seqIdx0 := lane0 + v*utlLaneCount
		seqIdx1 := lane0 + 1 + v*utlLaneCount
		var val0, val1 uint32
		if seqIdx0 < len(values) {
			val0 = values[seqIdx0]
		}
		if seqIdx1 < len(values) {
			val1 = values[seqIdx1]
		}

		acc0 |= (uint64(val0) & mask) << bitsInAcc
		acc1 |= (uint64(val1) & mask) << bitsInAcc
		bitsInAcc += bitWidth

		for bitsInAcc >= 32 {
			pair := uint64(uint32(acc0)) | (uint64(uint32(acc1)) << 32)
			bo.PutUint64(dst[outByteIdx:], pair)
			outByteIdx += utlSuperWordBytes
			acc0 >>= 32
			acc1 >>= 32
			bitsInAcc -= 32
		}
	}
	if bitsInAcc > 0 {
		pair := uint64(uint32(acc0)) | (uint64(uint32(acc1)) << 32)
		bo.PutUint64(dst[outByteIdx:], pair)
	}
}

// unpackLanesUTLScalar unpacks UTL lane-interleaved payload into sequential
// values using 64-bit pair-lane processing. Two adjacent lanes are unpacked
// simultaneously with a single uint64 load per super-word access.
// This is using the pattern described in the FastLanes paper.
func unpackLanesUTLScalar(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	for lanePair := 0; lanePair < utlLaneCount; lanePair += 2 {
		unpackLanePairUTL64(dst, payload, lanePair, bitWidth, count)
	}
}

// unpackLanePairUTL64 unpacks values for two adjacent UTL lanes using a pair
// of uint64 accumulators and uint64 loads.
// This is using the pattern described in the FastLanes paper.
func unpackLanePairUTL64(dst []uint32, payload []byte, lane0, bitWidth, count int) {
	var mask32 uint32
	if bitWidth >= 32 {
		mask32 = 0xFFFFFFFF
	} else {
		mask32 = (1 << bitWidth) - 1
	}

	var acc0, acc1 uint64
	var bitsInAcc int
	inByteIdx := lane0 * 4

	for v := range utlValuesPerLane {
		for bitsInAcc < bitWidth {
			if inByteIdx+8 > len(payload) {
				bitsInAcc = bitWidth
				break
			}
			pair := bo.Uint64(payload[inByteIdx:])
			acc0 |= uint64(uint32(pair)) << bitsInAcc
			acc1 |= (pair >> 32) << bitsInAcc
			inByteIdx += utlSuperWordBytes
			bitsInAcc += 32
		}

		seqIdx0 := lane0 + v*utlLaneCount
		seqIdx1 := lane0 + 1 + v*utlLaneCount
		if seqIdx0 < count {
			dst[seqIdx0] = uint32(acc0) & mask32
		}
		if seqIdx1 < count {
			dst[seqIdx1] = uint32(acc1) & mask32
		}

		acc0 >>= bitWidth
		acc1 >>= bitWidth
		bitsInAcc -= bitWidth
	}
}

// packBlockScalar packs uint32 values into a UTL block with configurable
// header type flags. Used by the uint64 sub-block encoding path.
// typeFlags must include the integer type (e.g. headerTypeUint64Flag),
// and optionally headerCombineFlag.
// hasCombine controls whether block2Len space is reserved in the metadata.
// Always writes from dst[0]; the caller handles Append semantics.
// When NoInPlace is set, the input values slice is not modified;
// scratch is used as a work buffer instead.
func packBlockScalar(flag Flag, values []uint32, dst []byte, scratch []uint32, typeFlags uint32, hasCombine bool) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	noInPlace := flag&NoInPlace != 0
	if noInPlace && len(scratch) < ScratchLenNoInPlace {
		scratch = make([]uint32, ScratchLenNoInPlace)
	}

	// Build header flags from caller-supplied type and encoding flags.
	headerFlags := typeFlags
	headerFlags |= uint32(flag&Special) << 14

	// workValues is the slice used for all steps after FOR/delta.
	// Without NoInPlace, workValues == values (in-place as before).
	// With NoInPlace, workValues points to scratch[0:blockSize].
	workValues := values
	workRedirected := false

	// Step 1: FOR - subtract minimum value if beneficial.
	var useFOR bool
	var baseValue uint32
	var forW int
	if flag&NoFOR == 0 {
		useFOR, baseValue, forW = selectBitWidthWithFOR(values)
		if useFOR {
			if noInPlace {
				// Write FOR-subtracted result into scratch, leaving values untouched.
				workValues = scratch[:len(values)]
				forSubtractScalar(workValues, values, baseValue)
				workRedirected = true
			} else {
				forSubtractScalar(values, values, baseValue)
			}
			headerFlags |= uint32(forW) << forWidthShift
		}
	}

	// Step 2: Delta encode per lane (with optional zigzag).
	if flag&Delta != 0 {
		if noInPlace && !workRedirected {
			// FOR was not applied, so workValues still aliases values.
			// Copy values into scratch before delta encoding modifies them.
			workValues = scratch[:len(values)]
			copy(workValues, values)
		}
		needZZ := deltaEncodePerLaneScalar(workValues)
		if needZZ {
			headerFlags |= headerZigZagFlag
		}
		headerFlags |= headerDeltaFlag
	}

	// Step 3: Select step bitwidth and identify exceptions.
	var bitWidth, excCount int
	if flag&NoPatch != 0 {
		bitWidth = selectBitWidthNoPatch(workValues)
	} else {
		bitWidth, excCount = selectBitWidth(workValues)
	}
	payloadSize := utlPayloadBytes(bitWidth)
	hasExceptions := excCount > 0
	forBBytes := forBaseBytes(forW)

	// Step 4a: No exceptions - write header + FOR base + packed payload.
	if !hasExceptions {
		pOff := payloadOffset(forBBytes, false, hasCombine)
		totalLen := pOff + payloadSize
		dst = ensureLen(dst, totalLen)
		bo.PutUint32(dst, encodeHeader(len(values), bitWidth, 0, headerFlags))
		if useFOR {
			writeFORBase(dst, baseValue, forW, false)
		}
		packLanesUTLScalar(dst[pOff:pOff+payloadSize], workValues, bitWidth)
		return dst[:totalLen], nil
	}

	// Step 4b: With exceptions - write header + svbLen + FOR base + payload + exc + svb.
	excIdxSize := excIndexSize(excCount)
	maxSvbLen := maxSVBEncodedLen(excCount)
	pOff := payloadOffset(forBBytes, true, hasCombine)
	maxTotalLen := pOff + payloadSize + excIdxSize + maxSvbLen

	dst = ensureLen(dst, maxTotalLen)
	bo.PutUint32(dst, encodeHeader(len(values), bitWidth, excCount, headerFlags))
	if useFOR {
		writeFORBase(dst, baseValue, forW, true)
	}

	// Pack lower bits into UTL payload.
	packLanesUTLScalar(dst[pOff:pOff+payloadSize], workValues, bitWidth)

	// Collect exception high bits and write exception index + StreamVByte data.
	// When NoInPlace is active, highBits shifts to scratch[blockSize:2*blockSize].
	var highBits []uint32
	if noInPlace {
		highBits = scratch[blockSize : 2*blockSize]
	} else {
		highBits = scratch[:blockSize]
	}
	excOff := pOff + payloadSize
	collectAndWriteExceptions(workValues, bitWidth, dst[excOff:], excCount, highBits)
	svbOffset := excOff + excIdxSize
	svbLen := encodeSVBIntoDst(dst[svbOffset:maxTotalLen], highBits[:excCount])

	// Patch svbLen field at fixed offset in header area.
	bo.PutUint16(dst[headerBytes:], uint16(svbLen))
	return dst[:svbOffset+svbLen], nil
}

// packUint32Scalar is the scalar implementation of PackUint32.
// Delegates to packBlockScalar with uint32 type flags.
func packUint32Scalar(flag Flag, dst []byte, scratch []uint32, values []uint32) ([]byte, error) {
	if len(values) == 0 || len(values) > blockSize {
		return nil, ErrInvalidBuffer
	}

	// Append and NoInPlace share a bit pattern: shifting the Append bit by
	// one makes the destination offset branchless (`0` or `len(dst)`).
	off := len(dst) * int((flag&Append)>>4)
	block, err := packBlockScalar(flag&^Append, values, dst[off:off], scratch, headerTypeUint32Flag, false)
	if err != nil {
		return nil, err
	}
	totalLen := len(block)
	if cap(dst) >= off+totalLen {
		return dst[:off+totalLen], nil
	}
	dst = ensureAppend(dst, off, totalLen)
	copy(dst[off:], block)
	return dst[:off+totalLen], nil
}

// unpackUint32Scalar is the shared unpack implementation for both uint32 and
// uint64 blocks. The forUint64 flag selects the int-type validator and enables
// combine-flag-aware payload offset calculation.
func unpackUint32Scalar(dst []uint32, scratch []uint32, buf []byte, forUint64 bool) ([]uint32, int, error) {
	if len(buf) < headerBytes {
		return nil, 0, ErrInvalidBuffer
	}

	// Decode the 4-byte header to extract all block parameters.
	header := bo.Uint32(buf)
	count, bitWidth, intType, excCount, forWidth, hasExceptions, hasDelta, hasZigZag, _, hasCombine := decodeHeader(header)
	hasFOR := forWidth > 0

	// Validate int type based on caller context (uint32 vs uint64 API).
	if forUint64 {
		if err := validateIntType64(intType); err != nil {
			return nil, 0, err
		}
	} else {
		if err := validateIntType(intType); err != nil {
			return nil, 0, err
		}
		hasCombine = false // combine is never valid for uint32 blocks
	}

	if uint(count-1) >= blockSize || uint(bitWidth) > 32 {
		if count == 0 {
			return dst[:0], headerBytes, nil
		}
		if count > blockSize {
			return nil, 0, ErrInvalidBlockLength
		}
		return nil, 0, ErrInvalidBuffer
	}

	// FOR64 single-block: forWidth=3 means 8-byte base, handled by uint64 caller.
	u64Single := forUint64 && isFor64SingleBlock(intType, forWidth, hasCombine)

	// Compute payload start: header [+ svbLen] [+ forBase] [+ block2Len].
	pOff := headerBytes
	if hasExceptions {
		pOff += svbLenBytes
	}
	var forBase uint32
	if hasFOR {
		if u64Single {
			pOff += for64BaseSize
		} else {
			forBase = readFORBase(buf, pOff, forWidth)
			pOff += forBaseBytes(forWidth)
		}
	}
	if hasCombine {
		pOff += block2LenBytes
	}

	payloadBytes := utlPayloadBytes(bitWidth)
	if len(buf) < pOff+payloadBytes {
		return nil, 0, ErrInvalidBuffer
	}

	if cap(dst) < blockSize {
		dst = make([]uint32, blockSize)
	}
	dst = dst[:blockSize]

	// Step 1: Unpack bit-packed payload into lane-interleaved values.
	payload := buf[pOff : pOff+payloadBytes]
	unpackLanesUTLScalar(dst, payload, blockSize, bitWidth)
	dst = dst[:count]
	consumed := pOff + payloadBytes

	// Step 2: Apply exceptions (OR in high bits from StreamVByte data).
	if hasExceptions {
		excStart := pOff + payloadBytes
		var err error
		consumed, err = applyExceptions(dst, buf, excStart, count, bitWidth, excCount, scratch)
		if err != nil {
			return nil, 0, err
		}
	}

	// Step 3: Delta decode per lane (reverse of per-lane delta encoding).
	if hasDelta {
		if hasZigZag {
			deltaDecodePerLaneScalar(dst, true)
		} else {
			overflowPos := deltaDecodePerLaneWithOverflowScalar(dst, false)
			if overflowPos > 0 {
				return nil, 0, &ErrOverflow{Position: overflowPos}
			}
		}
	}

	// Step 4: Add FOR base value back (skipped for FOR64; caller handles uint64 add).
	if hasFOR && !u64Single {
		forAddScalar(dst, count, forBase)
	}

	return dst, consumed, nil
}

// MaxBlockLength32 returns the maximum byte length of a single packed
// uint32 block for the given encoding flags. This is useful for
// pre-allocating destination buffers to avoid allocations during PackUint32.
//
// The returned value is a conservative upper bound. Actual block
// lengths are typically much smaller.
//
// Flags that affect the worst-case length:
//   - NoPatch: disables exceptions, reducing the maximum length.
//   - NoFOR: disables frame-of-reference, removing the FOR base bytes.
//
// Flags that do NOT affect the worst-case length (ignored):
//   - Delta, Special, Append.
//
// Examples:
//
//	MaxBlockLength32(0)              // 1018 (general case, exceptions possible)
//	MaxBlockLength32(NoPatch)        // 520  (no exceptions)
//	MaxBlockLength32(NoPatch|NoFOR)  // 516  (no exceptions, no FOR base)
//	MaxBlockLength32(NoFOR)          // 1014 (exceptions possible, no FOR base)
//
// Usage for buffer pre-allocation:
//
//	dst := make([]byte, 0, utlpfor.MaxBlockLength32(0))
//	dst, _ = utlpfor.PackUint32(utlpfor.Append, block, dst, scratch)
func MaxBlockLength32(flag Flag) int {
	if flag&NoPatch != 0 {
		maxPayload := utlPayloadBytes(32)
		maxForBase := forBaseBytes(forWidthU32)
		if flag&NoFOR != 0 {
			maxForBase = 0
		}
		return headerBytes + maxForBase + maxPayload
	}
	// General case: exceptions possible.
	// Worst case is bitWidth=28 (max step where exceptions still occur),
	// all 128 values as exceptions, bitmap mode for exception index.
	maxPayload := utlPayloadBytes(28)
	maxForBase := forBaseBytes(forWidthU32)
	if flag&NoFOR != 0 {
		maxForBase = 0
	}
	maxExcIdx := excBitmapThreshold
	maxSvbData := maxSVBEncodedLen(blockSize)
	return headerBytes + svbLenBytes + maxForBase +
		maxPayload + maxExcIdx + maxSvbData
}
