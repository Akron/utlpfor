package utlpfor

// packLanesUTLScalarNaive packs values into UTL lane-interleaved format
// using single-lane processing (one uint32 load/store per super-word access).
// Kept for benchmark comparison against the 64-bit pair-lane implementation.
func packLanesUTLScalarNaive(dst []byte, values []uint32, bitWidth int) {
	if bitWidth == 0 {
		return
	}
	for lane := range utlLaneCount {
		packLaneUTL(dst, values, lane, bitWidth)
	}
}

// packLaneUTL packs values for a single UTL lane using a uint64 accumulator.
func packLaneUTL(dst []byte, values []uint32, lane, bitWidth int) {
	var mask uint64
	if bitWidth >= 32 {
		mask = 0xFFFFFFFF
	} else {
		mask = uint64((1 << bitWidth) - 1)
	}

	var acc uint64
	var bitsInAcc int
	outByteIdx := lane * 4

	for v := range utlValuesPerLane {
		seqIdx := lane + v*utlLaneCount
		var val uint32
		if seqIdx < len(values) {
			val = values[seqIdx]
		}
		acc |= (uint64(val) & mask) << bitsInAcc
		bitsInAcc += bitWidth
		for bitsInAcc >= 32 {
			bo.PutUint32(dst[outByteIdx:], uint32(acc))
			outByteIdx += utlSuperWordBytes
			acc >>= 32
			bitsInAcc -= 32
		}
	}
	if bitsInAcc > 0 {
		bo.PutUint32(dst[outByteIdx:], uint32(acc))
	}
}

// unpackLanesUTLScalarNaive unpacks UTL lane-interleaved payload into
// sequential values using single-lane processing.
// Kept for benchmark comparison against the 64-bit pair-lane implementation.
func unpackLanesUTLScalarNaive(dst []uint32, payload []byte, count, bitWidth int) {
	if bitWidth == 0 {
		clear(dst[:count])
		return
	}
	for lane := range utlLaneCount {
		unpackLaneUTL(dst, payload, lane, bitWidth, count)
	}
}

// unpackLaneUTL unpacks values for a single UTL lane using a uint64 accumulator.
func unpackLaneUTL(dst []uint32, payload []byte, lane, bitWidth, count int) {
	var mask uint32
	if bitWidth >= 32 {
		mask = 0xFFFFFFFF
	} else {
		mask = (1 << bitWidth) - 1
	}

	var acc uint64
	var bitsInAcc int
	inByteIdx := lane * 4

	for v := range utlValuesPerLane {
		for bitsInAcc < bitWidth {
			if inByteIdx+4 > len(payload) {
				bitsInAcc = bitWidth
				break
			}
			acc |= uint64(bo.Uint32(payload[inByteIdx:])) << bitsInAcc
			inByteIdx += utlSuperWordBytes
			bitsInAcc += 32
		}
		value := uint32(acc) & mask
		acc >>= bitWidth
		bitsInAcc -= bitWidth
		seqIdx := lane + v*utlLaneCount
		if seqIdx < count {
			dst[seqIdx] = value
		}
	}
}
