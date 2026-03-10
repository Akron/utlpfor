//go:build goexperiment.simd && amd64

package utlpfor

import "simd/archsimd"

// findMinMaxSIMD computes the minimum and maximum of a uint32 slice using
// the best available SIMD level.
func findMinMaxSIMD(values []uint32) (uint32, uint32) {
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		return findMinMaxAVX512(values)
	case simdLevelAVX2:
		return findMinMaxAVX2(values)
	case simdLevelSSE2:
		return findMinMaxSSE2(values)
	default:
		return findMinMaxScalar(values)
	}
}

// forSubtractSIMD subtracts baseValue from each element using SIMD:
// dst[i] = src[i] - baseValue.
func forSubtractSIMD(dst, src []uint32, baseValue uint32) {
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		forSubtractAVX512(dst, src, baseValue)
	case simdLevelAVX2:
		forSubtractAVX2(dst, src, baseValue)
	case simdLevelSSE2:
		forSubtractSSE2(dst, src, baseValue)
	default:
		forSubtractScalar(dst, src, baseValue)
	}
}

// forAddSIMD adds baseValue to each of the first count elements using SIMD:
// output[i] += baseValue.
func forAddSIMD(output []uint32, count int, baseValue uint32) {
	switch simdLevel {
	case simdLevelAVX512VBMI, simdLevelAVX512:
		forAddAVX512(output, count, baseValue)
	case simdLevelAVX2:
		forAddAVX2(output, count, baseValue)
	case simdLevelSSE2:
		forAddSSE2(output, count, baseValue)
	default:
		forAddScalar(output, count, baseValue)
	}
}

// --- SSE2 (Uint32x4) implementations ---

// findMinMaxSSE2 computes min/max using SSE2 4-wide operations.
func findMinMaxSSE2(values []uint32) (uint32, uint32) {
	if len(values) < 4 {
		return findMinMaxScalar(values)
	}
	minVec := archsimd.LoadUint32x4Slice(values[:4])
	maxVec := minVec
	i := 4
	for ; i+4 <= len(values); i += 4 {
		chunk := archsimd.LoadUint32x4Slice(values[i:])
		minVec = minVec.Min(chunk)
		maxVec = maxVec.Max(chunk)
	}
	var minLanes, maxLanes [4]uint32
	minVec.Store(&minLanes)
	maxVec.Store(&maxLanes)
	minResult, maxResult := minLanes[0], maxLanes[0]
	for _, v := range minLanes[1:] {
		if v < minResult {
			minResult = v
		}
	}
	for _, v := range maxLanes[1:] {
		if v > maxResult {
			maxResult = v
		}
	}
	for ; i < len(values); i++ {
		if values[i] < minResult {
			minResult = values[i]
		}
		if values[i] > maxResult {
			maxResult = values[i]
		}
	}
	return minResult, maxResult
}

// forSubtractSSE2 subtracts baseValue from each element using SSE2.
func forSubtractSSE2(dst, src []uint32, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x4(baseValue)
	i := 0
	for ; i+4 <= len(src); i += 4 {
		v := archsimd.LoadUint32x4Slice(src[i:])
		v = v.Sub(baseVec)
		v.StoreSlice(dst[i:])
	}
	for ; i < len(src); i++ {
		dst[i] = src[i] - baseValue
	}
}

// forAddSSE2 adds baseValue to each of the first count elements using SSE2.
func forAddSSE2(output []uint32, count int, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x4(baseValue)
	i := 0
	for ; i+4 <= count; i += 4 {
		v := archsimd.LoadUint32x4Slice(output[i:])
		v = v.Add(baseVec)
		v.StoreSlice(output[i:])
	}
	for ; i < count; i++ {
		output[i] += baseValue
	}
}

// findMinMaxAVX2 computes min/max using AVX2 8-wide operations.
func findMinMaxAVX2(values []uint32) (uint32, uint32) {
	if len(values) < 8 {
		return findMinMaxScalar(values)
	}
	minVec := archsimd.LoadUint32x8Slice(values[:8])
	maxVec := minVec
	i := 8
	for ; i+8 <= len(values); i += 8 {
		chunk := archsimd.LoadUint32x8Slice(values[i:])
		minVec = minVec.Min(chunk)
		maxVec = maxVec.Max(chunk)
	}
	var minLanes, maxLanes [8]uint32
	minVec.Store(&minLanes)
	maxVec.Store(&maxLanes)
	minResult, maxResult := minLanes[0], maxLanes[0]
	for _, v := range minLanes[1:] {
		if v < minResult {
			minResult = v
		}
	}
	for _, v := range maxLanes[1:] {
		if v > maxResult {
			maxResult = v
		}
	}
	for ; i < len(values); i++ {
		if values[i] < minResult {
			minResult = values[i]
		}
		if values[i] > maxResult {
			maxResult = values[i]
		}
	}
	return minResult, maxResult
}

// forSubtractAVX2 subtracts baseValue from each element using AVX2.
func forSubtractAVX2(dst, src []uint32, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x8(baseValue)
	i := 0
	for ; i+8 <= len(src); i += 8 {
		v := archsimd.LoadUint32x8Slice(src[i:])
		v = v.Sub(baseVec)
		v.StoreSlice(dst[i:])
	}
	for ; i < len(src); i++ {
		dst[i] = src[i] - baseValue
	}
}

// forAddAVX2 adds baseValue to each of the first count elements using AVX2.
func forAddAVX2(output []uint32, count int, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x8(baseValue)
	i := 0
	for ; i+8 <= count; i += 8 {
		v := archsimd.LoadUint32x8Slice(output[i:])
		v = v.Add(baseVec)
		v.StoreSlice(output[i:])
	}
	for ; i < count; i++ {
		output[i] += baseValue
	}
}

// findMinMaxAVX512 computes min/max using AVX-512 16-wide operations.
func findMinMaxAVX512(values []uint32) (uint32, uint32) {
	if len(values) < 16 {
		return findMinMaxAVX2(values)
	}
	minVec := archsimd.LoadUint32x16Slice(values[:16])
	maxVec := minVec
	i := 16
	for ; i+16 <= len(values); i += 16 {
		chunk := archsimd.LoadUint32x16Slice(values[i:])
		minVec = minVec.Min(chunk)
		maxVec = maxVec.Max(chunk)
	}
	var minLanes, maxLanes [16]uint32
	minVec.Store(&minLanes)
	maxVec.Store(&maxLanes)
	minResult, maxResult := minLanes[0], maxLanes[0]
	for _, v := range minLanes[1:] {
		if v < minResult {
			minResult = v
		}
	}
	for _, v := range maxLanes[1:] {
		if v > maxResult {
			maxResult = v
		}
	}
	for ; i < len(values); i++ {
		if values[i] < minResult {
			minResult = values[i]
		}
		if values[i] > maxResult {
			maxResult = values[i]
		}
	}
	return minResult, maxResult
}

// forSubtractAVX512 subtracts baseValue from each element using AVX-512.
func forSubtractAVX512(dst, src []uint32, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x16(baseValue)
	i := 0
	for ; i+16 <= len(src); i += 16 {
		v := archsimd.LoadUint32x16Slice(src[i:])
		v = v.Sub(baseVec)
		v.StoreSlice(dst[i:])
	}
	for ; i < len(src); i++ {
		dst[i] = src[i] - baseValue
	}
}

// forAddAVX512 adds baseValue to each of the first count elements using AVX-512.
func forAddAVX512(output []uint32, count int, baseValue uint32) {
	baseVec := archsimd.BroadcastUint32x16(baseValue)
	i := 0
	for ; i+16 <= count; i += 16 {
		v := archsimd.LoadUint32x16Slice(output[i:])
		v = v.Add(baseVec)
		v.StoreSlice(output[i:])
	}
	for ; i < count; i++ {
		output[i] += baseValue
	}
}
