package utlpfor_test

import (
	"fmt"
	"slices"

	utlpfor "github.com/Akron/utlpfor"
)

func ExamplePackUint32() {
	values := []uint32{10, 20, 30, 40, 50} // 5 x 4 = 20 bytes unpacked

	packed, err := utlpfor.PackUint32(0, values, nil, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("packed %d values into %d bytes\n", len(values), len(packed))
	// Output:
	// packed 5 values into 18 bytes
}

func ExamplePackUint32_delta() {
	values := make([]uint32, 128) // 128 x 4 = 512 bytes unpacked
	for i := range values {
		values[i] = uint32(100000 + i*4)
	}

	plain, _ := utlpfor.PackUint32(0, slices.Clone(values), nil, nil)
	delta, _ := utlpfor.PackUint32(utlpfor.Delta, slices.Clone(values), nil, nil)

	fmt.Printf("plain: %d bytes, delta: %d bytes\n", len(plain), len(delta))
	// Output:
	// plain: 200 bytes, delta: 136 bytes
}

func ExamplePackUint32_flags() {
	values := make([]uint32, 128) // 128 x 4 = 512 bytes unpacked
	for i := range values {
		values[i] = uint32(4_000_000 + i%16)
	}
	values[10] = 4_500_000
	values[90] = 4_600_000

	plain, _ := utlpfor.PackUint32(0, slices.Clone(values), nil, nil)
	noFOR, _ := utlpfor.PackUint32(utlpfor.NoFOR, slices.Clone(values), nil, nil)
	noPatch, _ := utlpfor.PackUint32(utlpfor.NoPatch, slices.Clone(values), nil, nil)
	both, _ := utlpfor.PackUint32(utlpfor.NoFOR|utlpfor.NoPatch, slices.Clone(values), nil, nil)

	fmt.Printf("default: %d bytes\n", len(plain))
	fmt.Printf("NoFOR: %d bytes\n", len(noFOR))
	fmt.Printf("NoPatch: %d bytes\n", len(noPatch))
	fmt.Printf("NoFOR|NoPatch: %d bytes\n", len(both))
	// Output:
	// default: 81 bytes
	// NoFOR: 388 bytes
	// NoPatch: 328 bytes
	// NoFOR|NoPatch: 388 bytes
}

func ExamplePackUint32_append() {
	block1 := []uint32{1, 2, 3, 4}         // 4 x 4 = 16 bytes unpacked
	block2 := []uint32{100, 200, 300, 400} // 4 x 4 = 16 bytes unpacked

	var dst []byte
	var err error
	dst, err = utlpfor.PackUint32(utlpfor.Append, block1, dst, nil)
	if err != nil {
		panic(err)
	}
	firstBlockLen := len(dst)
	dst, err = utlpfor.PackUint32(utlpfor.Append, block2, dst, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("two blocks concatenated: %d + %d = %d bytes\n",
		firstBlockLen, len(dst)-firstBlockLen, len(dst))
	// Output:
	// two blocks concatenated: 15 + 17 = 32 bytes
}

func ExamplePackUint32_noInPlace() {
	// A shared fixed-size buffer packed multiple times without cloning.
	var buf [128]uint32
	for i := range buf {
		buf[i] = uint32(500 + i*3)
	}

	scratch := make([]uint32, utlpfor.ScratchLenNoInPlace)
	var dst []byte
	var err error

	// Append implies NoInPlace: the buf array is never modified.
	for range 3 {
		dst, err = utlpfor.PackUint32(utlpfor.Append, buf[:], dst, scratch)
		if err != nil {
			panic(err)
		}
	}
	fmt.Printf("packed 3 blocks (%d bytes), buf[0]=%d (unchanged)\n",
		len(dst), buf[0])
	// Output:
	// packed 3 blocks (588 bytes), buf[0]=500 (unchanged)
}

func ExamplePackUint32_zeroAlloc() {
	values := make([]uint32, 128) // 128 x 4 = 512 bytes unpacked
	for i := range values {
		values[i] = uint32(i)
	}

	// Pre-allocate dst with MaxBlockLength32 to avoid allocation during packing.
	dst := make([]byte, 0, utlpfor.MaxBlockLength32(0))

	// ScratchLenNoInPlace (256) is the minimum scratch buffer capacity
	// (in uint32 elements) for zero-allocation packing with Append or
	// NoInPlace. Use ScratchLen (128) when neither flag is needed.
	// For uint64 operations, use ScratchLen64 (384) instead.
	scratch := make([]uint32, utlpfor.ScratchLenNoInPlace)

	packed, err := utlpfor.PackUint32(utlpfor.Append, values, dst, scratch)
	if err != nil {
		panic(err)
	}
	fmt.Printf("packed %d values with zero allocations\n", len(values))
	_ = packed
	// Output:
	// packed 128 values with zero allocations
}

func ExampleUnpackUint32() {
	original := []uint32{10, 20, 30, 40, 50} // 5 x 4 = 20 bytes unpacked
	packed, _ := utlpfor.PackUint32(0, original, nil, nil)

	unpacked, consumed, err := utlpfor.UnpackUint32(packed, nil, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("unpacked %d values, consumed %d bytes\n", len(unpacked), consumed)
	fmt.Printf("values: %v\n", unpacked)
	// Output:
	// unpacked 5 values, consumed 18 bytes
	// values: [10 20 30 40 50]
}

func ExampleGetUint32() {
	values := []uint32{100, 200, 300, 400, 500}
	packed, _ := utlpfor.PackUint32(0, values, nil, nil)

	val, err := utlpfor.GetUint32(2, packed, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("value at position 2: %d\n", val)
	// Output:
	// value at position 2: 300
}

func ExampleBlockLength() {
	values := []uint32{10, 20, 30, 40, 50} // 5 x 4 = 20 bytes unpacked
	packed, _ := utlpfor.PackUint32(0, values, nil, nil)

	blockLen, err := utlpfor.BlockLength(packed)
	if err != nil {
		panic(err)
	}
	fmt.Printf("block length: %d bytes\n", blockLen)
	// Output:
	// block length: 18 bytes
}

func ExampleBlockLength_skipBlocks() {
	block1 := []uint32{1, 2, 3}
	block2 := []uint32{100, 200, 300}

	var buf []byte
	buf, _ = utlpfor.PackUint32(utlpfor.Append, block1, buf, nil)
	buf, _ = utlpfor.PackUint32(utlpfor.Append, block2, buf, nil)

	bl, _ := utlpfor.BlockLength(buf)
	fmt.Printf("first block: %d bytes\n", bl)

	unpacked, _, _ := utlpfor.UnpackUint32(buf[bl:], nil, nil)
	fmt.Printf("second block values: %v\n", unpacked)
	// Output:
	// first block: 13 bytes
	// second block values: [100 200 300]
}

func ExampleHeader() {
	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(i * 10)
	}
	packed, _ := utlpfor.PackUint32(utlpfor.Delta|utlpfor.NoPatch, slices.Clone(values), nil, nil)

	count, bitWidth, excCount, hasDelta, hasFOR, _, _, err := utlpfor.Header(packed)
	if err != nil {
		panic(err)
	}
	fmt.Printf("count=%d, bitWidth=%d, excCount=%d, delta=%v, FOR=%v\n",
		count, bitWidth, excCount, hasDelta, hasFOR)
	// Output:
	// count=128, bitWidth=8, excCount=0, delta=true, FOR=false
}

func ExamplePackUint64() {
	values := []uint64{1_000_000_000_000, 1_000_000_000_001, 1_000_000_000_002} // 3 x 8 = 24 bytes unpacked

	packed, err := utlpfor.PackUint64(0, values, nil, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("packed %d uint64 values into %d bytes\n", len(values), len(packed))
	// Output:
	// packed 3 uint64 values into 19 bytes
}

func ExampleUnpackUint64() {
	original := []uint64{1_000_000_000_000, 1_000_000_000_001, 1_000_000_000_002} // 3 x 8 = 24 bytes unpacked
	packed, _ := utlpfor.PackUint64(0, original, nil, nil)

	unpacked, consumed, err := utlpfor.UnpackUint64(packed, nil, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("unpacked %d values, consumed %d bytes\n", len(unpacked), consumed)
	fmt.Printf("values: %v\n", unpacked)
	// Output:
	// unpacked 3 values, consumed 19 bytes
	// values: [1000000000000 1000000000001 1000000000002]
}

func ExampleGetUint64() {
	values := []uint64{1_000_000_000_000, 1_000_000_000_001, 1_000_000_000_002} // 3 x 8 = 24 bytes unpacked
	packed, _ := utlpfor.PackUint64(0, values, nil, nil)

	val, err := utlpfor.GetUint64(1, packed, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("value at position 1: %d\n", val)
	// Output:
	// value at position 1: 1000000000001
}

func ExampleMaxBlockLength32() {
	plain := utlpfor.MaxBlockLength32(0)
	noPatch := utlpfor.MaxBlockLength32(utlpfor.NoPatch)

	fmt.Printf("max block (default): %d bytes\n", plain)
	fmt.Printf("max block (NoPatch): %d bytes\n", noPatch)
	// Output:
	// max block (default): 1018 bytes
	// max block (NoPatch): 520 bytes
}

func ExampleMaxBlockLength64() {
	plain := utlpfor.MaxBlockLength64(0)
	noPatch := utlpfor.MaxBlockLength64(utlpfor.NoPatch)

	fmt.Printf("max block64 (default): %d bytes\n", plain)
	fmt.Printf("max block64 (NoPatch): %d bytes\n", noPatch)
	// Output:
	// max block64 (default): 2038 bytes
	// max block64 (NoPatch): 1042 bytes
}

func ExampleNewUint32() {
	c := utlpfor.NewUint32()

	values := []uint32{10, 20, 30, 40, 50}

	packed, _ := c.Compress(0, nil, values)

	unpacked, _, _ := c.Decompress(nil, packed)

	fmt.Printf("round-trip: %v\n", unpacked)
	// Output:
	// round-trip: [10 20 30 40 50]
}

func ExampleNewUint32_delta() {
	c := utlpfor.NewUint32()

	values := make([]uint32, 128)
	for i := range values {
		values[i] = uint32(1000 + i*4)
	}

	packed, _ := c.Compress(utlpfor.Delta, nil, slices.Clone(values))

	fmt.Printf("delta-compressed 128 values into %d bytes\n", len(packed))

	unpacked, _, _ := c.Decompress(nil, packed)

	fmt.Printf("first 5 values: %v\n", unpacked[:5])
	// Output:
	// delta-compressed 128 values into 170 bytes
	// first 5 values: [1000 1004 1008 1012 1016]
}

func ExampleNewUint32_get() {
	c := utlpfor.NewUint32()

	values := []uint32{100, 200, 300, 400, 500}
	packed, _ := c.Compress(0, nil, values)

	val, _ := c.Get(2, packed)

	fmt.Printf("value at position 2: %d\n", val)
	// Output:
	// value at position 2: 300
}

func ExampleNewUint64() {
	c := utlpfor.NewUint64()

	values := []uint64{1_000_000_000_000, 1_000_000_000_001, 1_000_000_000_002}

	packed, _ := c.Compress(0, nil, values)

	unpacked, _, _ := c.Decompress(nil, packed)

	fmt.Printf("round-trip: %v\n", unpacked)
	// Output:
	// round-trip: [1000000000000 1000000000001 1000000000002]
}

func ExampleNewUint64_get() {
	c := utlpfor.NewUint64()

	values := []uint64{1_000_000_000_000, 2_000_000_000_000, 3_000_000_000_000}
	packed, _ := c.Compress(0, nil, values)

	val, _ := c.Get(1, packed)

	fmt.Printf("value at position 1: %d\n", val)
	// Output:
	// value at position 1: 2000000000000
}
