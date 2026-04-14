package utlpfor

// Delta indicates that the values should be delta-encoded before packing.
const Delta byte = 1 << 0

// NoFOR disables Frame-of-Reference analysis during packing.
// When set, the encoder skips the min/max scan and FOR cost evaluation,
// going directly to bitpacking + patching. This improves packing speed
// when it is known in advance that FOR will not be beneficial (e.g.
// delta-encoded strictly monotonic data).
const NoFOR byte = 1 << 1

// NoPatch disables exception analysis and patching during packing.
// The encoder uses the minimum step bitwidth that fits all values,
// producing zero exceptions. This yields faster packing at the cost
// of potentially larger output when a few outlier values inflate the
// bitwidth. Ideal for dictionary-compressed data where all values
// share a known maximum range. Implies NoFOR. NoPatch also benefits
// random access by guaranteeing no exception data needs to be decoded.
const NoPatch byte = 1 << 2
