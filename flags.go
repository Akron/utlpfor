package utlpfor

// Delta indicates that the values should be delta-encoded before packing.
const Delta byte = 1 << 0

// NoFOR disables Frame-of-Reference analysis during packing.
// When set, the encoder skips the min/max scan and FOR cost evaluation,
// going directly to bitpacking + patching. This improves packing speed
// when it is known in advance that FOR will not be beneficial (e.g.
// delta-encoded strictly monotonic data).
const NoFOR byte = 1 << 1
