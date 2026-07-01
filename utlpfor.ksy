meta:
  id: utlpfor_block
  title: UTL-PFOR Compressed Block
  endian: le
  file-extension: bin

doc: |
  A single UTL-PFOR compressed block containing up to 128 unsigned integers
  (uint32 or uint64). Uses Unified Transposed Layout (UTL/FastLanes) with
  16 lanes and 8 values per lane. Payload is organized in 64-byte super-words.

  Uint64 values are encoded via a double-block strategy: the 64-bit values
  are split into lower and upper 32-bit halves, each encoded as a standard
  uint32 sub-block. A FOR64 gateway optimization encodes the entire block as
  a single block when the value range fits in 32 bits after subtracting a
  64-bit minimum.

  Header layout (32-bit little-endian):
    bits  0- 7: count (number of values, 0-128)
    bits  8-12: bw_step_index (0-8 for 128-block, step bitwidth / 4)
    bits 13-14: int_type (0=uint8, 1=uint16, 2=uint32, 3=uint64)
    bits 15-16: for_width (00=no FOR, 01=uint8 1B, 10=uint16 2B,
                11=uint32 4B or uint64 8B; see below)
    bit  17:    SPECIAL flag (silently ignored by current decoder)
    bit  18:    reserved (must be 0)
    bit  19:    combine-with-next (uint64 double-block: Block 2 follows)
    bit  20:    E1 block-length mode (reserved, must be 0)
    bit  21:    E1 block-length all-exception flag (reserved, silently ignored)
    bit  22:    delta flag
    bit  23:    zigzag flag
    bits 24-31: exc_count (0 = no exceptions, 1-128 = exception count)

  for_width interpretation depends on context:
    - int_type=uint64, combine-with-next=0, for_width=11:
      FOR64 mode — 8-byte uint64 base (the block was range-reduced to uint32).
    - int_type=uint64, combine-with-next=1, for_width=11 (in either sub-block):
      Standard 4-byte uint32 base (each sub-block uses uint32 FOR semantics).
    - int_type=uint32, for_width=11:
      Standard 4-byte uint32 base.

  Step bitwidths: The encoded bw_step_index maps to the actual bitwidth
  as bit_width = bw_step_index * 4. Valid step bitwidths are:
  0, 4, 8, 12, 16, 20, 24, 28, 32 (step indices 0-8).

  Exception index format depends on exc_count:
    exc_count <= 16: sorted byte positions (exc_count bytes)
    exc_count >  16: bitmap (16 bytes, 128 bits, LSB-first per byte)

  Exception high bits are encoded using StreamVByte.

  Uint32 wire layouts:
    (no exc, no FOR):    [header:4][payload:N]
    (no exc, FOR):       [header:4][for_base:1|2|4][payload:N]
    (exc, no FOR):       [header:4][svb_length:2][payload:N][exc_index][svb_data]
    (exc, FOR):          [header:4][svb_length:2][for_base:1|2|4][payload:N][exc_index][svb_data]

  Uint64 wire layouts (int_type=3):
    Single-block, all values < 2^32 (combine=0, for_width!=11):
      Same as uint32 layout above.
    FOR64 single-block (combine=0, for_width=11):
      [header:4][svb_length:2?][for64_base:8][payload:N][exc_index?][svb_data?]
    Two-block (combine=1):
      Block 1: [header:4][svb_length:2?][for_base:0|1|2|4][block2_len:2][payload:N][exc_index?][svb_data?]
      Block 2: [header:4][svb_length:2?][for_base:0|1|2|4][payload:N][exc_index?][svb_data?]

  block2_len (uint16 LE) stores the total byte length of Block 2 (header
  through last data byte). Present only when combine-with-next=1.

seq:
  - id: header
    type: block_header
  - id: svb_length
    type: u2
    if: header.has_exceptions
    doc: StreamVByte encoded data length in bytes (uint16 LE). Always at offset 4.
  - id: for_base
    size: header.for_base_bytes
    if: header.has_for
    doc: |
      FOR base value. Width depends on context:
      - uint32 blocks: 1 byte (uint8), 2 bytes (uint16 LE), or 4 bytes (uint32 LE).
      - uint64 FOR64 single-block (int_type=3, combine=0, for_width=11):
        8 bytes (uint64 LE).
  - id: block2_len
    type: u2
    if: header.has_combine
    doc: |
      Total byte length of Block 2 (uint16 LE). Only present when
      combine-with-next=1 (int_type=uint64, two-block mode). Enables
      BlockLength to compute combined size from Block 1's header area alone.
  - id: payload
    size: header.payload_size
    doc: |
      UTL-packed values in lane-interleaved format. Organized as
      ceil(bit_width/4) super-words of 64 bytes each.
  - id: exception_index
    type:
      switch-on: header.exc_count > 16
      cases:
        false: sorted_positions
        true: exception_bitmap
    if: header.has_exceptions
    doc: |
      Exception position index. Format depends on exc_count:
      <= 16 exceptions use sorted byte positions (one byte per exception).
      >  16 exceptions use a 128-bit bitmap (16 bytes, LSB-first per byte).
  - id: svb_data
    size: svb_length
    if: header.has_exceptions
    doc: StreamVByte encoded high bits of exception values.
  - id: block2
    type: utlpfor_block
    if: header.has_combine
    doc: |
      Block 2 (upper 32-bit halves) follows immediately after Block 1.
      Has its own full header with the same count, int_type=uint64,
      combine-with-next=0. Each sub-block independently selects bitwidth,
      exceptions, delta/zigzag, and FOR parameters.

types:
  block_header:
    seq:
      - id: raw
        type: u4
    instances:
      count:
        value: raw & 0xFF
        doc: Number of values in the block (0-128).
      bw_step_index:
        value: (raw >> 8) & 0x1F
        doc: |
          Step bitwidth index (5 bits). For 128-block mode: 0-8, actual
          bit width = bw_step_index * 4. Bit 12 must be 0 in 128-block mode.
          The 5th bit is reserved for the 256-block extension.
      bit_width:
        value: ((raw >> 8) & 0x1F) * 4
        doc: Actual bit width of packed values (0, 4, 8, ..., 32).
      int_type:
        value: (raw >> 13) & 0x03
        doc: Integer type at bits 13-14 (0=uint8, 1=uint16, 2=uint32, 3=uint64).
      for_width:
        value: (raw >> 15) & 0x03
        doc: |
          FOR width at bits 15-16 (00=no FOR, 01=uint8 1B, 10=uint16 2B,
          11=context-dependent: 4B for uint32 or two-block sub-blocks,
          8B for uint64 FOR64 single-block).
      has_for:
        value: ((raw >> 15) & 0x03) != 0
        doc: True when frame-of-reference compression is active.
      for_base_bytes:
        value: >-
          ((raw >> 15) & 0x03) == 0 ? 0 :
          ((raw >> 15) & 0x03) == 1 ? 1 :
          ((raw >> 15) & 0x03) == 2 ? 2 :
          (((raw >> 13) & 0x03) == 3 && (raw & 0x00080000) == 0) ? 8 : 4
        doc: |
          Number of bytes used for the FOR base value.
          0, 1, 2, 4 for standard blocks; 8 for FOR64 single-block
          (int_type=uint64, combine-with-next=0, for_width=11).
      has_combine:
        value: (raw & 0x00080000) != 0
        doc: |
          Combine-with-next flag (bit 19). When set with int_type=uint64,
          indicates a Block 2 (upper 32-bit halves) follows this block.
      has_delta:
        value: (raw & 0x00400000) != 0
        doc: Delta encoding flag (bit 22).
      has_zigzag:
        value: (raw & 0x00800000) != 0
        doc: Zigzag encoding flag (bit 23).
      exc_count:
        value: (raw >> 24) & 0xFF
        doc: Exception count (0 = no exceptions, 1-128 = count).
      has_exceptions:
        value: ((raw >> 24) & 0xFF) > 0
        doc: True when the block contains exceptions.
      payload_size:
        value: >-
          bit_width == 0 ? 0 :
          ((bit_width + 3) / 4) * 64
        doc: |
          Payload size in bytes. Increases in 64-byte super-word steps:
          bw 4 = 64, bw 8 = 128, ..., bw 32 = 512.

  sorted_positions:
    seq:
      - id: positions
        type: u1
        repeat: expr
        repeat-expr: _root.header.exc_count
    doc: |
      Sorted exception positions (one byte per exception, ascending order).
      Used when exc_count <= 16.

  exception_bitmap:
    seq:
      - id: bitmap
        size: 16
    doc: |
      128-bit bitmap (16 bytes) indicating exception positions.
      LSB-first ordering within each byte: bit i is at byte[i/8], position i%8.
      Used when exc_count > 16.
