meta:
  id: utl_pfor_block
  title: UTL-PFOR Compressed Block
  endian: le
  file-extension: bin

doc: |
  A single UTL-PFOR compressed block containing up to 128 unsigned 32-bit
  integers. Uses Unified Transposed Layout (UTL/FastLanes) with 16 lanes
  and 8 values per lane. Payload is organized in 64-byte super-words.

  Header layout (32-bit little-endian):
    bits  0- 7: count (number of values, 0-128)
    bits  8-12: bw_step_index (0-8 for 128-block, step bitwidth / 4)
    bits 13-14: int_type (0=uint8, 1=uint16, 2=uint32, 3=uint64)
    bits 15-16: for_width (00=no FOR, 01=uint8 1B, 10=uint16 2B, 11=uint32 4B)
    bits 17-18: reserved (2 contiguous bits, must be 0)
    bit  19:    E2 combine-with-next (reserved, must be 0)
    bit  20:    E1 block-length mode (reserved, must be 0)
    bit  21:    SPECIAL flag (reserved, silently ignored)
    bit  22:    delta flag
    bit  23:    zigzag flag
    bits 24-31: exc_count (0 = no exceptions, 1-128 = exception count)

  Step bitwidths: The encoded bw_step_index maps to the actual bitwidth
  as bit_width = bw_step_index * 4. Valid step bitwidths are:
  0, 4, 8, 12, 16, 20, 24, 28, 32 (step indices 0-8).

  Exception index format depends on exc_count:
    exc_count <= 16: sorted byte positions (exc_count bytes)
    exc_count >  16: bitmap (16 bytes, 128 bits, LSB-first per byte)

  Exception high bits are encoded using StreamVByte.

  Wire layout (no exc, no FOR):    [header:4][payload:N]
  Wire layout (no exc, FOR):       [header:4][for_base:1|2|4][payload:N]
  Wire layout (exc, no FOR):       [header:4][svb_length:2][payload:N][exc_index][svb_data]
  Wire layout (exc, FOR):          [header:4][svb_length:2][for_base:1|2|4][payload:N][exc_index][svb_data]

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
      FOR base value (minimum of original values). Width depends on
      for_width: 1 byte (uint8), 2 bytes (uint16 LE), or 4 bytes (uint32 LE).
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
          FOR width at bits 15-16 (00=no FOR, 01=uint8 1B, 10=uint16 2B, 11=uint32 4B).
      has_for:
        value: ((raw >> 15) & 0x03) != 0
        doc: True when frame-of-reference compression is active.
      for_base_bytes:
        value: >-
          ((raw >> 15) & 0x03) == 0 ? 0 :
          ((raw >> 15) & 0x03) == 1 ? 1 :
          ((raw >> 15) & 0x03) == 2 ? 2 : 4
        doc: Number of bytes used for the FOR base value (0, 1, 2, or 4).
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
