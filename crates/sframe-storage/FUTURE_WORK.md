# Future Work: Performance Optimizations

## Integer Unpack Specialization (`codec_integer.rs`)

The `read_packed_values` function uses a generic bit-extraction loop for all bit
widths. Since bit widths are always powers of 2 (1, 2, 4, 8, 16, 32, 64), we
can use dedicated unpack routines per width — matching what the C++ code does
with Duff's device in `integer_pack_impl.hpp`:

- **1-bit**: Process 8 values per byte with shift-and-mask
- **2-bit**: Process 4 values per byte
- **4-bit**: Process 2 values per byte (low/high nibble)
- **8/16/32**: Direct byte/word reads from the packed buffer
- **64**: Already has a fast path

Must preserve the MSB-first padding offset for partial groups.

The FoR-delta loops also call `output.last().unwrap()` on every iteration, which
adds bounds checking. Caching the last value in a local variable avoids this.

## String Dictionary Cloning (`codec_string.rs`)

Dictionary-encoded strings clone the `String` on every lookup (`dict[idx].clone()`).
For columns with high repetition this means N heap allocations for N rows. Using
`Arc<str>` in the dictionary would make lookups a cheap reference-count bump.

## Vector Slice Copies (`codec_vector.rs`)

`flat_values[offset..len].to_vec()` copies each sub-vector's data. Could return
slices into a shared backing buffer or use `Arc<[f64]>` sub-slices to avoid the
per-row allocation.

## Float Bit Rotation (`codec_float.rs`)

The legacy float decode does `(encoded >> 1) | (encoded << 63)` per value.
This is a rotate-right-1 and is straightforward to vectorize with SIMD (SSE2/NEON).
The integer-to-f64 reinterpret loop is also a candidate for auto-vectorization
if restructured as `iter().map().collect()`.

## Bitmap Operations (`block_decode.rs`)

Counting defined values iterates bit-by-bit through the bitmap. Using `popcnt`
on the bitmap's backing words would be significantly faster. The merge loop that
interleaves defined values with Undefined sentinels could also benefit from
SIMD gather operations.

## Segment I/O Batching (`segment_reader.rs`)

Each block read does a separate seek + read syscall. When reading an entire
column sequentially, blocks are often adjacent or nearby in the file. Batching
adjacent reads into a single large read (or using buffered I/O) would reduce
syscall overhead.

The uncompressed-block path also does `raw.to_vec()` — a `Cow<[u8]>` would
avoid the copy when no decompression is needed.

## Minor Items

- `column_types.clone()` per segment in `sframe_reader.rs` — share via `Arc`
- Linear column name lookup in `read_column_by_name` — build a `HashMap` if
  callers look up many columns
