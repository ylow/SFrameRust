# Parallel CSV Tokenizer Design

## Problem

The Rust CSV parser is ~6x slower than the C++ implementation. Two root causes:
1. Sequential line-splitting (single-threaded tokenization)
2. `Vec<char>` + per-field `String` allocation overhead

The C++ parser achieves parallelism via a quote-parity bitset pre-scan that enables
parallel buffer partitioning while correctly handling multi-line quoted fields.

## Architecture

Per-chunk pipeline (50MB chunks):

```
fill_buffer (read 50MB raw bytes)
  → find_true_newline_positions (sequential, fast bitset scan)
  → parallel_tokenize_and_parse (N threads via rayon)
  → shift buffer (carry forward unparsed remainder)
```

### Quote-Parity Pre-Scan

Sequential pass over `&[u8]` buffer building a `DenseBitset` (1 bit per byte).
Tracks whether each position is inside a quoted field. Uses a fast inner loop
that skips non-special bytes (most bytes are data, not quote/escape/comment chars).

Handles: quote_char, escape_char, double_quote (self-canceling parity),
comment_char (skip to next newline).

### Parallel Tokenize+Parse

Split buffer into N equal chunks. Each thread:
1. Finds start position: first real newline after chunk start (using bitset)
2. Finds end position: first real newline after chunk end
3. Tokenizes + parses all lines in range

Thread 0 starts at buffer[0]. "Real newline" = newline where quote_parity bit is unset.

### Byte-Level Field Splitter

Replace `split_fields(line: &str) -> Vec<String>` (which converts to `Vec<char>`)
with `split_fields_bytes(line: &[u8]) -> Vec<String>`. Same state machine (bracket
lookahead, quote handling, escapes) operating on raw bytes. UTF-8 validation only
when producing final String values.

### Buffer Management

- `Vec<u8>` buffer, managed by main thread
- Read 50MB appended to buffer each iteration
- After parallel parse, shift unparsed remainder to front
- Multi-line quoted fields spanning chunk boundaries: quote_parity marks the
  incomplete field as in-quote, no thread parses it, it carries forward

### Integration

- Pass 1 (inference): raw byte chunks → quote_parity → parallel split_fields_bytes → type inference
- Pass 2 (parsing): background I/O → quote_parity → parallel tokenize+parse → consumer
- Existing `split_lines`/`split_fields` stay for backward compatibility

## Key Data Structures

```rust
struct DenseBitset {
    bits: Vec<u64>,
    len: usize,
}
```

All config characters (delimiter, quote, escape, comment) validated as ASCII at config time.

## Testing

- Existing csv_compat_tests unchanged (test old tokenizer)
- New tests: parallel tokenizer matches sequential output on same input
- Multi-line quoted fields across chunk boundaries
- Benchmark before/after

## Reference

C++ implementation: `parallel_csv_parser.cpp` in turicreate, specifically
`find_true_new_line_positions()` (quote parity scan) and `parse_thread()` (parallel parse).
