# Parallel CSV Tokenizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace sequential CSV tokenization with a quote-parity bitset pre-scan + parallel tokenize+parse, matching the C++ turicreate architecture.

**Architecture:** A sequential `find_true_newline_positions()` builds a compact bitset marking which byte positions are inside quotes. Then N threads each handle an equal chunk of the buffer, using the bitset to find real line boundaries and independently tokenize+parse their chunk. All operations work on `&[u8]` instead of `Vec<char>`.

**Tech Stack:** Rust, rayon (already in deps), no new crate dependencies.

---

### Task 1: DenseBitset + Module Skeleton

**Files:**
- Create: `crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs`
- Modify: `crates/sframe-query/src/algorithms/mod.rs`

**Step 1: Add module declaration**

In `mod.rs`, add:
```rust
pub mod csv_parallel_tokenizer;
```

**Step 2: Write DenseBitset tests**

```rust
// In csv_parallel_tokenizer.rs

/// Compact bitset — 1 bit per byte position in a CSV buffer.
/// Used to track quote parity: bit set means "inside a quoted field".
pub(crate) struct DenseBitset {
    bits: Vec<u64>,
    len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_basic() {
        let mut bs = DenseBitset::new(200);
        assert!(!bs.get(0));
        assert!(!bs.get(199));
        bs.set(0);
        bs.set(63);
        bs.set(64);
        bs.set(199);
        assert!(bs.get(0));
        assert!(bs.get(63));
        assert!(bs.get(64));
        assert!(bs.get(199));
        assert!(!bs.get(1));
        assert!(!bs.get(65));
    }

    #[test]
    fn test_bitset_clear() {
        let mut bs = DenseBitset::new(100);
        bs.set(50);
        assert!(bs.get(50));
        bs.clear_bit(50);
        assert!(!bs.get(50));
    }

    #[test]
    fn test_bitset_clear_all() {
        let mut bs = DenseBitset::new(200);
        bs.set(10);
        bs.set(100);
        bs.clear_all();
        assert!(!bs.get(10));
        assert!(!bs.get(100));
    }
}
```

**Step 3: Run tests — expect FAIL (methods not implemented)**

```bash
cargo test -p sframe-query --lib algorithms::csv_parallel_tokenizer -- --nocapture
```

**Step 4: Implement DenseBitset**

```rust
impl DenseBitset {
    pub fn new(len: usize) -> Self {
        let nwords = (len + 63) / 64;
        DenseBitset {
            bits: vec![0u64; nwords],
            len,
        }
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        self.bits[idx >> 6] |= 1u64 << (idx & 63);
    }

    #[inline(always)]
    pub fn clear_bit(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        self.bits[idx >> 6] &= !(1u64 << (idx & 63));
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len);
        (self.bits[idx >> 6] >> (idx & 63)) & 1 != 0
    }

    pub fn clear_all(&mut self) {
        self.bits.fill(0);
    }

    /// Resize and clear. Reuses allocation when possible.
    pub fn resize_and_clear(&mut self, len: usize) {
        let nwords = (len + 63) / 64;
        self.bits.resize(nwords, 0);
        self.bits.fill(0);
        self.len = len;
    }
}
```

**Step 5: Run tests — expect PASS**

```bash
cargo test -p sframe-query --lib algorithms::csv_parallel_tokenizer -- --nocapture
```

**Step 6: Commit**

```bash
git add crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs crates/sframe-query/src/algorithms/mod.rs
git commit -m "feat: add DenseBitset and csv_parallel_tokenizer module skeleton"
```

---

### Task 2: ByteConfig + Quote-Parity Scanner

**Files:**
- Modify: `crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs`

**Step 1: Add ByteConfig struct and find_true_newline_positions signature**

```rust
use super::csv_tokenizer::CsvConfig;
use sframe_types::error::{Result, SFrameError};

/// Pre-computed byte-level config for hot-path tokenization.
/// All special characters must be ASCII (validated at construction).
pub(crate) struct ByteConfig {
    pub delimiter: Vec<u8>,
    pub quote: u8,
    pub escape: u8,
    pub comment: Option<u8>,
    pub double_quote: bool,
    pub skip_initial_space: bool,
    pub is_regular_line_terminator: bool,
    pub line_terminator: Vec<u8>,
}

impl ByteConfig {
    pub fn from_config(config: &CsvConfig) -> Result<Self> {
        let quote = config.quote_char;
        let escape = config.escape_char;
        if !quote.is_ascii() {
            return Err(SFrameError::Format("quote_char must be ASCII".into()));
        }
        if !escape.is_ascii() {
            return Err(SFrameError::Format("escape_char must be ASCII".into()));
        }
        if let Some(c) = config.comment_char {
            if !c.is_ascii() {
                return Err(SFrameError::Format("comment_char must be ASCII".into()));
            }
        }
        let is_regular = config.line_terminator == "\n"
            || config.line_terminator == "\r\n"
            || config.line_terminator == "\r";
        Ok(ByteConfig {
            delimiter: config.delimiter.as_bytes().to_vec(),
            quote: quote as u8,
            escape: escape as u8,
            comment: config.comment_char.map(|c| c as u8),
            double_quote: config.double_quote,
            skip_initial_space: config.skip_initial_space,
            is_regular_line_terminator: is_regular,
            line_terminator: config.line_terminator.as_bytes().to_vec(),
        })
    }
}
```

**Step 2: Write quote-parity tests**

```rust
#[cfg(test)]
mod tests {
    // ... existing bitset tests ...

    #[test]
    fn test_quote_parity_no_quotes() {
        let buf = b"hello,world\n1,2\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // No quotes → no bits set
        for i in 0..buf.len() {
            assert!(!bp.get(i), "bit {} should not be set", i);
        }
    }

    #[test]
    fn test_quote_parity_simple_quote() {
        // "hi" — positions of h and i should be in-quote
        let buf = b"\"hi\"\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // byte 0 = " → parity flips to true, bit set
        assert!(bp.get(0));
        // byte 1 = h → in quote, set
        assert!(bp.get(1));
        // byte 2 = i → in quote, set
        assert!(bp.get(2));
        // byte 3 = " → parity flips to false, not set
        assert!(!bp.get(3));
        // byte 4 = \n → not in quote
        assert!(!bp.get(4));
    }

    #[test]
    fn test_quote_parity_multiline() {
        // A quoted field spanning a newline:
        // "hello\nworld"\n
        let buf = b"\"hello\nworld\"\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // The \n at position 6 is inside quotes → bit set
        assert!(bp.get(6));
        // The \n at position 13 is outside quotes → bit not set
        assert!(!bp.get(13));
    }

    #[test]
    fn test_quote_parity_escaped_quote() {
        // "say \"hi\"" — escape prevents quote from toggling parity
        let buf = b"\"say \\\"hi\\\"\"\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // Final \n should be outside quotes
        assert!(!bp.get(buf.len() - 1));
    }

    #[test]
    fn test_quote_parity_comment_outside_quote() {
        let buf = b"# this is a comment\n1,2\n";
        let mut config = CsvConfig::default();
        config.comment_char = Some('#');
        let cfg = ByteConfig::from_config(&config).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // Nothing should be in-quote
        for i in 0..buf.len() {
            assert!(!bp.get(i));
        }
    }
}
```

**Step 3: Run tests — expect FAIL**

```bash
cargo test -p sframe-query --lib algorithms::csv_parallel_tokenizer -- --nocapture
```

**Step 4: Implement find_true_newline_positions**

Port of C++ `find_true_new_line_positions()`. Key: fast inner loop that skips
non-special bytes, only enters slow path for quote/escape/comment chars.

```rust
/// Build a bitset marking which byte positions are inside quoted fields.
///
/// After this, a newline at position `i` is a *real* line boundary iff
/// `!bitset.get(i)`. This enables parallel line splitting: each thread
/// can independently find its start/end boundaries using the bitset.
///
/// Ported from C++ `find_true_new_line_positions()` in parallel_csv_parser.cpp.
pub(crate) fn find_true_newline_positions(buffer: &[u8], config: &ByteConfig) -> DenseBitset {
    let mut bp = DenseBitset::new(buffer.len());
    if buffer.is_empty() {
        return bp;
    }

    let quote = config.quote;
    let escape = config.escape;
    let has_comment = config.comment.is_some();
    let comment = config.comment.unwrap_or(0);

    let mut cur_in_quote = false;
    // `not_esc` = true means the previous char was NOT an escape char.
    // Using the inverted flag saves a negation per iteration (C++ trick).
    let mut not_esc = true;
    let mut idx: usize = 0;
    let mut pos: usize = 0;
    let buf = buffer;
    let len = buf.len();

    if has_comment {
        while pos < len {
            // Fast path: skip bytes that aren't special
            if not_esc {
                let start = pos;
                while pos < len
                    && buf[pos] != comment
                    && buf[pos] != quote
                    && buf[pos] != escape
                {
                    pos += 1;
                }
                let end = pos;
                if cur_in_quote {
                    for i in idx..end {
                        bp.set(i);
                    }
                }
                idx = end;
                if pos >= len {
                    break;
                }
            }

            let c = buf[pos];

            // Comment outside quotes: skip to next newline
            if c == comment && not_esc && !cur_in_quote {
                let next = advance_past_newline(buf, pos, len, config);
                let advanced = next.0;
                idx = idx + (advanced - pos);
                pos = advanced;
                if !next.1 {
                    break; // no newline found, done
                }
                continue;
            }

            let is_quote_char = (c == quote) && not_esc;
            cur_in_quote ^= is_quote_char;
            bp.set_to(idx, cur_in_quote);
            // not_esc logic: if we weren't escaped, check if current char
            // IS the escape char. If we were escaped, unconditionally clear.
            not_esc = !not_esc || c != escape;
            idx += 1;
            pos += 1;
        }
    } else {
        // No comment char — simpler/faster loop
        while pos < len {
            if not_esc {
                let start_pos = pos;
                while pos < len && buf[pos] != quote && buf[pos] != escape {
                    pos += 1;
                }
                let end = pos;
                if cur_in_quote {
                    for i in idx..end {
                        bp.set(i);
                    }
                }
                idx = end;
                if pos >= len {
                    break;
                }
            }

            let c = buf[pos];
            let is_quote_char = (c == quote) && not_esc;
            cur_in_quote ^= is_quote_char;
            bp.set_to(idx, cur_in_quote);
            not_esc = !not_esc || c != escape;
            idx += 1;
            pos += 1;
        }
    }

    bp
}
```

Also add `set_to` helper on DenseBitset:

```rust
#[inline(always)]
pub fn set_to(&mut self, idx: usize, val: bool) {
    debug_assert!(idx < self.len);
    if val {
        self.set(idx);
    } else {
        self.clear_bit(idx);
    }
}
```

**Step 5: Implement advance_past_newline**

```rust
/// Advance past the next newline starting at `pos`. Returns (new_pos, found).
/// For standard terminators (\n, \r, \r\n), handles all three.
/// For custom terminators, matches the exact byte sequence.
#[inline]
fn advance_past_newline(buf: &[u8], mut pos: usize, end: usize, config: &ByteConfig) -> (usize, bool) {
    if config.is_regular_line_terminator {
        while pos < end {
            if buf[pos] == b'\n' {
                return (pos + 1, true);
            } else if buf[pos] == b'\r' {
                if pos + 1 < end && buf[pos + 1] == b'\n' {
                    return (pos + 2, true);
                }
                return (pos + 1, true);
            }
            pos += 1;
        }
    } else if !config.line_terminator.is_empty() {
        let term = &config.line_terminator;
        while pos + term.len() <= end {
            if &buf[pos..pos + term.len()] == term.as_slice() {
                return (pos + term.len(), true);
            }
            pos += 1;
        }
    }
    (end, false)
}

/// Like advance_past_newline but only accepts newlines where
/// the quote_parity bit is NOT set (i.e., real line boundaries).
fn advance_past_newline_with_parity(
    buf: &[u8],
    mut pos: usize,
    end: usize,
    config: &ByteConfig,
    parity: &DenseBitset,
) -> (usize, bool) {
    let mut found = false;
    while pos < end {
        let (next, matched) = advance_past_newline(buf, pos, end, config);
        if !matched {
            return (next, false);
        }
        // Check the parity of the byte just before next (the last byte of the newline).
        // If parity is set, this newline is inside quotes — skip it.
        if parity.get(next - 1) {
            pos = next;
            found = false;
            continue;
        }
        return (next, true);
    }
    (pos, false)
}
```

**Step 6: Run tests — expect PASS**

```bash
cargo test -p sframe-query --lib algorithms::csv_parallel_tokenizer -- --nocapture
```

**Step 7: Commit**

```bash
git add crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs
git commit -m "feat: quote-parity scanner (find_true_newline_positions)"
```

---

### Task 3: Byte-Level Field Splitter

Port `split_fields` and `finish_field` from `csv_tokenizer.rs` to operate on `&[u8]`.

**Files:**
- Modify: `crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs`

**Step 1: Write tests that compare byte-level splitter against existing char-level splitter**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::csv_tokenizer;

    /// Helper: verify byte-level split matches char-level split for same input.
    fn assert_fields_match(line: &str, config: &CsvConfig) {
        let expected = csv_tokenizer::split_fields(line, config);
        let bcfg = ByteConfig::from_config(config).unwrap();
        let actual = split_fields_bytes(line.as_bytes(), &bcfg);
        assert_eq!(expected, actual, "mismatch for line: {:?}", line);
    }

    #[test]
    fn test_byte_split_basic() {
        let config = CsvConfig::default();
        assert_fields_match("1,2,3", &config);
        assert_fields_match("hello, world, foo", &config);
    }

    #[test]
    fn test_byte_split_quoted() {
        let config = CsvConfig::default();
        assert_fields_match("\"hello, world\",42", &config);
        assert_fields_match("\"say \"\"hi\"\"\",bar", &config);
    }

    #[test]
    fn test_byte_split_escaped() {
        let config = CsvConfig::default();
        assert_fields_match("\"hello\\nworld\",42", &config);
        assert_fields_match("\"tab\\there\"", &config);
    }

    #[test]
    fn test_byte_split_brackets() {
        let config = CsvConfig::default();
        assert_fields_match("foo,[1,2,3]", &config);
        assert_fields_match("foo,{a:1,b:2}", &config);
        assert_fields_match("test,[1,[2,3],4]", &config);
    }

    #[test]
    fn test_byte_split_comment() {
        let config = CsvConfig::default();
        assert_fields_match("hello # comment", &config);
    }

    #[test]
    fn test_byte_split_trailing_delimiter() {
        let config = CsvConfig::default();
        assert_fields_match("a,b,", &config);
        assert_fields_match(",", &config);
    }

    #[test]
    fn test_byte_split_multichar_delimiter() {
        let config = CsvConfig {
            delimiter: "::".to_string(),
            ..Default::default()
        };
        assert_fields_match("a::b::c", &config);
    }
}
```

**Step 2: Run tests — expect FAIL**

```bash
cargo test -p sframe-query --lib algorithms::csv_parallel_tokenizer::tests::test_byte_split -- --nocapture
```

**Step 3: Implement split_fields_bytes and helpers**

Port `split_fields`, `finish_field`, `unescape_csv_string`, `find_balanced_close`,
`has_colon_at_depth0`, and `skip_post_bracket` from `csv_tokenizer.rs` to operate
on `&[u8]`.

Key differences from the char-level version:
- Input is `&[u8]` instead of `&str`
- No `Vec<char>` allocation — index directly into the byte slice
- Field accumulation uses `Vec<u8>` buffer
- Final `String` produced via `String::from_utf8` (or `from_utf8_lossy`)
- All char comparisons become byte comparisons (ASCII-only special chars)

The functions to port (reference the existing implementations in `csv_tokenizer.rs`):

```rust
/// Byte-level field splitting. Same semantics as csv_tokenizer::split_fields
/// but operates on raw bytes for performance.
pub(crate) fn split_fields_bytes(line: &[u8], config: &ByteConfig) -> Vec<String> {
    // Same state machine as split_fields in csv_tokenizer.rs:
    // - in_quote, escape_next, in_field, had_delimiter
    // - bracket_depth, brace_depth
    // - bracket/brace lookahead at field start
    // - comment char handling
    // All operating on &[u8] with byte comparisons.
    // See csv_tokenizer.rs:417-593 for the char-level reference.
    ...
}

fn finish_field_bytes(raw: &[u8], config: &ByteConfig) -> String {
    // Trim whitespace, unquote, unescape. Same as csv_tokenizer::finish_field
    // but on bytes. Convert to String at the end.
    ...
}

fn unescape_csv_bytes(s: &[u8], escape: u8) -> Vec<u8> {
    // Same as csv_tokenizer::unescape_csv_string but on bytes.
    // Handles \n, \t, \r, \\, \", \', \/, \b, \f, \uXXXX (incl. surrogates).
    ...
}

fn find_balanced_close_bytes(
    buf: &[u8], start: usize, open: u8, close: u8,
    delim: &[u8], skip_initial_space: bool,
) -> Option<usize> {
    // Same as csv_tokenizer::find_balanced_close but on bytes.
    ...
}

fn has_colon_at_depth0_bytes(buf: &[u8]) -> bool {
    // Same as csv_tokenizer::has_colon_at_depth0 but on bytes.
    ...
}
```

These are direct ports — same logic, same edge cases, just `&[u8]` instead of
`&[char]`/`Vec<char>`. The reference implementations are in `csv_tokenizer.rs`.

**Step 4: Run tests — expect PASS**

```bash
cargo test -p sframe-query --lib algorithms::csv_parallel_tokenizer::tests::test_byte_split -- --nocapture
```

**Step 5: Add exhaustive comparison test using csv_compat_tests patterns**

Write a test that runs the byte-level splitter against ALL test cases from
`csv_compat_tests.rs` (the ones that use `split_fields`). This ensures full
parity with the char-level tokenizer:

```rust
#[test]
fn test_byte_split_exhaustive_parity() {
    // Run a variety of edge cases from csv_compat_tests through both
    // tokenizers and verify identical output.
    let cases = vec![
        ("a,b,c", CsvConfig::default()),
        ("\"hello, world\",42", CsvConfig::default()),
        // ... (pull representative cases from csv_compat_tests.rs)
    ];
    for (line, config) in &cases {
        assert_fields_match(line, config);
    }
}
```

**Step 6: Run full test suite — expect PASS**

```bash
cargo test -p sframe-query -- --nocapture
```

**Step 7: Commit**

```bash
git add crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs
git commit -m "feat: byte-level field splitter (split_fields_bytes)"
```

---

### Task 4: Parallel Tokenize Orchestration

**Files:**
- Modify: `crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs`

**Step 1: Write tests for parallel tokenization**

```rust
#[test]
fn test_parallel_tokenize_basic() {
    let input = b"a,b,c\n1,2,3\n4,5,6\n7,8,9\n";
    let config = CsvConfig::default();
    let bcfg = ByteConfig::from_config(&config).unwrap();
    let (rows, last_parsed) = parallel_tokenize(input, &bcfg);
    assert_eq!(last_parsed, input.len());
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0], vec!["a", "b", "c"]);
    assert_eq!(rows[1], vec!["1", "2", "3"]);
    assert_eq!(rows[2], vec!["4", "5", "6"]);
    assert_eq!(rows[3], vec!["7", "8", "9"]);
}

#[test]
fn test_parallel_tokenize_multiline_quote() {
    let input = b"a,b\n\"line1\nline2\",42\n3,4\n";
    let config = CsvConfig::default();
    let bcfg = ByteConfig::from_config(&config).unwrap();
    let (rows, _) = parallel_tokenize(input, &bcfg);
    assert_eq!(rows.len(), 3); // header + 2 data rows
    assert_eq!(rows[0], vec!["a", "b"]);
    assert_eq!(rows[1], vec!["line1\nline2", "42"]);
    assert_eq!(rows[2], vec!["3", "4"]);
}

#[test]
fn test_parallel_tokenize_incomplete_line() {
    // Buffer ends mid-line — last_parsed should NOT include it
    let input = b"a,b\n1,2\n3,4";  // no trailing newline on last line
    let config = CsvConfig::default();
    let bcfg = ByteConfig::from_config(&config).unwrap();
    let (rows, last_parsed) = parallel_tokenize(input, &bcfg);
    // "3,4" has no terminator. In non-EOF mode this is incomplete.
    // Call parallel_tokenize_eof for EOF handling.
    assert_eq!(rows.len(), 2);
    assert_eq!(last_parsed, 8); // after "1,2\n"
}

#[test]
fn test_parallel_tokenize_eof() {
    // At EOF, the last line is included even without terminator
    let input = b"a,b\n1,2\n3,4";
    let config = CsvConfig::default();
    let bcfg = ByteConfig::from_config(&config).unwrap();
    let (rows, last_parsed) = parallel_tokenize_eof(input, &bcfg);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[2], vec!["3", "4"]);
    assert_eq!(last_parsed, input.len());
}

#[test]
fn test_parallel_matches_sequential() {
    // Compare parallel tokenizer output against sequential tokenizer
    let input = "name,value\n\"hello, world\",42\nfoo,[1,2,3]\n\"multi\nline\",99\nbar,{a:1}\n";
    let config = CsvConfig::default();
    let (seq_header, seq_rows) = csv_tokenizer::tokenize(input, &config);

    let bcfg = ByteConfig::from_config(&config).unwrap();
    let (par_rows, _) = parallel_tokenize_eof(input.as_bytes(), &bcfg);

    // Sequential tokenize separates header; parallel returns all rows.
    let mut expected = Vec::new();
    if let Some(h) = seq_header {
        expected.push(h);
    }
    expected.extend(seq_rows);
    assert_eq!(par_rows, expected);
}
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement parallel_tokenize**

```rust
use rayon::prelude::*;

/// Result of parallel tokenization: (rows, last_parsed_byte_offset).
/// `rows` are in document order. `last_parsed` is the byte offset past
/// the last fully-parsed line. Bytes `[last_parsed..buffer.len()]` are
/// an incomplete line that should be carried forward to the next buffer.
pub(crate) fn parallel_tokenize(
    buffer: &[u8],
    config: &ByteConfig,
) -> (Vec<Vec<String>>, usize) {
    parallel_tokenize_inner(buffer, config, false)
}

/// Like parallel_tokenize, but treats the buffer as the final chunk (EOF).
/// Ensures the last line is parsed even without a trailing newline.
pub(crate) fn parallel_tokenize_eof(
    buffer: &[u8],
    config: &ByteConfig,
) -> (Vec<Vec<String>>, usize) {
    parallel_tokenize_inner(buffer, config, true)
}

fn parallel_tokenize_inner(
    buffer: &[u8],
    config: &ByteConfig,
    eof: bool,
) -> (Vec<Vec<String>>, usize) {
    if buffer.is_empty() {
        return (Vec::new(), 0);
    }

    // 1. Build quote parity
    let mut parity = find_true_newline_positions(buffer, config);
    if eof {
        // At EOF, force the last byte to have no parity so the last line
        // is always treated as complete.
        parity.clear_bit(buffer.len() - 1);
    }

    // 2. Determine thread count
    let nthreads = rayon::current_num_threads().max(1);
    let len = buffer.len();

    // 3. Parallel parse: each thread handles a section
    let results: Vec<(Vec<Vec<String>>, usize)> = (0..nthreads)
        .into_par_iter()
        .map(|tid| {
            parse_thread(buffer, &parity, config, tid, nthreads)
        })
        .collect();

    // 4. Merge: concatenate rows in order, find last parsed position
    let mut all_rows = Vec::new();
    let mut last_parsed = 0usize;
    for (rows, thread_last) in results {
        all_rows.extend(rows);
        if thread_last > last_parsed {
            last_parsed = thread_last;
        }
    }

    (all_rows, last_parsed)
}

/// Parse a section of the buffer assigned to `thread_id` out of `nthreads`.
/// Returns (rows, last_parsed_position).
///
/// Port of C++ parse_thread() in parallel_csv_parser.cpp.
fn parse_thread(
    buffer: &[u8],
    parity: &DenseBitset,
    config: &ByteConfig,
    thread_id: usize,
    nthreads: usize,
) -> (Vec<Vec<String>>, usize) {
    let len = buffer.len();
    let step = len / nthreads;

    let mut pstart = thread_id * step;
    let mut pend = if thread_id == nthreads - 1 { len } else { (thread_id + 1) * step };

    // Find start: thread 0 starts at 0. Others find first real newline after pstart.
    let start_found;
    if thread_id == 0 {
        start_found = true;
    } else {
        // For multi-char terminators, shift backwards
        if !config.is_regular_line_terminator && config.line_terminator.len() > 1 {
            let shift = config.line_terminator.len() - 1;
            if pstart >= shift {
                pstart -= shift;
            }
        }
        let (pos, found) = advance_past_newline_with_parity(buffer, pstart, pend, config, parity);
        pstart = pos;
        start_found = found;
    }

    if !start_found {
        return (Vec::new(), pstart);
    }

    // Find end: advance past the first real newline after pend
    if !config.is_regular_line_terminator && config.line_terminator.len() > 1 {
        let shift = config.line_terminator.len() - 1;
        if pend >= shift {
            pend -= shift;
        }
    }
    let (end_pos, _) = advance_past_newline_with_parity(buffer, pend, len, config, parity);
    pend = end_pos;

    // Parse lines in [pstart..pend)
    let mut rows = Vec::new();
    let mut line_start = pstart;
    let mut pos = pstart;

    if config.line_terminator.is_empty() {
        // No line terminator: entire range is one line
        let line = &buffer[pstart..pend];
        if !line.is_empty() {
            rows.push(split_fields_bytes(line, config));
        }
        return (rows, pend);
    }

    while pos < pend {
        if is_newline_byte(buffer, pos, config) && !parity.get(pos) {
            let line = &buffer[line_start..pos];
            if !line.is_empty() {
                rows.push(split_fields_bytes(line, config));
            }
            let (next, _) = advance_past_newline_with_parity(
                buffer, pos, pend, config, parity,
            );
            line_start = next;
            pos = next;
        } else {
            pos += 1;
        }
    }

    (rows, line_start)
}

/// Check if the byte at `pos` starts a newline sequence.
#[inline]
fn is_newline_byte(buf: &[u8], pos: usize, config: &ByteConfig) -> bool {
    if config.is_regular_line_terminator {
        buf[pos] == b'\n' || buf[pos] == b'\r'
    } else if !config.line_terminator.is_empty() {
        let term = &config.line_terminator;
        pos + term.len() <= buf.len() && &buf[pos..pos + term.len()] == term.as_slice()
    } else {
        false
    }
}
```

**Step 4: Run tests — expect PASS**

```bash
cargo test -p sframe-query --lib algorithms::csv_parallel_tokenizer -- --nocapture
```

**Step 5: Commit**

```bash
git add crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs
git commit -m "feat: parallel CSV tokenization with quote-parity bitset"
```

---

### Task 5: Integrate with CsvStreamingParse

Replace sequential tokenization in `CsvStreamingParse::open` (pass 1) and
`CsvStreamingParse::parse_chunks` (pass 2) with the parallel tokenizer.

**Files:**
- Modify: `crates/sframe-query/src/algorithms/csv_parser.rs`

**Step 1: Refactor CsvStreamingParse::open to use parallel tokenizer**

Replace the current approach (BufReader → split_lines_streaming → parallel
split_fields) with:
- Read raw bytes into a managed `Vec<u8>` buffer
- Build quote parity + parallel tokenize per chunk
- Feed rows into type inference

Key changes in `open()`:
```rust
pub fn open(path: &str, options: &CsvOptions) -> Result<Self> {
    use super::csv_parallel_tokenizer::{ByteConfig, parallel_tokenize, parallel_tokenize_eof};

    let config = options_to_config(options);
    let bcfg = ByteConfig::from_config(&config)?;
    let na_values: HashSet<String> = options.na_values.iter().cloned().collect();
    let hint_map: std::collections::HashMap<&str, FlexTypeEnum> = options
        .type_hints
        .iter()
        .map(|(n, t)| (n.as_str(), *t))
        .collect();

    let mut file = std::fs::File::open(path).map_err(SFrameError::Io)?;
    let mut buffer: Vec<u8> = Vec::with_capacity(CSV_CHUNK_SIZE + CSV_CHUNK_SIZE / 4);
    let mut column_names: Option<Vec<String>> = None;
    let mut infer_state: Option<TypeInferenceState> = None;
    let mut rows_skipped = 0usize;
    let mut row_count = 0usize;

    loop {
        // Read up to CSV_CHUNK_SIZE bytes, appending to buffer
        let old_len = buffer.len();
        buffer.resize(old_len + CSV_CHUNK_SIZE, 0);
        let n = std::io::Read::read(&mut file, &mut buffer[old_len..])
            .map_err(SFrameError::Io)?;
        buffer.truncate(old_len + n);
        let is_eof = n == 0 || n < CSV_CHUNK_SIZE;

        if buffer.is_empty() {
            break;
        }

        // Ensure buffer ends with a line terminator at EOF
        if is_eof && !buffer.is_empty() {
            ensure_line_terminator(&mut buffer, &bcfg);
        }

        // Parallel tokenize
        let (rows, last_parsed) = if is_eof {
            parallel_tokenize_eof(&buffer, &bcfg)
        } else {
            parallel_tokenize(&buffer, &bcfg)
        };

        // Shift buffer: keep unparsed remainder
        buffer.drain(..last_parsed);

        // Process rows for schema inference
        for row in rows {
            if csv_tokenizer::is_comment(&row.join(&config.delimiter), &config) {
                continue; // (better: check first field or raw line)
            }
            if rows_skipped < config.skip_rows {
                rows_skipped += 1;
                continue;
            }
            if column_names.is_none() {
                if config.has_header {
                    column_names = Some(row);
                } else {
                    let n = row.len();
                    column_names = Some((0..n).map(|i| format!("X{}", i + 1)).collect());
                    let mut state = TypeInferenceState::new(n);
                    state.observe_row(&row, &na_values);
                    infer_state = Some(state);
                    row_count += 1;
                }
                continue;
            }
            if let Some(limit) = config.row_limit {
                if row_count >= limit { continue; }
            }
            let ncols = column_names.as_ref().unwrap().len();
            if infer_state.is_none() {
                infer_state = Some(TypeInferenceState::new(ncols));
            }
            infer_state.as_mut().unwrap().observe_row(&row, &na_values);
            row_count += 1;
        }

        if is_eof { break; }
    }

    // ... rest unchanged (finalize types, output_indices, etc.) ...
}
```

Note: The comment-line detection needs refinement — currently `split_fields_bytes`
already handles comment lines (returns early). A better approach: add a raw-line
check before splitting, or detect comments in `parse_thread`. For now, keep it
simple: let `split_fields_bytes` handle inline comments, and add a separate check
for full-line comments (check if first non-space byte is comment_char).

**Step 2: Refactor CsvStreamingParse::parse_chunks**

Replace the background I/O thread that sends `Vec<String>` lines with one that
sends `Vec<u8>` raw byte buffers. The main thread builds quote parity and
parallel-parses.

Key changes:
```rust
pub fn parse_chunks<F>(&self, mut consumer: F) -> Result<()>
where
    F: FnMut(Vec<Vec<FlexType>>) -> Result<()>,
{
    use super::csv_parallel_tokenizer::{ByteConfig, parallel_tokenize, parallel_tokenize_eof};
    use rayon::prelude::*;

    let config = options_to_config(&self.options);
    let bcfg = ByteConfig::from_config(&config)?;
    let na_values: HashSet<String> = self.options.na_values.iter().cloned().collect();
    let col_types = &self.col_types_full;
    let col_indices: Vec<usize> = if let Some(ref out) = self.output_indices {
        out.clone()
    } else {
        (0..col_types.len()).collect()
    };
    let n_out = col_indices.len();

    let path = self.path.clone();
    let has_header = self.options.has_header;
    let skip_rows = self.options.skip_rows;
    let row_limit = self.options.row_limit;

    // Background I/O thread: reads raw byte chunks
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<(Vec<u8>, bool)>>(2);

    std::thread::spawn(move || {
        let mut file = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) => { let _ = tx.send(Err(SFrameError::Io(e))); return; }
        };
        let mut leftover: Vec<u8> = Vec::new();
        loop {
            let mut buf = std::mem::take(&mut leftover);
            let old_len = buf.len();
            buf.resize(old_len + CSV_CHUNK_SIZE, 0);
            let n = match std::io::Read::read(&mut file, &mut buf[old_len..]) {
                Ok(n) => n,
                Err(e) => { let _ = tx.send(Err(SFrameError::Io(e))); return; }
            };
            buf.truncate(old_len + n);
            let is_eof = n == 0 || n < CSV_CHUNK_SIZE;
            if buf.is_empty() { break; }
            if tx.send(Ok((buf, is_eof))).is_err() { return; }
            if is_eof { break; }
        }
    });

    // Main thread: parallel tokenize + parse
    let mut header_consumed = !has_header;
    let mut rows_skipped = 0usize;
    let mut total_rows = 0usize;
    let mut leftover: Vec<u8> = Vec::new();

    for msg in rx {
        let (mut chunk, is_eof) = msg?;

        // Prepend leftover from previous chunk
        if !leftover.is_empty() {
            let mut combined = std::mem::take(&mut leftover);
            combined.extend_from_slice(&chunk);
            chunk = combined;
        }

        if is_eof && !chunk.is_empty() {
            ensure_line_terminator(&mut chunk, &bcfg);
        }

        let (rows, last_parsed) = if is_eof {
            parallel_tokenize_eof(&chunk, &bcfg)
        } else {
            parallel_tokenize(&chunk, &bcfg)
        };

        // Save unparsed remainder
        if last_parsed < chunk.len() {
            leftover = chunk[last_parsed..].to_vec();
        }

        // Filter and parse rows
        let mut data_rows: Vec<Vec<String>> = Vec::with_capacity(rows.len());
        for row in rows {
            if !header_consumed {
                header_consumed = true;
                continue;
            }
            if rows_skipped < skip_rows {
                rows_skipped += 1;
                continue;
            }
            // Skip empty rows
            if row.is_empty() || (row.len() == 1 && row[0].trim().is_empty()) {
                continue;
            }
            if let Some(limit) = row_limit {
                if total_rows >= limit { break; }
            }
            data_rows.push(row);
            total_rows += 1;
        }

        if data_rows.is_empty() { continue; }

        // Parallel parse to FlexType (rayon over rows)
        let row_results: Vec<Result<Vec<FlexType>>> = data_rows
            .par_iter()
            .map(|fields| {
                let mut row = Vec::with_capacity(n_out);
                for &src_col in &col_indices {
                    let val_str = if src_col < fields.len() { &fields[src_col] } else { "" };
                    row.push(parse_cell(val_str, col_types[src_col], &na_values)?);
                }
                Ok(row)
            })
            .collect();

        // Transpose row-major → column-major
        let mut col_vecs: Vec<Vec<FlexType>> = (0..n_out)
            .map(|_| Vec::with_capacity(row_results.len()))
            .collect();
        for row_result in row_results {
            let row = row_result?;
            for (i, val) in row.into_iter().enumerate() {
                col_vecs[i].push(val);
            }
        }

        consumer(col_vecs)?;
    }

    Ok(())
}
```

Also add a helper:
```rust
/// Ensure buffer ends with a line terminator (for EOF handling).
fn ensure_line_terminator(buffer: &mut Vec<u8>, config: &ByteConfig) {
    if buffer.is_empty() { return; }
    if config.is_regular_line_terminator {
        let last = *buffer.last().unwrap();
        if last != b'\n' && last != b'\r' {
            buffer.push(b'\n');
        }
    } else if !config.line_terminator.is_empty() {
        let term = &config.line_terminator;
        if buffer.len() < term.len() || &buffer[buffer.len() - term.len()..] != term.as_slice() {
            buffer.extend_from_slice(term);
        }
    }
}
```

**Step 3: Run existing tests — expect PASS**

This is the critical step: all existing CSV tests (csv_compat_tests, sframe
from_csv tests, etc.) must still pass.

```bash
cargo test -p sframe-query -- --nocapture
cargo test -p sframe -- --nocapture
```

**Step 4: Fix any failures**

Most likely failure points:
- Comment line detection (currently done on raw text, need to handle on rows)
- Skip-rows counting (off by one)
- Row-limit boundary (need to break out of parallel results)
- Empty row handling

Debug by adding `eprintln!` traces and comparing parallel vs sequential output.

**Step 5: Commit**

```bash
git add crates/sframe-query/src/algorithms/csv_parser.rs crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs
git commit -m "feat: integrate parallel CSV tokenizer into CsvStreamingParse"
```

---

### Task 6: End-to-End Correctness Tests

**Files:**
- Modify: `crates/sframe-query/src/algorithms/csv_parallel_tokenizer.rs` (tests)
- Modify: `crates/sframe-query/src/algorithms/csv_compat_tests.rs` (optional)

**Step 1: Write large-scale correctness test**

Generate a large CSV with known content, parse it with both old and new paths,
compare results:

```rust
#[test]
fn test_parallel_large_csv() {
    use std::io::Write;
    let tmp = std::env::temp_dir().join("test_parallel_large.csv");
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "id,name,value").unwrap();
        for i in 0..100_000 {
            writeln!(f, "{},\"item_{}\",{:.2}", i, i, i as f64 * 0.1).unwrap();
        }
    }
    let sf = SFrame::from_csv(tmp.to_str().unwrap(), None).unwrap();
    assert_eq!(sf.num_rows().unwrap(), 100_000);
    // Verify first and last rows
    // ...
    std::fs::remove_file(&tmp).ok();
}
```

**Step 2: Write multi-line quoted field end-to-end test**

```rust
#[test]
fn test_parallel_multiline_csv() {
    use std::io::Write;
    let tmp = std::env::temp_dir().join("test_parallel_multiline.csv");
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "id,review").unwrap();
        writeln!(f, "1,\"good food\"").unwrap();
        writeln!(f, "2,\"hello").unwrap();
        writeln!(f, "world\"").unwrap();
        writeln!(f, "3,\"moof\"").unwrap();
    }
    let sf = SFrame::from_csv(tmp.to_str().unwrap(), None).unwrap();
    assert_eq!(sf.num_rows().unwrap(), 3);
    // Verify row 2 has "hello\nworld"
    // ...
    std::fs::remove_file(&tmp).ok();
}
```

**Step 3: Run all tests**

```bash
cargo test --workspace -- --nocapture
```

**Step 4: Commit**

```bash
git add -A
git commit -m "test: end-to-end parallel CSV tokenizer tests"
```

---

### Task 7: Benchmark and Performance Validation

**Files:**
- Use existing: `crates/sframe/examples/benchmark.rs`

**Step 1: Run benchmark before optimization (baseline already known)**

```bash
cargo run -p sframe --release --example benchmark 2>&1 | grep -i csv
```

**Step 2: Run benchmark after optimization**

```bash
cargo run -p sframe --release --example benchmark 2>&1 | grep -i csv
```

**Step 3: Profile if needed**

If performance is not sufficiently improved, profile with:
```bash
cargo flamegraph -p sframe --example benchmark
```

Common optimization targets:
- `split_fields_bytes` inner loop (ensure compiler vectorizes)
- String allocation in `finish_field_bytes` (consider `Cow<str>` for unquoted fields)
- `find_true_newline_positions` fast inner loop (verify no bounds checks)

**Step 4: Commit any perf fixes**

```bash
git add -A
git commit -m "perf: parallel CSV tokenizer — quote-parity bitset + byte-level ops"
```
