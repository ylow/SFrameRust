//! Parallel CSV tokenizer using quote-parity bitset for concurrent line splitting.
//!
//! The key insight: to split a CSV buffer into lines in parallel, each thread
//! needs to know whether a given newline is "real" (a row boundary) or "quoted"
//! (inside a quoted field). The [`DenseBitset`] tracks quote parity — bit *i*
//! is set when byte position *i* is inside a quoted field — so that parallel
//! workers can classify newlines without scanning from the start of the buffer.

use super::csv_tokenizer::CsvConfig;
use sframe_types::error::{Result, SFrameError};

/// Compact bitset -- 1 bit per byte position in a CSV buffer.
/// Used to track quote parity: bit set means "inside a quoted field".
pub(crate) struct DenseBitset {
    bits: Vec<u64>,
    /// Number of addressable bit positions.
    len: usize,
}

impl DenseBitset {
    /// Allocate a zeroed bitset that can hold `len` bits.
    pub fn new(len: usize) -> Self {
        let num_words = (len + 63) / 64;
        DenseBitset {
            bits: vec![0u64; num_words],
            len,
        }
    }

    /// Set the bit at `idx`.
    #[inline(always)]
    pub fn set(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    /// Clear the bit at `idx`.
    #[inline(always)]
    pub fn clear_bit(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] &= !(1u64 << bit);
    }

    /// Test whether the bit at `idx` is set.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len);
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] >> bit) & 1 != 0
    }

    /// Set the bit at `idx` to `val`.
    pub fn set_to(&mut self, idx: usize, val: bool) {
        debug_assert!(idx < self.len);
        if val {
            self.set(idx);
        } else {
            self.clear_bit(idx);
        }
    }

    /// Zero all bits, keeping the current allocation and length.
    pub fn clear_all(&mut self) {
        self.bits.fill(0);
    }

    /// Resize to `len` bits and zero everything.
    /// Reuses the existing heap allocation when possible.
    pub fn resize_and_clear(&mut self, len: usize) {
        let num_words = (len + 63) / 64;
        self.bits.clear();
        self.bits.resize(num_words, 0);
        self.len = len;
    }
}

// ---------------------------------------------------------------------------
// ByteConfig
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Newline advancement helpers
// ---------------------------------------------------------------------------

/// Advance past the next newline starting at `pos`. Returns (new_pos, found).
/// For standard terminators (\n, \r, \r\n), handles all three.
#[inline]
fn advance_past_newline(
    buf: &[u8],
    mut pos: usize,
    end: usize,
    config: &ByteConfig,
) -> (usize, bool) {
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

/// Like [`advance_past_newline`] but only accepts newlines where
/// the quote_parity bit is NOT set (i.e., real line boundaries).
fn advance_past_newline_with_parity(
    buf: &[u8],
    mut pos: usize,
    end: usize,
    config: &ByteConfig,
    parity: &DenseBitset,
) -> (usize, bool) {
    while pos < end {
        let (next, matched) = advance_past_newline(buf, pos, end, config);
        if !matched {
            return (next, false);
        }
        // Check parity of the byte just before next (last byte of newline).
        // If parity set, this newline is inside quotes — skip it.
        if parity.get(next - 1) {
            pos = next;
            continue;
        }
        return (next, true);
    }
    (pos, false)
}

// ---------------------------------------------------------------------------
// Quote-parity scanner
// ---------------------------------------------------------------------------

/// Build a bitset marking which byte positions are inside quoted fields.
///
/// After this, a newline at position `i` is a *real* line boundary iff
/// `!bitset.get(i)`. This enables parallel line splitting.
///
/// Ported from C++ `find_true_new_line_positions()` in parallel_csv_parser.cpp.
///
/// The algorithm scans the buffer byte-by-byte tracking quote state:
/// - If a byte is the quote char (not escaped), toggle `cur_in_quote`
/// - If `cur_in_quote`, mark the bit in the bitset
/// - If not in quote and the byte is a comment char, skip to next newline
/// - Fast inner loop: skip bytes that aren't quote, escape, or comment chars
pub(crate) fn find_true_newline_positions(buffer: &[u8], config: &ByteConfig) -> DenseBitset {
    let len = buffer.len();
    let mut parity = DenseBitset::new(len);

    if len == 0 {
        return parity;
    }

    let quote = config.quote;
    let escape = config.escape;

    if let Some(comment) = config.comment {
        // Variant with comment char: also check for comment_char, skip to
        // next newline when found outside quotes.
        scan_with_comment(buffer, len, quote, escape, comment, config, &mut parity);
    } else {
        // Simpler variant: only check quote and escape.
        scan_no_comment(buffer, len, quote, escape, &mut parity);
    }

    parity
}

/// Inner scan loop when there is no comment character.
fn scan_no_comment(
    buf: &[u8],
    len: usize,
    quote: u8,
    escape: u8,
    parity: &mut DenseBitset,
) {
    let mut pos = 0;
    let mut cur_in_quote = false;
    let mut not_esc = true;

    while pos < len {
        // Fast inner loop: skip non-special bytes.
        let start = pos;
        while pos < len && buf[pos] != quote && buf[pos] != escape {
            pos += 1;
        }
        // If cur_in_quote, set bits for all skipped positions.
        if cur_in_quote {
            for i in start..pos {
                parity.set(i);
            }
        }
        // Any non-escape byte resets not_esc to true. If the fast loop
        // advanced past at least one byte, those bytes are all non-special
        // (not escape, not quote), so not_esc must be true.
        if pos > start {
            not_esc = true;
        }
        // pos now either == len or points at a quote or escape byte.

        if pos >= len {
            break;
        }

        let c = buf[pos];
        let is_quote_char = (c == quote) && not_esc;
        cur_in_quote ^= is_quote_char;
        if cur_in_quote {
            parity.set(pos);
        }
        // Update not_esc: if c is escape, toggle not_esc; otherwise reset to true.
        not_esc = !not_esc || c != escape;
        pos += 1;
    }
}

/// Inner scan loop when a comment character is configured.
fn scan_with_comment(
    buf: &[u8],
    len: usize,
    quote: u8,
    escape: u8,
    comment: u8,
    config: &ByteConfig,
    parity: &mut DenseBitset,
) {
    let mut pos = 0;
    let mut cur_in_quote = false;
    let mut not_esc = true;

    while pos < len {
        // Fast inner loop: skip non-special bytes.
        let start = pos;
        while pos < len && buf[pos] != quote && buf[pos] != escape && buf[pos] != comment {
            pos += 1;
        }
        // If cur_in_quote, set bits for all skipped positions.
        if cur_in_quote {
            for i in start..pos {
                parity.set(i);
            }
        }
        // Any non-escape byte resets not_esc to true. If the fast loop
        // advanced past at least one byte, those bytes are all non-special,
        // so not_esc must be true.
        if pos > start {
            not_esc = true;
        }

        if pos >= len {
            break;
        }

        let c = buf[pos];

        // Handle comment outside quotes: skip to next newline.
        if c == comment && !cur_in_quote && not_esc {
            let (next_pos, _found) = advance_past_newline(buf, pos, len, config);
            // Comment bytes and the newline itself are not inside quotes,
            // so no bits to set. Just advance.
            pos = next_pos;
            // Reset not_esc since we jumped past the comment line.
            not_esc = true;
            continue;
        }

        let is_quote_char = (c == quote) && not_esc;
        cur_in_quote ^= is_quote_char;
        if cur_in_quote {
            parity.set(pos);
        }
        not_esc = !not_esc || c != escape;
        pos += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_basic() {
        let mut bs = DenseBitset::new(256);

        // All bits start as 0.
        for i in [0, 1, 63, 64, 65, 127, 128, 199, 255] {
            assert!(!bs.get(i), "bit {i} should start unset");
        }

        // Set specific positions including word boundaries.
        for i in [0, 63, 64, 199] {
            bs.set(i);
        }

        // Only the set positions should read as true.
        assert!(bs.get(0));
        assert!(bs.get(63));
        assert!(bs.get(64));
        assert!(bs.get(199));

        // Neighbors should still be unset.
        assert!(!bs.get(1));
        assert!(!bs.get(62));
        assert!(!bs.get(65));
        assert!(!bs.get(198));
        assert!(!bs.get(200));
    }

    #[test]
    fn test_bitset_clear() {
        let mut bs = DenseBitset::new(128);

        bs.set(10);
        bs.set(64);
        assert!(bs.get(10));
        assert!(bs.get(64));

        bs.clear_bit(10);
        assert!(!bs.get(10));
        assert!(bs.get(64)); // other bit unaffected

        bs.clear_bit(64);
        assert!(!bs.get(64));
    }

    #[test]
    fn test_bitset_clear_all() {
        let mut bs = DenseBitset::new(200);

        // Set a spread of bits across multiple words.
        for i in [0, 33, 64, 100, 127, 150, 199] {
            bs.set(i);
        }
        // Sanity: they are set.
        for i in [0, 33, 64, 100, 127, 150, 199] {
            assert!(bs.get(i));
        }

        bs.clear_all();

        // Every bit should be zero now.
        for i in 0..200 {
            assert!(!bs.get(i), "bit {i} should be cleared");
        }
    }

    #[test]
    fn test_bitset_set_to() {
        let mut bs = DenseBitset::new(128);

        // set_to true
        bs.set_to(42, true);
        assert!(bs.get(42));

        // set_to false
        bs.set_to(42, false);
        assert!(!bs.get(42));

        // set_to true then true again (idempotent)
        bs.set_to(99, true);
        bs.set_to(99, true);
        assert!(bs.get(99));

        // set_to false on an already-unset bit (idempotent)
        bs.set_to(10, false);
        assert!(!bs.get(10));
    }

    // -----------------------------------------------------------------------
    // ByteConfig + quote-parity scanner tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quote_parity_no_quotes() {
        let buf = b"hello,world\n1,2\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        for i in 0..buf.len() {
            assert!(!bp.get(i), "bit {} should not be set", i);
        }
    }

    #[test]
    fn test_quote_parity_simple_quote() {
        let buf = b"\"hi\"\n";
        // byte 0 = " -> parity ON
        // byte 1 = h -> in quote
        // byte 2 = i -> in quote
        // byte 3 = " -> parity OFF
        // byte 4 = \n -> not in quote
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        assert!(bp.get(0));  // " (parity flips to true)
        assert!(bp.get(1));  // h
        assert!(bp.get(2));  // i
        assert!(!bp.get(3)); // " (parity flips to false)
        assert!(!bp.get(4)); // \n
    }

    #[test]
    fn test_quote_parity_multiline() {
        let buf = b"\"hello\nworld\"\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // The \n at position 6 is inside quotes -> bit set
        assert!(bp.get(6), "quoted newline should have parity set");
        // The \n at position 13 is outside quotes -> bit not set
        assert!(!bp.get(13), "unquoted newline should not have parity set");
    }

    #[test]
    fn test_quote_parity_escaped_quote() {
        // "say \"hi\""\n -- escape prevents quote from toggling parity
        // Bytes: " s a y   \ " h i \ " " \n
        let buf = b"\"say \\\"hi\\\"\"\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // Final \n should be outside quotes
        assert!(
            !bp.get(buf.len() - 1),
            "final newline should not have parity set"
        );
    }

    #[test]
    fn test_quote_parity_double_quote() {
        // ""hi"" is two adjacent quotes -- parity flips and unflips
        // Actually double_quote doesn't affect parity scan (it self-cancels)
        let buf = b"\"\"hi\"\"\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        assert!(
            !bp.get(buf.len() - 1),
            "final newline should not have parity set"
        );
    }

    #[test]
    fn test_quote_parity_comment() {
        let buf = b"# this is a comment\n1,2\n";
        let mut config = CsvConfig::default();
        config.comment_char = Some('#');
        let cfg = ByteConfig::from_config(&config).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // Nothing should be in-quote (comment skips to newline)
        for i in 0..buf.len() {
            assert!(!bp.get(i), "bit {} should not be set", i);
        }
    }

    #[test]
    fn test_quote_parity_escape_then_nonspecial_then_quote() {
        // Verify that the fast inner loop correctly resets not_esc.
        // Buffer: \x "hi" \n
        // The \x is escape + non-special byte. After the fast loop skips 'x',
        // not_esc must be true so the next " is recognized as a real quote.
        let buf = b"\\x\"hi\"\n";
        let mut config = CsvConfig::default();
        config.comment_char = None; // use no-comment variant
        let cfg = ByteConfig::from_config(&config).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // Bytes: \ x " h i " \n
        // pos 0: \ (escape), not_esc was true -> not_esc = false
        // pos 1: fast loop: 'x' is not quote, not escape -> skip, pos=2
        //         fast loop: '"' is quote -> stop. pos=2, start=1, skipped 1 byte
        //         not_esc reset to true
        // pos 2: " (quote), not_esc=true -> is_quote_char=true, cur_in_quote=true
        //         set bit 2
        // pos 3-4: h, i -> in quote, bits set
        // pos 5: " (quote), cur_in_quote flips to false, bit not set
        // pos 6: \n, not in quote, bit not set
        assert!(!bp.get(0)); // \ not in quote
        assert!(!bp.get(1)); // x not in quote (escape seq consumed)
        assert!(bp.get(2));  // " opens quote
        assert!(bp.get(3));  // h in quote
        assert!(bp.get(4));  // i in quote
        assert!(!bp.get(5)); // " closes quote
        assert!(!bp.get(6)); // \n not in quote
    }

    #[test]
    fn test_advance_past_newline_basic() {
        let buf = b"hello\nworld\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let (pos, found) = advance_past_newline(buf, 0, buf.len(), &cfg);
        assert!(found);
        assert_eq!(pos, 6); // past the \n
        let (pos2, found2) = advance_past_newline(buf, 6, buf.len(), &cfg);
        assert!(found2);
        assert_eq!(pos2, 12);
    }

    #[test]
    fn test_advance_past_newline_crlf() {
        let buf = b"hello\r\nworld\r\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let (pos, found) = advance_past_newline(buf, 0, buf.len(), &cfg);
        assert!(found);
        assert_eq!(pos, 7); // past \r\n
    }

    #[test]
    fn test_advance_with_parity_skips_quoted_newline() {
        let buf = b"\"hello\nworld\"\n";
        let cfg = ByteConfig::from_config(&CsvConfig::default()).unwrap();
        let bp = find_true_newline_positions(buf, &cfg);
        // advance_past_newline_with_parity should skip the \n at pos 6 (in quote)
        // and find the \n at pos 13
        let (pos, found) =
            advance_past_newline_with_parity(buf, 0, buf.len(), &cfg, &bp);
        assert!(found);
        assert_eq!(pos, 14); // past the real \n at position 13
    }
}
