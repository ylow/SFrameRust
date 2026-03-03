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

// ---------------------------------------------------------------------------
// Byte-level field splitter
// ---------------------------------------------------------------------------

/// Scan ahead from `start` to find a matching close bracket/brace on `&[u8]`.
/// Returns the index of the closing bracket, or None if unbalanced.
/// Respects quote nesting inside brackets. Also verifies that the closing
/// bracket is followed by delimiter, end-of-line, or whitespace (not mid-data).
fn find_balanced_close_bytes(
    buf: &[u8],
    start: usize,
    open: u8,
    close: u8,
    delim: &[u8],
    skip_initial_space: bool,
) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_q = false;
    let mut esc = false;
    let mut i = start;
    while i < buf.len() {
        if esc {
            esc = false;
            i += 1;
            continue;
        }
        let ch = buf[i];
        if ch == b'\\' && in_q {
            esc = true;
            i += 1;
            continue;
        }
        if ch == b'"' {
            in_q = !in_q;
        } else if !in_q {
            if ch == open {
                depth += 1;
            } else if ch == close {
                depth -= 1;
                if depth == 0 {
                    let after = i + 1;
                    if after >= buf.len() {
                        return Some(i); // end of line -- valid
                    }
                    // Check for delimiter BEFORE skipping whitespace
                    if !delim.is_empty()
                        && after + delim.len() <= buf.len()
                        && buf[after..after + delim.len()] == *delim
                    {
                        return Some(i);
                    }
                    // For space-based delimiters, any whitespace counts
                    if !delim.is_empty()
                        && delim.iter().all(|&c| (c as char).is_whitespace())
                        && (buf[after] as char).is_whitespace()
                    {
                        return Some(i);
                    }
                    // Skip optional whitespace, then check for delimiter
                    let mut check = after;
                    if skip_initial_space {
                        while check < buf.len() && buf[check] == b' ' {
                            check += 1;
                        }
                    }
                    if check >= buf.len() {
                        return Some(i); // only whitespace left -- valid
                    }
                    if !delim.is_empty()
                        && check + delim.len() <= buf.len()
                        && buf[check..check + delim.len()] == *delim
                    {
                        return Some(i);
                    }
                    // Not followed by delimiter -- not a valid bracketed field.
                    return None;
                }
            }
        }
        i += 1;
    }
    None
}

/// Check if a byte slice contains a `:` at bracket depth 0 outside quotes.
/// Used to validate `{...}` content looks like a dict before committing to
/// bracket lookahead.
fn has_colon_at_depth0_bytes(buf: &[u8]) -> bool {
    let mut depth = 0i32;
    let mut in_q = false;
    let mut esc = false;
    for &ch in buf {
        if esc {
            esc = false;
            continue;
        }
        if ch == b'\\' && in_q {
            esc = true;
            continue;
        }
        if ch == b'"' {
            in_q = !in_q;
        } else if !in_q {
            match ch {
                b'[' | b'{' => depth += 1,
                b']' | b'}' => depth -= 1,
                b':' if depth == 0 => return true,
                _ => {}
            }
        }
    }
    false
}

/// Skip delimiter and whitespace after a bracket-consumed field (byte-level).
///
/// Checks for delimiter BEFORE skipping spaces (the delimiter might BE spaces).
/// Then skips any remaining spaces per skip_initial_space.
fn skip_post_bracket_bytes(
    buf: &[u8],
    i: &mut usize,
    delim: &[u8],
    config: &ByteConfig,
    had_delimiter: &mut bool,
) {
    if *i >= buf.len() {
        return;
    }

    // Check for delimiter first (it might be whitespace)
    if !delim.is_empty()
        && *i + delim.len() <= buf.len()
        && buf[*i..*i + delim.len()] == *delim
    {
        *had_delimiter = true;
        *i += delim.len();
        if config.skip_initial_space {
            while *i < buf.len() && buf[*i] == b' ' {
                *i += 1;
            }
        }
        return;
    }

    // For space-based delimiters, any whitespace counts as delimiter
    if !delim.is_empty()
        && delim.iter().all(|&c| (c as char).is_whitespace())
        && (buf[*i] as char).is_whitespace()
    {
        *had_delimiter = true;
        *i += 1;
        if config.skip_initial_space {
            while *i < buf.len() && buf[*i] == b' ' {
                *i += 1;
            }
        }
        return;
    }

    // Skip optional whitespace, then check for delimiter
    if config.skip_initial_space {
        while *i < buf.len() && buf[*i] == b' ' {
            *i += 1;
        }
    }
    if *i < buf.len()
        && !delim.is_empty()
        && *i + delim.len() <= buf.len()
        && buf[*i..*i + delim.len()] == *delim
    {
        *had_delimiter = true;
        *i += delim.len();
        if config.skip_initial_space {
            while *i < buf.len() && buf[*i] == b' ' {
                *i += 1;
            }
        }
    }
}

/// Process C-style escape sequences in a byte slice.
/// Handles: \n, \t, \r, \\, \", \', \/, \b, \f, \uXXXX (including surrogate pairs).
/// Outputs UTF-8 bytes.
fn unescape_csv_bytes(s: &[u8], escape: u8) -> Vec<u8> {
    let mut result = Vec::with_capacity(s.len());
    let mut i = 0;

    while i < s.len() {
        if s[i] == escape {
            i += 1;
            if i >= s.len() {
                result.push(escape);
                break;
            }
            match s[i] {
                b'n' => { result.push(b'\n'); i += 1; }
                b't' => { result.push(b'\t'); i += 1; }
                b'r' => { result.push(b'\r'); i += 1; }
                b'\\' => { result.push(b'\\'); i += 1; }
                b'"' => { result.push(b'"'); i += 1; }
                b'\'' => { result.push(b'\''); i += 1; }
                b'/' => { result.push(b'/'); i += 1; }
                b'b' => { result.push(0x08); i += 1; }
                b'f' => { result.push(0x0C); i += 1; }
                b'u' => {
                    i += 1; // skip 'u'
                    // Need 4 hex digits
                    if i + 4 <= s.len() {
                        let hex = &s[i..i + 4];
                        if let Some(cp) = parse_hex4(hex) {
                            if (0xD800..=0xDBFF).contains(&cp) {
                                // High surrogate -- look for low surrogate
                                if i + 4 + 2 + 4 <= s.len()
                                    && s[i + 4] == escape
                                    && s[i + 5] == b'u'
                                {
                                    if let Some(cp2) = parse_hex4(&s[i + 6..i + 10]) {
                                        if (0xDC00..=0xDFFF).contains(&cp2) {
                                            let combined = 0x10000
                                                + ((cp - 0xD800) << 10)
                                                + (cp2 - 0xDC00);
                                            if let Some(c) = char::from_u32(combined) {
                                                let mut buf = [0u8; 4];
                                                let encoded = c.encode_utf8(&mut buf);
                                                result.extend_from_slice(encoded.as_bytes());
                                                i += 10; // 4 + escape + u + 4
                                                continue;
                                            }
                                        }
                                    }
                                }
                                // Bad surrogate pair -- keep literal \uXXXX
                                result.push(escape);
                                result.push(b'u');
                                result.extend_from_slice(hex);
                                i += 4;
                            } else if let Some(c) = char::from_u32(cp) {
                                let mut buf = [0u8; 4];
                                let encoded = c.encode_utf8(&mut buf);
                                result.extend_from_slice(encoded.as_bytes());
                                i += 4;
                            } else {
                                // Invalid codepoint: replacement char
                                let mut buf = [0u8; 4];
                                let encoded = '\u{FFFD}'.encode_utf8(&mut buf);
                                result.extend_from_slice(encoded.as_bytes());
                                i += 4;
                            }
                        } else {
                            // Not valid hex
                            result.push(escape);
                            result.push(b'u');
                            result.extend_from_slice(hex);
                            i += 4;
                        }
                    } else {
                        // Not enough digits
                        let remaining = &s[i..];
                        result.push(escape);
                        result.push(b'u');
                        result.extend_from_slice(remaining);
                        i = s.len();
                    }
                }
                other => {
                    result.push(escape);
                    result.push(other);
                    i += 1;
                }
            }
        } else {
            result.push(s[i]);
            i += 1;
        }
    }

    result
}

/// Parse exactly 4 ASCII hex digits into a u32.
#[inline]
fn parse_hex4(hex: &[u8]) -> Option<u32> {
    if hex.len() < 4 {
        return None;
    }
    let mut val = 0u32;
    for &b in &hex[..4] {
        let digit = match b {
            b'0'..=b'9' => (b - b'0') as u32,
            b'a'..=b'f' => (b - b'a' + 10) as u32,
            b'A'..=b'F' => (b - b'A' + 10) as u32,
            _ => return None,
        };
        val = (val << 4) | digit;
    }
    Some(val)
}

/// Process a raw field value on bytes: trim, unquote, unescape, handle double-quote.
fn finish_field_bytes(raw: &[u8], config: &ByteConfig) -> String {
    // Trim ASCII whitespace
    let trimmed = trim_ascii(raw);

    let q = config.quote;
    if trimmed.len() >= 2
        && trimmed[0] == q
        && trimmed[trimmed.len() - 1] == q
    {
        let inner = &trimmed[1..trimmed.len() - 1];

        // Handle double-quote escaping
        let unquoted = if config.double_quote {
            let double_q = [q, q];
            replace_bytes(inner, &double_q, &[q])
        } else {
            inner.to_vec()
        };

        // Process escape sequences
        let unescaped = unescape_csv_bytes(&unquoted, config.escape);
        String::from_utf8(unescaped).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    } else {
        // Not quoted -- just convert trimmed bytes to string
        String::from_utf8(trimmed.to_vec()).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }
}

/// Trim leading and trailing ASCII whitespace from a byte slice.
#[inline]
fn trim_ascii(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|&b| !b.is_ascii_whitespace()).unwrap_or(s.len());
    let end = s.iter().rposition(|&b| !b.is_ascii_whitespace()).map_or(start, |p| p + 1);
    &s[start..end]
}

/// Replace all occurrences of `needle` in `haystack` with `replacement`.
fn replace_bytes(haystack: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    if needle.is_empty() {
        return haystack.to_vec();
    }
    let mut result = Vec::with_capacity(haystack.len());
    let mut i = 0;
    while i < haystack.len() {
        if i + needle.len() <= haystack.len() && &haystack[i..i + needle.len()] == needle {
            result.extend_from_slice(replacement);
            i += needle.len();
        } else {
            result.push(haystack[i]);
            i += 1;
        }
    }
    result
}

/// Split a line (as bytes) into fields, respecting quotes and bracket/brace nesting.
///
/// This is the byte-level equivalent of `split_fields` in `csv_tokenizer.rs`.
/// It produces IDENTICAL output for all inputs.
pub(crate) fn split_fields_bytes(line: &[u8], config: &ByteConfig) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = Vec::new();
    let delim = &config.delimiter;
    let mut i = 0;
    let mut in_quote = false;
    let mut escape_next = false;
    let mut in_field = false;
    let mut had_delimiter = false;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;

    // Skip leading whitespace before first field
    if config.skip_initial_space {
        while i < line.len() && line[i] == b' ' {
            i += 1;
        }
    }

    while i < line.len() {
        if escape_next {
            current.push(line[i]);
            escape_next = false;
            i += 1;
            continue;
        }

        let ch = line[i];

        // Handle escape in quoted context
        if ch == config.escape && in_quote {
            current.push(ch);
            escape_next = true;
            i += 1;
            continue;
        }

        // Handle quote character
        if ch == config.quote {
            if in_quote {
                // Check for double-quote escape
                if config.double_quote
                    && i + 1 < line.len()
                    && line[i + 1] == config.quote
                {
                    current.push(ch);
                    current.push(line[i + 1]);
                    i += 2;
                    continue;
                }
                // End of quoted field
                current.push(ch);
                in_quote = false;
                i += 1;
                continue;
            } else if current.is_empty()
                || current.iter().all(|&b| (b as char).is_whitespace())
            {
                // Start of quoted field
                current.push(ch);
                in_quote = true;
                i += 1;
                continue;
            }
        }

        if !in_quote {
            // Inline comment: terminates current field AND line
            if let Some(cc) = config.comment {
                if ch == cc && bracket_depth == 0 && brace_depth == 0 {
                    fields.push(finish_field_bytes(&current, config));
                    return fields;
                }
            }

            // Bracket/brace lookahead at field start
            let at_field_start = current.is_empty()
                || current.iter().all(|&b| (b as char).is_whitespace());

            if at_field_start && bracket_depth == 0 && brace_depth == 0 {
                if ch == b'[' {
                    if let Some(close_pos) = find_balanced_close_bytes(line, i, b'[', b']', delim, config.skip_initial_space) {
                        let field_bytes = &line[i..=close_pos];
                        fields.push(finish_field_bytes(field_bytes, config));
                        current.clear();
                        in_field = false;
                        had_delimiter = false;
                        i = close_pos + 1;
                        skip_post_bracket_bytes(line, &mut i, delim, config, &mut had_delimiter);
                        continue;
                    }
                } else if ch == b'{' {
                    if let Some(close_pos) = find_balanced_close_bytes(line, i, b'{', b'}', delim, config.skip_initial_space) {
                        let inner = &line[i + 1..close_pos];
                        if has_colon_at_depth0_bytes(inner) {
                            let field_bytes = &line[i..=close_pos];
                            fields.push(finish_field_bytes(field_bytes, config));
                            current.clear();
                            in_field = false;
                            had_delimiter = false;
                            i = close_pos + 1;
                            skip_post_bracket_bytes(line, &mut i, delim, config, &mut had_delimiter);
                            continue;
                        }
                    }
                }
            }

            // Inside an already-open bracket group
            if bracket_depth > 0 || brace_depth > 0 {
                match ch {
                    b'[' => bracket_depth += 1,
                    b']' => bracket_depth = (bracket_depth - 1).max(0),
                    b'{' => brace_depth += 1,
                    b'}' => brace_depth = (brace_depth - 1).max(0),
                    _ => {}
                }
            }

            // Check for delimiter (only at top level)
            if bracket_depth == 0 && brace_depth == 0 && !delim.is_empty() {
                if i + delim.len() <= line.len()
                    && line[i..i + delim.len()] == *delim
                {
                    fields.push(finish_field_bytes(&current, config));
                    current.clear();
                    in_field = false;
                    had_delimiter = true;
                    i += delim.len();
                    if config.skip_initial_space {
                        while i < line.len() && line[i] == b' ' {
                            i += 1;
                        }
                    }
                    continue;
                }
            }
        }

        current.push(ch);
        in_field = true;
        i += 1;
    }

    // Push trailing field
    if in_field || had_delimiter {
        fields.push(finish_field_bytes(&current, config));
    }
    fields
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

    // -----------------------------------------------------------------------
    // Byte-level field splitter parity tests
    // -----------------------------------------------------------------------

    use super::super::csv_tokenizer;

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
        assert_fields_match("", &config);
    }

    #[test]
    fn test_byte_split_quoted() {
        let config = CsvConfig::default();
        assert_fields_match("\"hello, world\",42", &config);
        assert_fields_match("\"say \"\"hi\"\"\",bar", &config);
        assert_fields_match("\"\"", &config);
    }

    #[test]
    fn test_byte_split_escaped() {
        let config = CsvConfig::default();
        assert_fields_match("\"hello\\nworld\",42", &config);
        assert_fields_match("\"tab\\there\"", &config);
        assert_fields_match("\"unicode\\u0041\"", &config);
    }

    #[test]
    fn test_byte_split_brackets() {
        let config = CsvConfig::default();
        assert_fields_match("foo,[1,2,3]", &config);
        assert_fields_match("foo,{a:1,b:2}", &config);
        assert_fields_match("test,[1,[2,3],4]", &config);
        assert_fields_match("x,{}", &config);  // not a dict (no colon)
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
        assert_fields_match("a,,b", &config);
    }

    #[test]
    fn test_byte_split_multichar_delimiter() {
        let config = CsvConfig {
            delimiter: "::".to_string(),
            ..Default::default()
        };
        assert_fields_match("a::b::c", &config);
    }

    #[test]
    fn test_byte_split_space_delimiter() {
        let config = CsvConfig {
            delimiter: " ".to_string(),
            skip_initial_space: false,
            ..Default::default()
        };
        assert_fields_match("a b c", &config);
    }

    #[test]
    fn test_byte_split_unicode_content() {
        let config = CsvConfig::default();
        assert_fields_match("hello,\u{4e16}\u{754c},foo", &config);
        assert_fields_match("\"\u{65e5}\u{672c}\u{8a9e}\",42", &config);
    }

    // -----------------------------------------------------------------------
    // Exhaustive parity tests from csv_compat_tests.rs patterns
    // -----------------------------------------------------------------------

    /// Escape-string helper matching C++ default_escape_string().
    fn esc(s: &str) -> String {
        let inner = s
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        format!("\"{}\"", inner)
    }

    #[test]
    fn test_byte_split_parity_basic_delimiters() {
        // Test with various delimiters like csv_compat_tests
        for dlm in &[",", " ", ";", "::", "  ", "\t\t", "\t"] {
            let config = CsvConfig {
                delimiter: dlm.to_string(),
                ..Default::default()
            };
            let d = *dlm;
            // basic CSV line
            let line = format!(
                "1.1{d}1{d}one{d}[1,1,1]{d}{{1:1,\"a\":\"a\"}}{d}[a,a]"
            );
            assert_fields_match(&line, &config);
        }
    }

    #[test]
    fn test_byte_split_parity_quoted_basic() {
        for dlm in &[",", " ", ";", "::", "\t"] {
            let config = CsvConfig {
                delimiter: dlm.to_string(),
                ..Default::default()
            };
            let d = *dlm;
            let line = format!(
                "{}{d}{}{d}{}{d}{}{d}{}{d}{}",
                esc("1.1"),
                esc("1"),
                esc("one"),
                esc("[1,1,1]"),
                esc("{1:1,\"a\":\"a\"}"),
                esc("[a,a]"),
            );
            assert_fields_match(&line, &config);
        }
    }

    #[test]
    fn test_byte_split_parity_embedded_strings() {
        for dlm in &[",", " ", "\t", "\t\t", "  ", "::"] {
            let config = CsvConfig {
                delimiter: dlm.to_string(),
                ..Default::default()
            };
            let d = *dlm;
            // Unquoted brackets that don't balance
            assert_fields_match(&format!("[abc{d}[1,1,1]"), &config);
            assert_fields_match(&format!("cde]{d}[2,2,2]"), &config);
            assert_fields_match(&format!("a[a]b{d}[3,3,3]"), &config);
            // Quoted brackets
            assert_fields_match(&format!("\"[abc\"{d}[1,1,1]"), &config);
            assert_fields_match(&format!("\"cde]\"{d}[2,2,2]"), &config);
            assert_fields_match(&format!("\"a[a]b\"{d}[3,3,3]"), &config);
        }
    }

    #[test]
    fn test_byte_split_parity_interesting() {
        // From the interesting() test in csv_compat_tests
        let config = CsvConfig {
            delimiter: ";".to_string(),
            double_quote: true,
            ..Default::default()
        };
        assert_fields_match("1.1 ;1;[1 2 3];\"hello\\\\\"", &config);
        assert_fields_match("2.2;2; [4 5 6];\"wor;ld\"", &config);
        assert_fields_match(" 3.3; 3;[9 2];\"\"\"w\"\"\"", &config);
        assert_fields_match("Pokemon  ;;; NA ", &config);
    }

    #[test]
    fn test_byte_split_parity_excess_whitespace() {
        // From excess_white_space() test
        let config = CsvConfig {
            delimiter: " ".to_string(),
            ..Default::default()
        };
        assert_fields_match("float int str  vec    dict rec", &config);
        assert_fields_match("  1.1 1 one   [1,1,1]  {1 : 1 , \"a\"  : \"a\"}    [a,a]", &config);
        assert_fields_match(" 2.2 2 two   [2,2,2] {2:2,\"b\":\"b\"} [b,b]", &config);
        assert_fields_match("3.3 3 three [3,3,3]  {3:3,  \"c\":\"c\"} [c,c]  \t", &config);
    }

    #[test]
    fn test_byte_split_parity_issue_1514() {
        // From another_wierd_bracketing_thing_issue_1514()
        let config = CsvConfig {
            delimiter: "\t".to_string(),
            ..Default::default()
        };
        assert_fields_match("1\t{\t()\t{}\t{}\t(}\t})\t}\tdebugging", &config);
        assert_fields_match("3\t--\t({})\t{()}\t{}\t({\t{)\t}\tdebugging", &config);
    }

    #[test]
    fn test_byte_split_parity_escape_parsing() {
        // From escape_parsing() test
        let config = CsvConfig {
            delimiter: " ".to_string(),
            ..Default::default()
        };
        assert_fields_match("\"\\n\"  \"\\n\"", &config);
        assert_fields_match("\"\\t\"  \"\\0abf\"", &config);
        assert_fields_match("\"\\\"a\"  \"\\\"b\"", &config);
        assert_fields_match("{\"a\":\"\\\"\"} [a,\"b\",\"\\\"c\"]", &config);
    }

    #[test]
    fn test_byte_split_parity_non_escaped() {
        // Unquoted escape sequences stay literal
        let config = CsvConfig {
            delimiter: " ".to_string(),
            ..Default::default()
        };
        assert_fields_match("\\n  \\n", &config);
        assert_fields_match("\\t  \\0abf", &config);
    }

    #[test]
    fn test_byte_split_parity_string_integers() {
        let config = CsvConfig {
            double_quote: true,
            ..Default::default()
        };
        // "1,"""1""""
        assert_fields_match("1,\"\"\"1\"\"\"", &config);
        // 2,"\"2\""
        assert_fields_match("2,\"\\\"2\\\"\"", &config);
        // Simple quoted integers
        assert_fields_match("1,\"1\"", &config);
        assert_fields_match("2,\"2\"", &config);
    }

    #[test]
    fn test_byte_split_parity_single_string_column() {
        let config = CsvConfig {
            delimiter: "\n".to_string(),
            ..Default::default()
        };
        assert_fields_match("\"\"", &config);
        assert_fields_match("{\"a\":\"b\"}", &config);
        assert_fields_match("{\"\":\"\"}", &config);
    }

    #[test]
    fn test_byte_split_parity_unicode_surrogates() {
        let config = CsvConfig {
            delimiter: "\n".to_string(),
            ..Default::default()
        };
        assert_fields_match("{\"good_surrogates\": \"\\uD834\\uDD1E\"}", &config);
        assert_fields_match("{\"bad_surrogates\": \"\\uD834\u{2019}\"}", &config);
        assert_fields_match("{\"bad_surrogates2\": \"\\uD834\" }", &config);
        assert_fields_match("{\"bad_surrogates3\": \"\\uD834\\uDD\" }", &config);
        assert_fields_match("{\"bad_json\": \"\\u442G\" }", &config);
    }

    #[test]
    fn test_byte_split_parity_missing_tab_values() {
        let config = CsvConfig {
            delimiter: "\t".to_string(),
            ..Default::default()
        };
        assert_fields_match("1\t\t  b", &config);
        assert_fields_match("2\t\t", &config);
        assert_fields_match("3\t  c\t d ", &config);
    }

    #[test]
    fn test_byte_split_parity_tab_delimited_list() {
        let config = CsvConfig {
            delimiter: "\t".to_string(),
            ..Default::default()
        };
        assert_fields_match("xxx\t[1,2,3]\t[1,2,3]", &config);
    }
}
