//! Custom CSV tokenizer with bracket-aware field splitting.
//!
//! Unlike the `csv` crate, this tokenizer respects bracket/brace nesting
//! inside fields. This means `[1,2,3]` stays as a single field even with
//! a comma delimiter. It also supports:
//!
//! - Comment lines (`#` prefix)
//! - C-style escape sequences in quoted fields (`\n`, `\t`, `\"`)
//! - Double-quote escaping (`""` → `"`)
//! - Configurable delimiter (including multi-char)
//! - NA value sets (custom strings → Undefined)
//! - Skip rows and row limits
//! - Column subsetting

/// Configuration for the CSV tokenizer.
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// Field delimiter. Default: ","
    pub delimiter: String,
    /// Line terminator. Default: "\n" (also accepts \r\n and \r)
    pub line_terminator: String,
    /// Escape character. Default: '\\'
    pub escape_char: char,
    /// Quote character. Default: '"'
    pub quote_char: char,
    /// Whether to treat doubled quote chars as escaped quote. Default: true
    pub double_quote: bool,
    /// Skip initial whitespace in each field. Default: true
    pub skip_initial_space: bool,
    /// Comment character. Lines starting with this are skipped.
    pub comment_char: Option<char>,
    /// Whether the first non-skipped row is a header. Default: true
    pub has_header: bool,
    /// Strings that should be treated as NA/Undefined.
    pub na_values: Vec<String>,
    /// Number of rows to skip at the beginning.
    pub skip_rows: usize,
    /// Maximum rows to read (None = all).
    pub row_limit: Option<usize>,
    /// Only output these columns (by name). None = all.
    pub output_columns: Option<Vec<String>>,
}

impl Default for CsvConfig {
    fn default() -> Self {
        CsvConfig {
            delimiter: ",".to_string(),
            line_terminator: "\n".to_string(),
            escape_char: '\\',
            quote_char: '"',
            double_quote: true,
            skip_initial_space: true,
            comment_char: Some('#'),
            has_header: true,
            na_values: Vec::new(),
            row_limit: None,
            skip_rows: 0,
            output_columns: None,
        }
    }
}

/// Tokenize a CSV string into rows of fields.
///
/// Returns (header, rows) where header is the first row if `has_header` is true,
/// and rows are the remaining data rows.
pub fn tokenize(content: &str, config: &CsvConfig) -> (Option<Vec<String>>, Vec<Vec<String>>) {
    let lines = split_lines(content, config);

    let mut iter = lines.into_iter();

    // Skip initial rows
    for _ in 0..config.skip_rows {
        iter.next();
    }

    // Skip comment lines and extract header
    let header = if config.has_header {
        loop {
            match iter.next() {
                None => return (Some(Vec::new()), Vec::new()),
                Some(line) => {
                    if is_comment(&line, config) {
                        continue;
                    }
                    let fields = split_fields(&line, config);
                    break Some(fields);
                }
            }
        }
    } else {
        None
    };

    // Read data rows
    let mut rows = Vec::new();
    let limit = config.row_limit.unwrap_or(usize::MAX);

    for line in iter {
        if rows.len() >= limit {
            break;
        }
        if is_comment(&line, config) {
            continue;
        }
        if line.is_empty() || line.trim().is_empty() {
            continue;
        }
        let fields = split_fields(&line, config);
        rows.push(fields);
    }

    (header, rows)
}

/// Split content into lines, respecting the configured line terminator.
/// For standard terminators, handles \r\n, \r, \n correctly.
/// Also handles quoted fields that span multiple lines.
fn split_lines(content: &str, config: &CsvConfig) -> Vec<String> {
    // Empty line terminator → entire content is one "line"
    if config.line_terminator.is_empty() {
        if content.is_empty() {
            return Vec::new();
        }
        return vec![content.to_string()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut escape_next = false;
    let chars: Vec<char> = content.chars().collect();
    let mut i = 0;

    // Standard line terminator handling
    let is_standard_terminator = config.line_terminator == "\n"
        || config.line_terminator == "\r\n"
        || config.line_terminator == "\r";

    while i < chars.len() {
        if escape_next {
            current.push(chars[i]);
            escape_next = false;
            i += 1;
            continue;
        }

        let ch = chars[i];

        if ch == config.escape_char && in_quote {
            current.push(ch);
            escape_next = true;
            i += 1;
            continue;
        }

        if ch == config.quote_char {
            current.push(ch);
            if in_quote && config.double_quote && i + 1 < chars.len() && chars[i + 1] == config.quote_char {
                // Double quote — not end of quoted field
                current.push(chars[i + 1]);
                i += 2;
                continue;
            }
            in_quote = !in_quote;
            i += 1;
            continue;
        }

        if !in_quote {
            if is_standard_terminator {
                if ch == '\r' {
                    // \r or \r\n
                    lines.push(std::mem::take(&mut current));
                    i += 1;
                    if i < chars.len() && chars[i] == '\n' {
                        i += 1; // consume \n after \r
                    }
                    continue;
                }
                if ch == '\n' {
                    lines.push(std::mem::take(&mut current));
                    i += 1;
                    continue;
                }
            } else {
                // Custom line terminator
                let term_chars: Vec<char> = config.line_terminator.chars().collect();
                if i + term_chars.len() <= chars.len()
                    && chars[i..i + term_chars.len()] == term_chars[..]
                {
                    lines.push(std::mem::take(&mut current));
                    i += term_chars.len();
                    continue;
                }
            }
        }

        current.push(ch);
        i += 1;
    }

    if !current.is_empty() {
        lines.push(current);
    }

    lines
}

/// Check if a line is a comment.
pub fn is_comment(line: &str, config: &CsvConfig) -> bool {
    if let Some(comment_char) = config.comment_char {
        let trimmed = if config.skip_initial_space {
            line.trim_start()
        } else {
            line
        };
        trimmed.starts_with(comment_char)
    } else {
        false
    }
}

/// Scan ahead from `start` to find a matching close bracket/brace.
/// Returns the index of the closing bracket, or None if unbalanced.
/// Respects quote nesting inside brackets. Also verifies that the closing
/// bracket is followed by delimiter, end-of-line, or whitespace (not mid-data).
/// The `delim_chars` parameter is used for this validation.
fn find_balanced_close(
    chars: &[char],
    start: usize,
    open: char,
    close: char,
    delim_chars: &[char],
    skip_initial_space: bool,
) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_q = false;
    let mut esc = false;
    let mut i = start;
    while i < chars.len() {
        if esc {
            esc = false;
            i += 1;
            continue;
        }
        let ch = chars[i];
        if ch == '\\' && in_q {
            esc = true;
            i += 1;
            continue;
        }
        if ch == '"' {
            in_q = !in_q;
        } else if !in_q {
            if ch == open {
                depth += 1;
            } else if ch == close {
                depth -= 1;
                if depth == 0 {
                    // Verify: the char after closing bracket must be delimiter,
                    // whitespace, or end-of-line. Otherwise this isn't a real
                    // structured field (e.g. `{}` in `{}\tdebugging`).
                    let after = i + 1;
                    if after >= chars.len() {
                        return Some(i); // end of line — valid
                    }
                    // Check for delimiter BEFORE skipping whitespace (because
                    // the delimiter itself might be whitespace).
                    if !delim_chars.is_empty()
                        && after + delim_chars.len() <= chars.len()
                        && chars[after..after + delim_chars.len()] == *delim_chars
                    {
                        return Some(i); // delimiter follows — valid
                    }
                    // For space-based delimiters, any whitespace counts
                    if !delim_chars.is_empty()
                        && delim_chars.iter().all(|c| c.is_whitespace())
                        && chars[after].is_whitespace()
                    {
                        return Some(i);
                    }
                    // Skip optional whitespace, then check for delimiter
                    let mut check = after;
                    if skip_initial_space {
                        while check < chars.len() && chars[check] == ' ' {
                            check += 1;
                        }
                    }
                    if check >= chars.len() {
                        return Some(i); // only whitespace left — valid
                    }
                    if !delim_chars.is_empty()
                        && check + delim_chars.len() <= chars.len()
                        && chars[check..check + delim_chars.len()] == *delim_chars
                    {
                        return Some(i); // delimiter after whitespace — valid
                    }
                    // Not followed by delimiter — not a valid bracketed field.
                    return None;
                }
            }
        }
        i += 1;
    }
    None
}

/// Skip delimiter and whitespace after a bracket-consumed field.
///
/// Checks for delimiter BEFORE skipping spaces (the delimiter might BE spaces).
/// Then skips any remaining spaces per skip_initial_space.
fn skip_post_bracket(
    chars: &[char],
    i: &mut usize,
    delim_chars: &[char],
    config: &CsvConfig,
    had_delimiter: &mut bool,
) {
    if *i >= chars.len() {
        return;
    }

    // Check for delimiter first (it might be whitespace)
    if !delim_chars.is_empty()
        && *i + delim_chars.len() <= chars.len()
        && chars[*i..*i + delim_chars.len()] == *delim_chars
    {
        *had_delimiter = true;
        *i += delim_chars.len();
        // Skip additional spaces after delimiter
        if config.skip_initial_space {
            while *i < chars.len() && chars[*i] == ' ' {
                *i += 1;
            }
        }
        return;
    }

    // For space-based delimiters, any whitespace counts as delimiter
    if !delim_chars.is_empty()
        && delim_chars.iter().all(|c| c.is_whitespace())
        && chars[*i].is_whitespace()
    {
        *had_delimiter = true;
        *i += 1;
        if config.skip_initial_space {
            while *i < chars.len() && chars[*i] == ' ' {
                *i += 1;
            }
        }
        return;
    }

    // Skip optional whitespace, then check for delimiter
    if config.skip_initial_space {
        while *i < chars.len() && chars[*i] == ' ' {
            *i += 1;
        }
    }
    if *i < chars.len()
        && !delim_chars.is_empty()
        && *i + delim_chars.len() <= chars.len()
        && chars[*i..*i + delim_chars.len()] == *delim_chars
    {
        *had_delimiter = true;
        *i += delim_chars.len();
        if config.skip_initial_space {
            while *i < chars.len() && chars[*i] == ' ' {
                *i += 1;
            }
        }
    }
}

/// Check if a char slice contains a `:` at bracket depth 0 outside quotes.
/// Used to validate `{...}` content looks like a dict before committing to
/// bracket lookahead.
fn has_colon_at_depth0(chars: &[char]) -> bool {
    let mut depth = 0i32;
    let mut in_q = false;
    let mut esc = false;
    for &ch in chars {
        if esc {
            esc = false;
            continue;
        }
        if ch == '\\' && in_q {
            esc = true;
            continue;
        }
        if ch == '"' {
            in_q = !in_q;
        } else if !in_q {
            match ch {
                '[' | '{' => depth += 1,
                ']' | '}' => depth -= 1,
                ':' if depth == 0 => return true,
                _ => {}
            }
        }
    }
    false
}

/// Split a line into fields, respecting quotes and bracket/brace nesting.
///
/// Replicates the C++ csv_line_tokenizer state machine:
/// - When `[` or `{` appears at the start of a field, a lookahead verifies the
///   brackets balance. If they do, the entire bracketed expression becomes one
///   field. If they don't, the bracket is treated as a regular character (no
///   depth tracking). This matches the C++ lookahead/canceltoken pattern.
/// - Comment char (when set) terminates the current field AND the line, matching
///   C++ IN_FIELD comment handling.
/// - `skip_initial_space` skips spaces at the start of each field (including the
///   first), matching C++ START_FIELD behavior.
pub fn split_fields(line: &str, config: &CsvConfig) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = line.chars().collect();
    let delim_chars: Vec<char> = config.delimiter.chars().collect();
    let mut i = 0;
    let mut in_quote = false;
    let mut escape_next = false;
    // Tracks whether we're in the middle of building a field (IN_FIELD state).
    // When false, we're in START_FIELD state and a trailing empty doesn't
    // produce an extra field.
    let mut in_field = false;
    // Set when a delimiter was the last thing we processed (need trailing empty).
    let mut had_delimiter = false;
    // When we've verified brackets via lookahead, track depth to suppress
    // delimiter matching until the brackets close.
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;

    // Skip leading whitespace before first field (C++ START_FIELD skip)
    if config.skip_initial_space {
        while i < chars.len() && chars[i] == ' ' {
            i += 1;
        }
    }

    while i < chars.len() {
        if escape_next {
            current.push(chars[i]);
            escape_next = false;
            i += 1;
            continue;
        }

        let ch = chars[i];

        // Handle escape in quoted context
        if ch == config.escape_char && in_quote {
            current.push(ch);
            escape_next = true;
            i += 1;
            continue;
        }

        // Handle quote character
        if ch == config.quote_char {
            if in_quote {
                // Check for double-quote escape
                if config.double_quote
                    && i + 1 < chars.len()
                    && chars[i + 1] == config.quote_char
                {
                    current.push(ch);
                    current.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                // End of quoted field
                current.push(ch);
                in_quote = false;
                i += 1;
                continue;
            } else if current.is_empty()
                || current.chars().all(|c| c.is_whitespace())
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
            // (C++ IN_FIELD and START_FIELD both handle comment_char)
            if let Some(cc) = config.comment_char {
                if ch == cc && bracket_depth == 0 && brace_depth == 0 {
                    // Push whatever we have as a field and stop
                    fields.push(finish_field(&current, config));
                    return fields;
                }
            }

            // Bracket/brace lookahead at field start: verify brackets balance
            // before committing to depth tracking. If unbalanced, treat the
            // bracket as a regular character (C++ canceltoken pattern).
            let at_field_start = current.is_empty()
                || current.chars().all(|c| c.is_whitespace());

            if at_field_start && bracket_depth == 0 && brace_depth == 0 {
                if ch == '[' {
                    if let Some(close_pos) = find_balanced_close(&chars, i, '[', ']', &delim_chars, config.skip_initial_space) {
                        // Verified: balanced brackets. Consume the entire
                        // bracketed expression as one field.
                        let field_str: String = chars[i..=close_pos].iter().collect();
                        fields.push(finish_field(&field_str, config));
                        current = String::new();
                        in_field = false;
                        had_delimiter = false;
                        i = close_pos + 1;
                        skip_post_bracket(&chars, &mut i, &delim_chars, config, &mut had_delimiter);
                        continue;
                    }
                    // Unbalanced: treat '[' as regular character, fall through
                } else if ch == '{' {
                    if let Some(close_pos) = find_balanced_close(&chars, i, '{', '}', &delim_chars, config.skip_initial_space) {
                        // For `{...}` groups, verify the content looks like a
                        // dict (contains at least one `:` at depth 0 outside
                        // quotes). This prevents `{`, `{}`, `{foo}` from being
                        // treated as dict fields when they're just text.
                        let inner = &chars[i + 1..close_pos];
                        if has_colon_at_depth0(inner) {
                            let field_str: String =
                                chars[i..=close_pos].iter().collect();
                            fields.push(finish_field(&field_str, config));
                            current = String::new();
                            in_field = false;
                            had_delimiter = false;
                            i = close_pos + 1;
                            skip_post_bracket(&chars, &mut i, &delim_chars, config, &mut had_delimiter);
                            continue;
                        }
                        // Not dict-like: fall through to regular character
                    }
                    // Unbalanced: treat '{' as regular character, fall through
                }
            }

            // Inside an already-open bracket group (from nested brackets within
            // a balanced expression that was NOT consumed by lookahead — this
            // handles cases like `[1,[2,3],4]` where the outer brackets were
            // consumed but we're re-entered via a different path; in practice
            // the lookahead above handles the common case).
            if bracket_depth > 0 || brace_depth > 0 {
                match ch {
                    '[' => bracket_depth += 1,
                    ']' => bracket_depth = (bracket_depth - 1).max(0),
                    '{' => brace_depth += 1,
                    '}' => brace_depth = (brace_depth - 1).max(0),
                    _ => {}
                }
            }

            // Check for delimiter (only at top level)
            if bracket_depth == 0 && brace_depth == 0 && !delim_chars.is_empty()
                && i + delim_chars.len() <= chars.len()
                    && chars[i..i + delim_chars.len()] == delim_chars[..]
                {
                    fields.push(finish_field(&current, config));
                    current = String::new();
                    in_field = false;
                    had_delimiter = true;
                    i += delim_chars.len();
                    // Skip initial space after delimiter
                    if config.skip_initial_space {
                        while i < chars.len() && chars[i] == ' ' {
                            i += 1;
                        }
                    }
                    continue;
                }
        }

        current.push(ch);
        in_field = true;
        i += 1;
    }

    // Push trailing field: if we were building a field (IN_FIELD) or if a
    // delimiter was the last thing processed (trailing empty field).
    if in_field || had_delimiter {
        fields.push(finish_field(&current, config));
    }
    fields
}

/// Process a raw field value: unquote, unescape, handle double-quote.
fn finish_field(raw: &str, config: &CsvConfig) -> String {
    let trimmed = raw.trim();

    // Check for quoted field
    let q = config.quote_char;
    if trimmed.len() >= 2
        && trimmed.starts_with(q)
        && trimmed.ends_with(q)
    {
        let inner = &trimmed[q.len_utf8()..trimmed.len() - q.len_utf8()];

        // Handle double-quote escaping
        let unquoted = if config.double_quote {
            let double_q: String = [q, q].iter().collect();
            inner.replace(&double_q, &q.to_string())
        } else {
            inner.to_string()
        };

        // Process escape sequences
        unescape_csv_string(&unquoted, config.escape_char)
    } else {
        trimmed.to_string()
    }
}

/// Process C-style escape sequences in a CSV field.
fn unescape_csv_string(s: &str, escape_char: char) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == escape_char {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('\'') => result.push('\''),
                Some('/') => result.push('/'),
                Some('b') => result.push('\u{0008}'),
                Some('f') => result.push('\u{000C}'),
                Some('u') => {
                    // Unicode escape
                    let hex: String = chars.by_ref().take(4).collect();
                    if hex.len() == 4 {
                        if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                            if (0xD800..=0xDBFF).contains(&cp) {
                                // High surrogate — look for low surrogate
                                let mut peek = chars.clone();
                                if peek.next() == Some(escape_char) && peek.next() == Some('u') {
                                    let hex2: String = peek.by_ref().take(4).collect();
                                    if hex2.len() == 4 {
                                        if let Ok(cp2) = u32::from_str_radix(&hex2, 16) {
                                            if (0xDC00..=0xDFFF).contains(&cp2) {
                                                let combined = 0x10000
                                                    + ((cp - 0xD800) << 10)
                                                    + (cp2 - 0xDC00);
                                                if let Some(c) = char::from_u32(combined) {
                                                    result.push(c);
                                                    // Advance real iterator
                                                    chars.next(); // escape_char
                                                    chars.next(); // 'u'
                                                    for _ in 0..4 {
                                                        chars.next();
                                                    }
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }
                                // Bad surrogate pair — keep literal \uXXXX
                                // (C++ parity: bad surrogates are not decoded)
                                result.push(escape_char);
                                result.push('u');
                                result.push_str(&hex);
                            } else if let Some(c) = char::from_u32(cp) {
                                result.push(c);
                            } else {
                                result.push('\u{FFFD}');
                            }
                        } else {
                            result.push(escape_char);
                            result.push('u');
                            result.push_str(&hex);
                        }
                    } else {
                        result.push(escape_char);
                        result.push('u');
                        result.push_str(&hex);
                    }
                }
                Some(other) => {
                    result.push(escape_char);
                    result.push(other);
                }
                None => result.push(escape_char),
            }
        } else {
            result.push(ch);
        }
    }

    result
}

/// State for resumable/streaming line splitting.
///
/// Carry this across chunk boundaries so that a quoted field spanning
/// two chunks is handled correctly.
#[derive(Debug, Clone, Default)]
pub struct TokenizerState {
    /// Whether we are currently inside a quoted field.
    pub in_quote: bool,
    /// Whether the next character is escape-escaped.
    pub escape_next: bool,
    /// Partial line accumulated so far (not yet newline-terminated).
    pub partial_line: String,
}

/// Split a byte chunk into complete lines, respecting quote state.
///
/// Returns the complete lines found in this chunk. Updates `state` with
/// any partial line carried forward to the next chunk. The `state.partial_line`
/// holds already-processed characters that are part of the current incomplete
/// line — they are NOT re-scanned for quotes/escapes.
///
/// This is the streaming-friendly version of `split_lines`.
pub fn split_lines_streaming(
    chunk: &str,
    config: &CsvConfig,
    state: &mut TokenizerState,
) -> Vec<String> {
    let mut lines = Vec::new();
    // Start with the partial line from the previous chunk (already processed)
    let mut current = std::mem::take(&mut state.partial_line);
    let chars: Vec<char> = chunk.chars().collect();
    let mut i = 0;

    let mut in_quote = state.in_quote;
    let mut escape_next = state.escape_next;

    let is_standard_terminator = config.line_terminator == "\n"
        || config.line_terminator == "\r\n"
        || config.line_terminator == "\r";

    while i < chars.len() {
        if escape_next {
            current.push(chars[i]);
            escape_next = false;
            i += 1;
            continue;
        }

        let ch = chars[i];

        if ch == config.escape_char && in_quote {
            current.push(ch);
            escape_next = true;
            i += 1;
            continue;
        }

        if ch == config.quote_char {
            current.push(ch);
            if in_quote && config.double_quote && i + 1 < chars.len() && chars[i + 1] == config.quote_char {
                current.push(chars[i + 1]);
                i += 2;
                continue;
            }
            in_quote = !in_quote;
            i += 1;
            continue;
        }

        if !in_quote {
            if is_standard_terminator {
                if ch == '\r' {
                    lines.push(std::mem::take(&mut current));
                    i += 1;
                    if i < chars.len() && chars[i] == '\n' {
                        i += 1;
                    }
                    continue;
                }
                if ch == '\n' {
                    lines.push(std::mem::take(&mut current));
                    i += 1;
                    continue;
                }
            } else {
                let term_chars: Vec<char> = config.line_terminator.chars().collect();
                if i + term_chars.len() <= chars.len()
                    && chars[i..i + term_chars.len()] == term_chars[..]
                {
                    lines.push(std::mem::take(&mut current));
                    i += term_chars.len();
                    continue;
                }
            }
        }

        current.push(ch);
        i += 1;
    }

    // Save state for next chunk
    state.in_quote = in_quote;
    state.escape_next = escape_next;
    state.partial_line = current;

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        let csv = "a,b,c\n1,2,3\n4,5,6\n";
        let config = CsvConfig::default();
        let (header, rows) = tokenize(csv, &config);

        assert_eq!(header.unwrap(), vec!["a", "b", "c"]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["1", "2", "3"]);
        assert_eq!(rows[1], vec!["4", "5", "6"]);
    }

    #[test]
    fn test_bracket_nesting() {
        let csv = "name,data\nfoo,\"[1,2,3]\"\nbar,[4,5,6]\n";
        let config = CsvConfig::default();
        let (header, rows) = tokenize(csv, &config);

        assert_eq!(header.unwrap(), vec!["name", "data"]);
        assert_eq!(rows[0], vec!["foo", "[1,2,3]"]);
        assert_eq!(rows[1], vec!["bar", "[4,5,6]"]);
    }

    #[test]
    fn test_dict_nesting() {
        let csv = "name,data\nfoo,{a:1,b:2}\nbar,\"{\"\"c\"\":3}\"\n";
        let config = CsvConfig::default();
        let (header, rows) = tokenize(csv, &config);

        assert_eq!(header.unwrap(), vec!["name", "data"]);
        assert_eq!(rows[0], vec!["foo", "{a:1,b:2}"]);
        assert_eq!(rows[1], vec!["bar", "{\"c\":3}"]);
    }

    #[test]
    fn test_comment_lines() {
        let csv = "a,b\n# comment\n1,2\n#another\n3,4\n";
        let config = CsvConfig::default();
        let (header, rows) = tokenize(csv, &config);

        assert_eq!(header.unwrap(), vec!["a", "b"]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["1", "2"]);
        assert_eq!(rows[1], vec!["3", "4"]);
    }

    #[test]
    fn test_skip_rows() {
        let csv = "skip1\nskip2\na,b\n1,2\n";
        let config = CsvConfig {
            skip_rows: 2,
            ..Default::default()
        };
        let (header, rows) = tokenize(csv, &config);

        assert_eq!(header.unwrap(), vec!["a", "b"]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec!["1", "2"]);
    }

    #[test]
    fn test_row_limit() {
        let csv = "a,b\n1,2\n3,4\n5,6\n7,8\n";
        let config = CsvConfig {
            row_limit: Some(2),
            ..Default::default()
        };
        let (_, rows) = tokenize(csv, &config);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_escape_sequences_in_quoted() {
        let csv = "a\n\"hello\\nworld\"\n\"tab\\there\"\n";
        let config = CsvConfig::default();
        let (_, rows) = tokenize(csv, &config);

        assert_eq!(rows[0][0], "hello\nworld");
        assert_eq!(rows[1][0], "tab\there");
    }

    #[test]
    fn test_double_quote_escaping() {
        let csv = "a\n\"say \"\"hi\"\"\"\n";
        let config = CsvConfig::default();
        let (_, rows) = tokenize(csv, &config);

        assert_eq!(rows[0][0], "say \"hi\"");
    }

    #[test]
    fn test_semicolon_delimiter() {
        let csv = "a;b;c\n1;2;3\n";
        let config = CsvConfig {
            delimiter: ";".to_string(),
            ..Default::default()
        };
        let (header, rows) = tokenize(csv, &config);

        assert_eq!(header.unwrap(), vec!["a", "b", "c"]);
        assert_eq!(rows[0], vec!["1", "2", "3"]);
    }

    #[test]
    fn test_tab_delimiter() {
        let csv = "a\tb\tc\n1\t2\t3\n";
        let config = CsvConfig {
            delimiter: "\t".to_string(),
            ..Default::default()
        };
        let (header, rows) = tokenize(csv, &config);

        assert_eq!(header.unwrap(), vec!["a", "b", "c"]);
        assert_eq!(rows[0], vec!["1", "2", "3"]);
    }

    #[test]
    fn test_no_header() {
        let csv = "1,2,3\n4,5,6\n";
        let config = CsvConfig {
            has_header: false,
            ..Default::default()
        };
        let (header, rows) = tokenize(csv, &config);

        assert!(header.is_none());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["1", "2", "3"]);
    }

    #[test]
    fn test_crlf_line_endings() {
        let csv = "a,b\r\n1,2\r\n3,4\r\n";
        let config = CsvConfig::default();
        let (header, rows) = tokenize(csv, &config);

        assert_eq!(header.unwrap(), vec!["a", "b"]);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_quoted_field_with_delimiter() {
        let csv = "a,b\n\"hello, world\",42\n";
        let config = CsvConfig::default();
        let (_, rows) = tokenize(csv, &config);

        assert_eq!(rows[0], vec!["hello, world", "42"]);
    }

    #[test]
    fn test_multiline_quoted_field() {
        let csv = "a,b\n\"line1\nline2\",42\n";
        let config = CsvConfig::default();
        let (_, rows) = tokenize(csv, &config);

        assert_eq!(rows[0], vec!["line1\nline2", "42"]);
    }

    #[test]
    fn test_nested_list_with_commas() {
        let csv = "name,data\ntest,[1,[2,3],4]\n";
        let config = CsvConfig::default();
        let (_, rows) = tokenize(csv, &config);

        assert_eq!(rows[0], vec!["test", "[1,[2,3],4]"]);
    }

    #[test]
    fn test_streaming_matches_regular() {
        let csv = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n";
        let config = CsvConfig::default();

        // Regular split_lines
        let regular = split_lines(csv, &config);

        // Streaming: feed entire content as one chunk
        let mut state = TokenizerState::default();
        let mut streaming = split_lines_streaming(csv, &config, &mut state);
        // Flush the partial line
        if !state.partial_line.is_empty() {
            streaming.push(std::mem::take(&mut state.partial_line));
        }

        assert_eq!(regular, streaming);
    }

    #[test]
    fn test_streaming_across_chunks() {
        let config = CsvConfig::default();
        let mut state = TokenizerState::default();

        // Split "a,b\n1,2\n3,4\n" into two chunks
        let lines1 = split_lines_streaming("a,b\n1,", &config, &mut state);
        assert_eq!(lines1, vec!["a,b"]);
        assert_eq!(state.partial_line, "1,");

        let lines2 = split_lines_streaming("2\n3,4\n", &config, &mut state);
        assert_eq!(lines2, vec!["1,2", "3,4"]);
        assert!(state.partial_line.is_empty());
    }

    #[test]
    fn test_streaming_quoted_field_across_chunks() {
        let config = CsvConfig::default();
        let mut state = TokenizerState::default();

        // Quoted field "line1\nline2" split across chunks
        let lines1 = split_lines_streaming("a,b\n\"line1\n", &config, &mut state);
        assert_eq!(lines1, vec!["a,b"]);
        assert!(state.in_quote); // still inside quoted field

        let lines2 = split_lines_streaming("line2\",42\n", &config, &mut state);
        assert_eq!(lines2, vec!["\"line1\nline2\",42"]);
        assert!(!state.in_quote);
    }
}
