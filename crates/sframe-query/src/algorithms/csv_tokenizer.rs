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
        if line.is_empty() {
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
fn is_comment(line: &str, config: &CsvConfig) -> bool {
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

/// Split a line into fields, respecting quotes and bracket/brace nesting.
fn split_fields(line: &str, config: &CsvConfig) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = line.chars().collect();
    let delim_chars: Vec<char> = config.delimiter.chars().collect();
    let mut i = 0;
    let mut in_quote = false;
    let mut escape_next = false;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;

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
            // Track bracket/brace nesting
            match ch {
                '[' => bracket_depth += 1,
                ']' => bracket_depth = (bracket_depth - 1).max(0),
                '{' => brace_depth += 1,
                '}' => brace_depth = (brace_depth - 1).max(0),
                _ => {}
            }

            // Check for delimiter (only at top level)
            if bracket_depth == 0 && brace_depth == 0 && !delim_chars.is_empty() {
                if i + delim_chars.len() <= chars.len()
                    && chars[i..i + delim_chars.len()] == delim_chars[..]
                {
                    fields.push(finish_field(&current, config));
                    current = String::new();
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
        }

        current.push(ch);
        i += 1;
    }

    fields.push(finish_field(&current, config));
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
                Some('0') => result.push('\0'),
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
                                result.push('\u{FFFD}');
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
}
