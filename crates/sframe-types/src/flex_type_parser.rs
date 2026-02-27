//! Recursive descent parser for FlexType values from string representations.
//!
//! This is the Rust equivalent of the C++ `flexible_type_spirit_parser`.
//! It parses a string into the best-matching FlexType:
//!
//! **Parse priority** (first match wins):
//! 1. Float (requires decimal point: `1.5`, `-3.14e10`, `inf`, `nan`)
//! 2. Integer (`123`, `-42`)
//! 3. Vector (`[1.0, 2.0]` or `[1 2 3]` or `[1;2;3]` — all-numeric)
//! 4. List (`[a, "b", 3]` — heterogeneous, comma-separated)
//! 5. Dict (`{"key": "value", 1: 2}`)
//! 6. String (fallback)
//!
//! When the parser sees `[` or `{`, it attempts to parse the full bracketed
//! structure. If parsing fails (mismatched brackets, invalid content), it
//! falls back to treating the entire input as a string.

use std::sync::Arc;

use crate::flex_type::FlexType;

/// Parse a string into the best-matching FlexType.
///
/// This is the main entry point. It tries each type in priority order
/// and falls back to String if nothing else matches.
pub fn parse_flextype(s: &str) -> FlexType {
    let s = s.trim();
    if s.is_empty() {
        return FlexType::Undefined;
    }

    // Try structured types first if input starts with bracket/brace
    let first = s.as_bytes()[0];
    if first == b'[' {
        // Try vector first (all-numeric), then list (heterogeneous)
        if let Some(v) = try_parse_vector(s) {
            return v;
        }
        if let Some(v) = try_parse_list(s) {
            return v;
        }
        // Bracket didn't parse as vector or list — fall through to string
        return FlexType::String(Arc::from(s));
    }
    if first == b'{' {
        if let Some(v) = try_parse_dict(s) {
            return v;
        }
        return FlexType::String(Arc::from(s));
    }

    // Try numeric types
    // Float first (requires dot), then integer
    if let Some(v) = try_parse_float_strict(s) {
        return v;
    }
    if let Some(v) = try_parse_integer(s) {
        return v;
    }

    // Everything else is a string
    FlexType::String(Arc::from(s))
}

/// Try to parse as a float. Requires a decimal point, 'inf', 'nan', or
/// scientific notation to distinguish from integers.
fn try_parse_float_strict(s: &str) -> Option<FlexType> {
    // Must contain '.', 'e', 'E', 'inf', 'nan' to be a float (not integer)
    let lower = s.to_ascii_lowercase();
    let has_dot = s.contains('.');
    let has_exp = lower.contains('e');
    let is_special = lower == "inf"
        || lower == "-inf"
        || lower == "+inf"
        || lower == "nan"
        || lower == "-nan"
        || lower == "+nan"
        || lower == "infinity"
        || lower == "-infinity"
        || lower == "+infinity";

    if !has_dot && !has_exp && !is_special {
        return None;
    }

    s.parse::<f64>().ok().map(FlexType::Float)
}

/// Try to parse as an integer.
fn try_parse_integer(s: &str) -> Option<FlexType> {
    s.parse::<i64>().ok().map(FlexType::Integer)
}

/// Try to parse as a vector: `[num, num, ...]` with flexible separators.
///
/// Vectors require ALL elements to be numeric (f64). Separators can be
/// comma, semicolon, or whitespace. Returns None if parsing fails.
fn try_parse_vector(s: &str) -> Option<FlexType> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return None;
    }

    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        return Some(FlexType::Vector(Arc::from(Vec::<f64>::new().as_slice())));
    }

    // Split by commas, semicolons, or whitespace
    let elements = split_vector_elements(inner);
    let mut values = Vec::with_capacity(elements.len());

    for elem in &elements {
        let elem = elem.trim();
        if elem.is_empty() {
            continue;
        }
        match elem.parse::<f64>() {
            Ok(v) => values.push(v),
            Err(_) => return None, // Not all-numeric → not a vector
        }
    }

    Some(FlexType::Vector(Arc::from(values)))
}

/// Split vector elements by comma, semicolon, or whitespace.
/// Tries comma first, then semicolon, then whitespace.
fn split_vector_elements(s: &str) -> Vec<&str> {
    // If contains comma, split by comma
    if s.contains(',') {
        return s.split(',').collect();
    }
    // If contains semicolon, split by semicolon
    if s.contains(';') {
        return s.split(';').collect();
    }
    // Otherwise split by whitespace
    s.split_whitespace().collect()
}

/// Try to parse as a list: `[elem, elem, ...]` with comma separators.
///
/// Lists are heterogeneous — elements are recursively parsed as FlexType.
/// If parsing fails (mismatched brackets), returns None.
fn try_parse_list(s: &str) -> Option<FlexType> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return None;
    }

    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        // Empty brackets — could be empty vector or empty list.
        // The C++ returns empty vector for []. We already handled that
        // in try_parse_vector, so if we get here return empty list.
        return Some(FlexType::List(Arc::from(Vec::<FlexType>::new().as_slice())));
    }

    // Split by commas respecting bracket/brace nesting
    let elements = split_top_level(inner, ',')?;
    let mut values = Vec::with_capacity(elements.len());

    for elem in &elements {
        let elem = elem.trim();
        if elem.is_empty() {
            values.push(FlexType::Undefined);
        } else {
            values.push(parse_element(elem));
        }
    }

    Some(FlexType::List(Arc::from(values)))
}

/// Try to parse as a dict: `{key: value, key: value, ...}`
///
/// Keys and values are recursively parsed. Key-value pairs are separated
/// by commas (or spaces in the C++ version, but we use commas for clarity).
fn try_parse_dict(s: &str) -> Option<FlexType> {
    let s = s.trim();
    if !s.starts_with('{') || !s.ends_with('}') {
        return None;
    }

    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        return Some(FlexType::Dict(Arc::from(
            Vec::<(FlexType, FlexType)>::new().as_slice(),
        )));
    }

    // Split by commas respecting nesting
    let pairs = split_top_level(inner, ',')?;
    let mut entries = Vec::with_capacity(pairs.len());

    for pair in &pairs {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }

        // Find the colon separating key from value, respecting nesting and quotes
        let colon_pos = find_top_level_colon(pair)?;
        let key_str = pair[..colon_pos].trim();
        let val_str = pair[colon_pos + 1..].trim();

        let key = parse_element(key_str);
        let value = parse_element(val_str);
        entries.push((key, value));
    }

    Some(FlexType::Dict(Arc::from(entries)))
}

/// Parse a single element within a list or dict.
/// Handles quoted strings, nested structures, and atomic types.
fn parse_element(s: &str) -> FlexType {
    let s = s.trim();
    if s.is_empty() {
        return FlexType::Undefined;
    }

    // Quoted string
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        let inner = &s[1..s.len() - 1];
        let unescaped = unescape_string(inner);
        return FlexType::String(Arc::from(unescaped.as_str()));
    }

    // Nested structure
    let first = s.as_bytes()[0];
    if first == b'[' {
        if let Some(v) = try_parse_vector(s) {
            return v;
        }
        if let Some(v) = try_parse_list(s) {
            return v;
        }
        return FlexType::String(Arc::from(s));
    }
    if first == b'{' {
        if let Some(v) = try_parse_dict(s) {
            return v;
        }
        return FlexType::String(Arc::from(s));
    }

    // Atomic types: float (with dot), integer, then string fallback
    if let Some(v) = try_parse_float_strict(s) {
        return v;
    }
    if let Some(v) = try_parse_integer(s) {
        return v;
    }

    FlexType::String(Arc::from(s))
}

/// Split a string by a delimiter, respecting bracket/brace/quote nesting.
/// Returns None if brackets/braces are mismatched.
fn split_top_level(s: &str, delim: char) -> Option<Vec<&str>> {
    let mut result = Vec::new();
    let mut depth_bracket = 0i32;
    let mut depth_brace = 0i32;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escape_next = false;
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        if ch == '\\' && (in_single_quote || in_double_quote) {
            escape_next = true;
            continue;
        }

        if ch == '"' && !in_single_quote {
            in_double_quote = !in_double_quote;
            continue;
        }
        if ch == '\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
            continue;
        }

        if in_single_quote || in_double_quote {
            continue;
        }

        match ch {
            '[' => depth_bracket += 1,
            ']' => {
                depth_bracket -= 1;
                if depth_bracket < 0 {
                    return None;
                }
            }
            '{' => depth_brace += 1,
            '}' => {
                depth_brace -= 1;
                if depth_brace < 0 {
                    return None;
                }
            }
            c if c == delim && depth_bracket == 0 && depth_brace == 0 => {
                result.push(&s[start..i]);
                start = i + ch.len_utf8();
            }
            _ => {}
        }
    }

    if depth_bracket != 0 || depth_brace != 0 || in_single_quote || in_double_quote {
        return None;
    }

    result.push(&s[start..]);
    Some(result)
}

/// Find the position of the first top-level colon in a string,
/// respecting bracket/brace/quote nesting.
fn find_top_level_colon(s: &str) -> Option<usize> {
    let mut depth_bracket = 0i32;
    let mut depth_brace = 0i32;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escape_next = false;

    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        if ch == '\\' && (in_single_quote || in_double_quote) {
            escape_next = true;
            continue;
        }

        if ch == '"' && !in_single_quote {
            in_double_quote = !in_double_quote;
            continue;
        }
        if ch == '\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
            continue;
        }

        if in_single_quote || in_double_quote {
            continue;
        }

        match ch {
            '[' => depth_bracket += 1,
            ']' => depth_bracket -= 1,
            '{' => depth_brace += 1,
            '}' => depth_brace -= 1,
            ':' if depth_bracket == 0 && depth_brace == 0 => return Some(i),
            _ => {}
        }
    }

    None
}

/// Process C-style escape sequences in a string.
fn unescape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
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
                    // Unicode escape: \uXXXX
                    let hex: String = chars.by_ref().take(4).collect();
                    if hex.len() == 4 {
                        if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                            // Check for surrogate pair
                            if (0xD800..=0xDBFF).contains(&cp) {
                                // High surrogate — look for \uXXXX low surrogate
                                let mut peek_chars = chars.clone();
                                let next1 = peek_chars.next();
                                let next2 = peek_chars.next();
                                if next1 == Some('\\') && next2 == Some('u') {
                                    let hex2: String = peek_chars.by_ref().take(4).collect();
                                    if hex2.len() == 4 {
                                        if let Ok(cp2) = u32::from_str_radix(&hex2, 16) {
                                            if (0xDC00..=0xDFFF).contains(&cp2) {
                                                // Valid surrogate pair
                                                let combined = 0x10000
                                                    + ((cp - 0xD800) << 10)
                                                    + (cp2 - 0xDC00);
                                                if let Some(c) = char::from_u32(combined) {
                                                    result.push(c);
                                                    // Advance the real iterator past \uXXXX
                                                    chars.next(); // '\'
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
                                // Invalid/incomplete surrogate pair — output replacement char
                                result.push('\u{FFFD}');
                            } else if let Some(c) = char::from_u32(cp) {
                                result.push(c);
                            } else {
                                result.push('\u{FFFD}');
                            }
                        } else {
                            result.push_str("\\u");
                            result.push_str(&hex);
                        }
                    } else {
                        result.push_str("\\u");
                        result.push_str(&hex);
                    }
                }
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
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

    // === Integer parsing ===

    #[test]
    fn test_parse_integer() {
        assert_eq!(parse_flextype("123"), FlexType::Integer(123));
        assert_eq!(parse_flextype("-42"), FlexType::Integer(-42));
        assert_eq!(parse_flextype("0"), FlexType::Integer(0));
    }

    #[test]
    fn test_parse_integer_with_whitespace() {
        assert_eq!(parse_flextype("  123  "), FlexType::Integer(123));
    }

    // === Float parsing ===

    #[test]
    fn test_parse_float() {
        assert_eq!(parse_flextype("1.5"), FlexType::Float(1.5));
        assert_eq!(parse_flextype("-3.14"), FlexType::Float(-3.14));
        assert_eq!(parse_flextype("0.0"), FlexType::Float(0.0));
    }

    #[test]
    fn test_parse_float_scientific() {
        match parse_flextype("1.5e10") {
            FlexType::Float(v) => assert!((v - 1.5e10).abs() < 1.0),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_float_special() {
        match parse_flextype("inf") {
            FlexType::Float(v) => assert!(v.is_infinite() && v > 0.0),
            other => panic!("Expected Float(inf), got {:?}", other),
        }
        match parse_flextype("-inf") {
            FlexType::Float(v) => assert!(v.is_infinite() && v < 0.0),
            other => panic!("Expected Float(-inf), got {:?}", other),
        }
        match parse_flextype("nan") {
            FlexType::Float(v) => assert!(v.is_nan()),
            other => panic!("Expected Float(nan), got {:?}", other),
        }
    }

    // === String parsing ===

    #[test]
    fn test_parse_string() {
        assert_eq!(
            parse_flextype("hello"),
            FlexType::String(Arc::from("hello"))
        );
        assert_eq!(
            parse_flextype("hello world"),
            FlexType::String(Arc::from("hello world"))
        );
    }

    #[test]
    fn test_parse_empty() {
        assert_eq!(parse_flextype(""), FlexType::Undefined);
        assert_eq!(parse_flextype("   "), FlexType::Undefined);
    }

    // === Vector parsing ===

    #[test]
    fn test_parse_vector_comma() {
        match parse_flextype("[1.0, 2.0, 3.0]") {
            FlexType::Vector(v) => {
                assert_eq!(v.as_ref(), &[1.0, 2.0, 3.0]);
            }
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vector_space() {
        match parse_flextype("[1.0 2.0 3.0]") {
            FlexType::Vector(v) => {
                assert_eq!(v.as_ref(), &[1.0, 2.0, 3.0]);
            }
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vector_semicolon() {
        match parse_flextype("[1.0;2.0;3.0]") {
            FlexType::Vector(v) => {
                assert_eq!(v.as_ref(), &[1.0, 2.0, 3.0]);
            }
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vector_integers_become_floats() {
        // Integers in brackets are still valid vector elements
        match parse_flextype("[1, 2, 3]") {
            FlexType::Vector(v) => {
                assert_eq!(v.as_ref(), &[1.0, 2.0, 3.0]);
            }
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_empty_vector() {
        match parse_flextype("[]") {
            FlexType::Vector(v) => {
                assert_eq!(v.len(), 0);
            }
            other => panic!("Expected empty Vector, got {:?}", other),
        }
    }

    // === List parsing ===

    #[test]
    fn test_parse_list_heterogeneous() {
        match parse_flextype("[1, \"hello\", 2.5]") {
            FlexType::List(v) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[0], FlexType::Integer(1));
                assert_eq!(v[1], FlexType::String(Arc::from("hello")));
                assert_eq!(v[2], FlexType::Float(2.5));
            }
            other => panic!("Expected List, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_list_nested() {
        match parse_flextype("[[1, 2], [3, 4]]") {
            FlexType::List(v) => {
                assert_eq!(v.len(), 2);
                match &v[0] {
                    FlexType::Vector(inner) => assert_eq!(inner.as_ref(), &[1.0, 2.0]),
                    other => panic!("Expected Vector, got {:?}", other),
                }
                match &v[1] {
                    FlexType::Vector(inner) => assert_eq!(inner.as_ref(), &[3.0, 4.0]),
                    other => panic!("Expected Vector, got {:?}", other),
                }
            }
            other => panic!("Expected List, got {:?}", other),
        }
    }

    // === Dict parsing ===

    #[test]
    fn test_parse_dict() {
        match parse_flextype("{\"a\": 1, \"b\": 2}") {
            FlexType::Dict(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0].0, FlexType::String(Arc::from("a")));
                assert_eq!(v[0].1, FlexType::Integer(1));
                assert_eq!(v[1].0, FlexType::String(Arc::from("b")));
                assert_eq!(v[1].1, FlexType::Integer(2));
            }
            other => panic!("Expected Dict, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_dict_unquoted_keys() {
        match parse_flextype("{a: 1, b: 2}") {
            FlexType::Dict(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0].0, FlexType::String(Arc::from("a")));
                assert_eq!(v[0].1, FlexType::Integer(1));
            }
            other => panic!("Expected Dict, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_empty_dict() {
        match parse_flextype("{}") {
            FlexType::Dict(v) => {
                assert_eq!(v.len(), 0);
            }
            other => panic!("Expected empty Dict, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_dict_nested() {
        match parse_flextype("{\"a\": {\"b\": 1}}") {
            FlexType::Dict(v) => {
                assert_eq!(v.len(), 1);
                assert_eq!(v[0].0, FlexType::String(Arc::from("a")));
                match &v[0].1 {
                    FlexType::Dict(inner) => {
                        assert_eq!(inner[0].0, FlexType::String(Arc::from("b")));
                        assert_eq!(inner[0].1, FlexType::Integer(1));
                    }
                    other => panic!("Expected Dict value, got {:?}", other),
                }
            }
            other => panic!("Expected Dict, got {:?}", other),
        }
    }

    // === Fallback / edge cases ===

    #[test]
    fn test_mismatched_bracket_becomes_string() {
        // Mismatched brackets should fall back to string
        assert_eq!(
            parse_flextype("[abc"),
            FlexType::String(Arc::from("[abc"))
        );
        assert_eq!(
            parse_flextype("{abc"),
            FlexType::String(Arc::from("{abc"))
        );
    }

    #[test]
    fn test_bracket_with_non_numeric_becomes_list_or_string() {
        // [a, b, c] — not numeric, so try list; elements are strings
        match parse_flextype("[a, b, c]") {
            FlexType::List(v) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[0], FlexType::String(Arc::from("a")));
                assert_eq!(v[1], FlexType::String(Arc::from("b")));
                assert_eq!(v[2], FlexType::String(Arc::from("c")));
            }
            other => panic!("Expected List, got {:?}", other),
        }
    }

    // === Escape sequences ===

    #[test]
    fn test_unescape_string() {
        assert_eq!(unescape_string(r"hello\nworld"), "hello\nworld");
        assert_eq!(unescape_string(r"tab\there"), "tab\there");
        assert_eq!(unescape_string(r"back\\slash"), "back\\slash");
        assert_eq!(unescape_string(r#"say \"hi\""#), "say \"hi\"");
    }

    #[test]
    fn test_unescape_unicode() {
        assert_eq!(unescape_string(r"\u0041"), "A");
        assert_eq!(unescape_string(r"\u00E9"), "é");
    }

    #[test]
    fn test_unescape_surrogate_pair() {
        // U+1D11E = Musical Symbol G Clef (𝄞)
        assert_eq!(unescape_string(r"\uD834\uDD1E"), "𝄞");
    }

    // === Quoted elements in structured types ===

    #[test]
    fn test_parse_dict_with_quoted_values() {
        match parse_flextype("{\"key\": \"value with spaces\"}") {
            FlexType::Dict(v) => {
                assert_eq!(v.len(), 1);
                assert_eq!(v[0].0, FlexType::String(Arc::from("key")));
                assert_eq!(
                    v[0].1,
                    FlexType::String(Arc::from("value with spaces"))
                );
            }
            other => panic!("Expected Dict, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_list_with_quoted_strings() {
        match parse_flextype("[\"hello, world\", \"foo\"]") {
            FlexType::List(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], FlexType::String(Arc::from("hello, world")));
                assert_eq!(v[1], FlexType::String(Arc::from("foo")));
            }
            other => panic!("Expected List, got {:?}", other),
        }
    }

    // === Mixed-type dict values ===

    #[test]
    fn test_parse_dict_mixed_types() {
        match parse_flextype("{\"name\": \"John\", \"age\": 30, \"score\": 9.5}") {
            FlexType::Dict(v) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[0].1, FlexType::String(Arc::from("John")));
                assert_eq!(v[1].1, FlexType::Integer(30));
                assert_eq!(v[2].1, FlexType::Float(9.5));
            }
            other => panic!("Expected Dict, got {:?}", other),
        }
    }
}
