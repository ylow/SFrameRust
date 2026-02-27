//! String codec for V2 block format.
//!
//! Two encoding modes:
//!
//! - **Dictionary encoding** (first byte = 1, true): Used when ≤64 unique strings.
//!   Encodes a dictionary of unique strings, then FoR-encodes indices into the dict.
//!
//! - **Direct encoding** (first byte = 0, false): Used when >64 unique strings.
//!   FoR-encodes string lengths, then stores concatenated raw bytes.

use std::collections::HashMap;
use std::io::{Cursor, Read, Write};

use sframe_types::error::{Result, SFrameError};
use sframe_types::varint::{decode_varint, encode_varint};

use crate::codec_integer::{decode_integers_for_reader, encode_integers_for};

/// Decode a string block, returning the decoded strings.
pub fn decode_strings(data: &[u8], num_elements: usize) -> Result<Vec<String>> {
    let mut cursor = Cursor::new(data);

    // First byte: use_dictionary_encoding flag
    let mut flag = [0u8; 1];
    cursor.read_exact(&mut flag)?;
    let use_dict = flag[0] != 0;

    if use_dict {
        decode_strings_dict(&mut cursor, num_elements)
    } else {
        decode_strings_direct(&mut cursor, num_elements)
    }
}

/// Dictionary encoding: read dict, then FoR-decode indices.
fn decode_strings_dict(cursor: &mut Cursor<&[u8]>, num_elements: usize) -> Result<Vec<String>> {
    // Read dictionary length
    let dict_len = decode_varint(cursor)? as usize;
    if dict_len > 64 {
        return Err(SFrameError::Format(format!(
            "Dictionary size {} exceeds maximum of 64",
            dict_len
        )));
    }

    // Read dictionary entries
    let mut dict = Vec::with_capacity(dict_len);
    for _ in 0..dict_len {
        let str_len = decode_varint(cursor)? as usize;
        let mut buf = vec![0u8; str_len];
        cursor.read_exact(&mut buf)?;
        let s = String::from_utf8(buf)
            .map_err(|e| SFrameError::Format(format!("Invalid UTF-8 in dict: {}", e)))?;
        dict.push(s);
    }

    // Read FoR-encoded indices directly from the cursor
    let indices = decode_integers_for_reader(cursor, num_elements)?;

    // Map indices to strings
    let mut result = Vec::with_capacity(num_elements);
    for idx in indices {
        let idx = idx as usize;
        if idx >= dict.len() {
            return Err(SFrameError::Format(format!(
                "Dictionary index {} out of range (dict size {})",
                idx,
                dict.len()
            )));
        }
        result.push(dict[idx].clone());
    }

    Ok(result)
}

/// Direct encoding: FoR-decode lengths, then read concatenated bytes.
fn decode_strings_direct(cursor: &mut Cursor<&[u8]>, num_elements: usize) -> Result<Vec<String>> {
    // Decode string lengths via FoR, advancing cursor past the FoR data
    let lengths = decode_integers_for_reader(cursor, num_elements)?;

    // Read concatenated string bytes
    let mut result = Vec::with_capacity(num_elements);
    for &len in &lengths {
        let len = len as usize;
        let mut buf = vec![0u8; len];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| SFrameError::Format(format!("Failed to read string bytes: {}", e)))?;
        let s = String::from_utf8(buf)
            .map_err(|e| SFrameError::Format(format!("Invalid UTF-8 in string: {}", e)))?;
        result.push(s);
    }

    Ok(result)
}

// ==========================================================================
// Encoding
// ==========================================================================

/// Maximum unique strings for dictionary encoding.
const MAX_DICT_SIZE: usize = 64;

/// Encode strings using dictionary or direct encoding.
pub fn encode_strings(writer: &mut (impl Write + ?Sized), values: &[&str]) -> Result<()> {
    // Build dictionary of unique strings
    let mut dict_map: HashMap<&str, usize> = HashMap::new();
    let mut dict: Vec<&str> = Vec::new();
    for &s in values {
        if !dict_map.contains_key(s) {
            if dict.len() >= MAX_DICT_SIZE {
                // Too many unique strings, fall back to direct encoding
                return encode_strings_direct(writer, values);
            }
            dict_map.insert(s, dict.len());
            dict.push(s);
        }
    }

    // Dictionary encoding
    encode_strings_dict(writer, values, &dict, &dict_map)
}

/// Dictionary encoding: write dict, then FoR-encode indices.
fn encode_strings_dict(
    writer: &mut (impl Write + ?Sized),
    values: &[&str],
    dict: &[&str],
    dict_map: &HashMap<&str, usize>,
) -> Result<()> {
    // Flag: dictionary encoding
    writer.write_all(&[1u8])?;

    // Dictionary length
    encode_varint(dict.len() as u64, writer)?;

    // Dictionary entries: varint length + raw bytes
    for &s in dict {
        encode_varint(s.len() as u64, writer)?;
        writer.write_all(s.as_bytes())?;
    }

    // FoR-encode indices
    let indices: Vec<i64> = values.iter().map(|&s| dict_map[s] as i64).collect();
    encode_integers_for(writer, &indices)?;

    Ok(())
}

/// Direct encoding: FoR-encode lengths, then write concatenated bytes.
fn encode_strings_direct(writer: &mut (impl Write + ?Sized), values: &[&str]) -> Result<()> {
    // Flag: direct encoding
    writer.write_all(&[0u8])?;

    // FoR-encode lengths
    let lengths: Vec<i64> = values.iter().map(|s| s.len() as i64).collect();
    encode_integers_for(writer, &lengths)?;

    // Concatenated raw bytes
    for &s in values {
        writer.write_all(s.as_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_strings_dict() {
        let values = vec!["hello", "world", "hello", "foo", "world"];
        let mut buf = Vec::new();
        encode_strings(&mut buf, &values).unwrap();
        let decoded = decode_strings(&buf, values.len()).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_encode_decode_strings_direct() {
        // Create >64 unique strings to force direct encoding
        let strings: Vec<String> = (0..100).map(|i| format!("str_{}", i)).collect();
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        let mut buf = Vec::new();
        encode_strings(&mut buf, &refs).unwrap();
        let decoded = decode_strings(&buf, refs.len()).unwrap();
        let expected: Vec<String> = refs.iter().map(|s| s.to_string()).collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_encode_decode_empty_strings() {
        let values = vec!["", "", "a", ""];
        let mut buf = Vec::new();
        encode_strings(&mut buf, &values).unwrap();
        let decoded = decode_strings(&buf, values.len()).unwrap();
        assert_eq!(decoded, values);
    }
}
