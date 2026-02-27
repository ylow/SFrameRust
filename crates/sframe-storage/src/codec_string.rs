//! String codec for V2 block format.
//!
//! Two encoding modes:
//!
//! - **Dictionary encoding** (first byte = 1, true): Used when ≤64 unique strings.
//!   Encodes a dictionary of unique strings, then FoR-encodes indices into the dict.
//!
//! - **Direct encoding** (first byte = 0, false): Used when >64 unique strings.
//!   FoR-encodes string lengths, then stores concatenated raw bytes.

use std::io::{Cursor, Read};

use sframe_types::error::{Result, SFrameError};
use sframe_types::varint::decode_varint;

use crate::codec_integer::decode_integers_for_reader;

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

#[cfg(test)]
mod tests {
    // String codec tests will be done via the typed block decoder
    // against the actual sample data, since constructing valid
    // FoR-encoded test data by hand is error-prone.
}
