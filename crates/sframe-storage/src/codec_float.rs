//! Float codec for V2 block format.
//!
//! Two encoding modes selected by a reserved byte:
//!
//! - **Legacy (0):** Doubles are bit-rotated to u64 (sign-magnitude rotation),
//!   then encoded via frame-of-reference integer codec.
//! - **Integer (1):** Doubles that are whole numbers are cast to i64 and
//!   encoded via frame-of-reference integer codec.
//!
//! The BLOCK_ENCODING_EXTENSION flag in BlockInfo indicates the new format.
//! Without that flag, legacy encoding is assumed (no reserved byte).

use std::io::{Cursor, Read};

use sframe_types::error::{Result, SFrameError};

use crate::codec_integer::decode_integers_for;

/// Reserved byte values for float encoding.
const LEGACY_ENCODING: u8 = 0;
const INTEGER_ENCODING: u8 = 1;

/// Decode a float block.
///
/// If `has_encoding_extension` is true, the first byte is a reserved byte
/// selecting the encoding mode. Otherwise, legacy encoding is assumed.
pub fn decode_floats(
    data: &[u8],
    num_elements: usize,
    has_encoding_extension: bool,
) -> Result<Vec<f64>> {
    let mut cursor = Cursor::new(data);

    if has_encoding_extension {
        let mut reserved = [0u8; 1];
        cursor.read_exact(&mut reserved)?;

        match reserved[0] {
            LEGACY_ENCODING => decode_floats_legacy(&data[1..], num_elements),
            INTEGER_ENCODING => decode_floats_integer(&data[1..], num_elements),
            other => Err(SFrameError::Format(format!(
                "Unknown float encoding reserved byte: {}",
                other
            ))),
        }
    } else {
        decode_floats_legacy(data, num_elements)
    }
}

/// Legacy float encoding: bit-rotate doubles to u64, then FoR.
fn decode_floats_legacy(data: &[u8], num_elements: usize) -> Result<Vec<f64>> {
    let integers = decode_integers_for(data, num_elements)?;
    let mut result = Vec::with_capacity(num_elements);
    for val in integers {
        let bits = val as u64;
        let unrotated = unrotate_double_bits(bits);
        result.push(f64::from_bits(unrotated));
    }
    Ok(result)
}

/// Integer float encoding: values are whole numbers stored as integers.
fn decode_floats_integer(data: &[u8], num_elements: usize) -> Result<Vec<f64>> {
    let integers = decode_integers_for(data, num_elements)?;
    let mut result = Vec::with_capacity(num_elements);
    for val in integers {
        result.push(val as f64);
    }
    Ok(result)
}

/// Undo the sign-magnitude rotation used for legacy float encoding.
///
/// C++ encoding: `encoded = (bits << 1) | (bits >> 63)`
/// Decoding: `bits = (encoded >> 1) | (encoded << 63)`
fn unrotate_double_bits(encoded: u64) -> u64 {
    (encoded >> 1) | (encoded << 63)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unrotate_double_bits_roundtrip() {
        let test_values: Vec<f64> = vec![0.0, 1.0, -1.0, 3.14, f64::MAX, f64::MIN];
        for val in test_values {
            let bits = val.to_bits();
            // Simulate C++ encoding
            let encoded = (bits << 1) | (bits >> 63);
            let decoded_bits = unrotate_double_bits(encoded);
            assert_eq!(decoded_bits, bits, "roundtrip failed for {}", val);
        }
    }
}
