//! Vector (Vec<f64>) codec for V2 block format.
//!
//! Encoding:
//!   [1 byte reserved] [FoR-encoded lengths] [float-encoded flattened values]
//!
//! The reserved byte value 0 indicates NEW_ENCODING (standard).

use std::io::{Cursor, Read, Write};

use sframe_types::error::{Result, SFrameError};

use crate::codec_float::{decode_floats, encode_floats};
use crate::codec_integer::{decode_integers_for_reader, encode_integers_for};

/// Decode a vector block, returning Vec<f64> for each element.
pub fn decode_vectors(
    data: &[u8],
    num_elements: usize,
    has_encoding_extension: bool,
) -> Result<Vec<Vec<f64>>> {
    let mut cursor = Cursor::new(data);

    // Reserved byte
    let mut reserved = [0u8; 1];
    cursor.read_exact(&mut reserved)?;

    // Decode vector lengths via FoR, advancing cursor past the FoR data
    let lengths = decode_integers_for_reader(&mut cursor, num_elements)?;

    // The rest of the data is float-encoded flattened values
    let values_offset = cursor.position() as usize;
    let values_data = &data[values_offset..];

    // Total number of float values
    let total_values: i64 = lengths.iter().sum();
    if total_values < 0 {
        return Err(SFrameError::Format(format!(
            "Negative total vector length: {}",
            total_values
        )));
    }

    // Decode all float values at once
    let flat_values = decode_floats(values_data, total_values as usize, has_encoding_extension)?;

    // Split into individual vectors
    let mut result = Vec::with_capacity(num_elements);
    let mut offset = 0usize;
    for &len in &lengths {
        let len = len as usize;
        if offset + len > flat_values.len() {
            return Err(SFrameError::Format(format!(
                "Vector data overflow: offset {} + len {} > total {}",
                offset,
                len,
                flat_values.len()
            )));
        }
        result.push(flat_values[offset..offset + len].to_vec());
        offset += len;
    }

    Ok(result)
}

// ==========================================================================
// Encoding
// ==========================================================================

/// Reserved byte for new vector encoding format.
const VECTOR_RESERVED_NEW: u8 = 0;

/// Encode vectors: reserved byte + FoR lengths + float-encoded values.
pub fn encode_vectors(writer: &mut (impl Write + ?Sized), vectors: &[&[f64]]) -> Result<()> {
    // Reserved byte
    writer.write_all(&[VECTOR_RESERVED_NEW])?;

    // Encode lengths
    let lengths: Vec<i64> = vectors.iter().map(|v| v.len() as i64).collect();
    encode_integers_for(writer, &lengths)?;

    // Flatten and encode values
    let flat: Vec<f64> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    encode_floats(writer, &flat)?;

    Ok(())
}
