//! Top-level typed block encoder.
//!
//! Encodes a vector of FlexType values into the V2 block format:
//!
//! [num_types: u8] [type_byte: u8 (if num_types 1 or 2)] [undefined_bitmap (if num_types==2)]
//! [type-specific encoded data]

use std::io::Write;


use sframe_types::bitset::DenseBitset;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};
use sframe_types::serialization::write_flex_type;

use crate::codec_float::encode_floats;
use crate::codec_integer::encode_integers_for;
use crate::codec_string::encode_strings;
use crate::codec_vector::encode_vectors;

/// Encode a typed block from a vector of FlexType values.
///
/// Returns the encoded bytes. The caller is responsible for LZ4 compression
/// and writing the BlockInfo.
pub fn encode_typed_block(values: &[FlexType]) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(vec![0u8]); // num_types = 0
    }

    // Determine the types present
    let mut has_undefined = false;
    let mut data_type: Option<FlexTypeEnum> = None;
    let mut is_homogeneous = true;

    for val in values {
        match val {
            FlexType::Undefined => {
                has_undefined = true;
            }
            other => {
                let t = other.type_enum();
                match data_type {
                    None => data_type = Some(t),
                    Some(existing) if existing == t => {}
                    Some(_) => {
                        is_homogeneous = false;
                        break;
                    }
                }
            }
        }
    }

    if !is_homogeneous {
        // Mixed types — fall back to raw FlexType serialization
        return encode_mixed_block(values);
    }

    let dtype = match data_type {
        Some(t) => t,
        None => {
            // All undefined
            let mut buf = Vec::new();
            buf.push(1u8); // num_types = 1
            buf.push(FlexTypeEnum::Undefined as u8);
            return Ok(buf);
        }
    };

    if has_undefined {
        encode_with_undefineds(values, dtype)
    } else {
        encode_homogeneous(values, dtype)
    }
}

/// Encode a homogeneous block (num_types = 1).
fn encode_homogeneous(values: &[FlexType], dtype: FlexTypeEnum) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.push(1u8); // num_types = 1
    buf.push(dtype as u8);
    encode_typed_values(&mut buf, values, dtype)?;
    Ok(buf)
}

/// Encode a block with undefineds (num_types = 2).
fn encode_with_undefineds(values: &[FlexType], dtype: FlexTypeEnum) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.push(2u8); // num_types = 2
    buf.push(dtype as u8);

    // Build and write undefined bitmap (bit set = undefined)
    let mut bitmap = DenseBitset::new(values.len());
    for (i, val) in values.iter().enumerate() {
        if matches!(val, FlexType::Undefined) {
            bitmap.set(i);
        }
    }
    bitmap.serialize(&mut buf)?;

    // Collect defined values only
    let defined: Vec<&FlexType> = values
        .iter()
        .filter(|v| !matches!(v, FlexType::Undefined))
        .collect();

    encode_typed_values_refs(&mut buf, &defined, dtype)?;
    Ok(buf)
}

/// Encode a mixed-type block (num_types > 2).
fn encode_mixed_block(values: &[FlexType]) -> Result<Vec<u8>> {
    // Count distinct types
    let mut type_set = std::collections::HashSet::new();
    for val in values {
        type_set.insert(val.type_enum());
    }

    let mut buf = Vec::new();
    buf.push(type_set.len() as u8);

    // Write type bytes
    for &t in &type_set {
        buf.push(t as u8);
    }

    // Raw FlexType serialization for each value
    for val in values {
        write_flex_type(&mut buf, val)?;
    }

    Ok(buf)
}

/// Encode values of a specific type.
fn encode_typed_values(
    writer: &mut (impl Write + ?Sized),
    values: &[FlexType],
    dtype: FlexTypeEnum,
) -> Result<()> {
    let refs: Vec<&FlexType> = values.iter().collect();
    encode_typed_values_refs(writer, &refs, dtype)
}

/// Encode values of a specific type from references.
fn encode_typed_values_refs(
    writer: &mut (impl Write + ?Sized),
    values: &[&FlexType],
    dtype: FlexTypeEnum,
) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }

    match dtype {
        FlexTypeEnum::Integer => {
            let ints: Vec<i64> = values
                .iter()
                .map(|v| match v {
                    FlexType::Integer(i) => Ok(*i),
                    _ => Err(SFrameError::Format("Expected Integer".to_string())),
                })
                .collect::<Result<_>>()?;
            encode_integers_for(writer, &ints)
        }
        FlexTypeEnum::Float => {
            let floats: Vec<f64> = values
                .iter()
                .map(|v| match v {
                    FlexType::Float(f) => Ok(*f),
                    _ => Err(SFrameError::Format("Expected Float".to_string())),
                })
                .collect::<Result<_>>()?;
            encode_floats(writer, &floats)
        }
        FlexTypeEnum::String => {
            let strs: Vec<&str> = values
                .iter()
                .map(|v| match v {
                    FlexType::String(s) => Ok(s.as_ref()),
                    _ => Err(SFrameError::Format("Expected String".to_string())),
                })
                .collect::<Result<_>>()?;
            encode_strings(writer, &strs)
        }
        FlexTypeEnum::Vector => {
            let vecs: Vec<&[f64]> = values
                .iter()
                .map(|v| match v {
                    FlexType::Vector(v) => Ok(v.as_ref()),
                    _ => Err(SFrameError::Format("Expected Vector".to_string())),
                })
                .collect::<Result<_>>()?;
            encode_vectors(writer, &vecs)
        }
        FlexTypeEnum::List | FlexTypeEnum::Dict | FlexTypeEnum::DateTime => {
            // Raw archive serialization
            for val in values {
                write_flex_type(writer, val)?;
            }
            Ok(())
        }
        FlexTypeEnum::Undefined => Ok(()),
    }
}
