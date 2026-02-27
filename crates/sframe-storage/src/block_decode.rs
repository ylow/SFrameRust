//! Top-level typed block decoder.
//!
//! A typed block (IS_FLEXIBLE_TYPE flag set) has the following layout:
//!
//! [num_types: u8] [type_byte: u8 (if num_types 1 or 2)] [undefined_bitmap (if num_types==2)]
//! [type-specific encoded data]
//!
//! num_types values:
//!   0: empty block
//!   1: homogeneous block (all values same type, no undefineds)
//!   2: homogeneous with undefineds (bitmap marks which are undefined)
//!   >2: mixed types — fall back to raw FlexType archive deserialization

use std::io::Cursor;
use std::sync::Arc;

use sframe_types::bitset::DenseBitset;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};
use sframe_types::serialization::read_flex_type;

use crate::block_info::BlockInfo;
use crate::codec_float::decode_floats;
use crate::codec_integer::decode_integers_for;
use crate::codec_string::decode_strings;
use crate::codec_vector::decode_vectors;

/// Decode a typed block into a vector of FlexType values.
pub fn decode_typed_block(data: &[u8], block_info: &BlockInfo) -> Result<Vec<FlexType>> {
    let num_elem = block_info.num_elem as usize;
    if num_elem == 0 || data.is_empty() {
        return Ok(Vec::new());
    }

    let mut cursor = Cursor::new(data);

    // Read num_types byte
    let mut num_types_buf = [0u8; 1];
    std::io::Read::read_exact(&mut cursor, &mut num_types_buf)?;
    let num_types = num_types_buf[0];

    match num_types {
        0 => Ok(Vec::new()),
        1 => {
            // Homogeneous block — all values same type, no undefineds
            let mut type_buf = [0u8; 1];
            std::io::Read::read_exact(&mut cursor, &mut type_buf)?;
            let dtype = FlexTypeEnum::try_from(type_buf[0])?;

            let pos = cursor.position() as usize;
            let type_data = &data[pos..];

            decode_typed_values(type_data, num_elem, dtype, block_info)
        }
        2 => {
            // Homogeneous with undefineds
            let mut type_buf = [0u8; 1];
            std::io::Read::read_exact(&mut cursor, &mut type_buf)?;
            let dtype = FlexTypeEnum::try_from(type_buf[0])?;

            // Read undefined bitmap
            let bitmap = DenseBitset::deserialize(&mut cursor)?;

            // Count defined values
            let defined_count = (0..num_elem).filter(|i| !bitmap.get(*i)).count();

            let pos = cursor.position() as usize;
            let type_data = &data[pos..];

            // Decode only the defined values
            let defined_values = decode_typed_values(type_data, defined_count, dtype, block_info)?;

            // Merge defined values with undefineds
            let mut result = Vec::with_capacity(num_elem);
            let mut def_idx = 0;
            for i in 0..num_elem {
                if bitmap.get(i) {
                    result.push(FlexType::Undefined);
                } else {
                    if def_idx >= defined_values.len() {
                        return Err(SFrameError::Format(
                            "Not enough defined values for bitmap".to_string(),
                        ));
                    }
                    result.push(defined_values[def_idx].clone());
                    def_idx += 1;
                }
            }

            Ok(result)
        }
        _ => {
            // Multiple types — raw FlexType deserialization
            let mut result = Vec::with_capacity(num_elem);
            for _ in 0..num_elem {
                result.push(read_flex_type(&mut cursor)?);
            }
            Ok(result)
        }
    }
}

/// Decode values of a specific type from encoded data.
fn decode_typed_values(
    data: &[u8],
    num_elements: usize,
    dtype: FlexTypeEnum,
    block_info: &BlockInfo,
) -> Result<Vec<FlexType>> {
    if num_elements == 0 {
        return Ok(Vec::new());
    }

    let has_ext = block_info.has_encoding_extension();

    match dtype {
        FlexTypeEnum::Integer => {
            let values = decode_integers_for(data, num_elements)?;
            Ok(values.into_iter().map(FlexType::Integer).collect())
        }
        FlexTypeEnum::Float => {
            let values = decode_floats(data, num_elements, has_ext)?;
            Ok(values.into_iter().map(FlexType::Float).collect())
        }
        FlexTypeEnum::String => {
            let values = decode_strings(data, num_elements)?;
            Ok(values
                .into_iter()
                .map(|s| FlexType::String(Arc::from(s)))
                .collect())
        }
        FlexTypeEnum::Vector => {
            let values = decode_vectors(data, num_elements, has_ext)?;
            Ok(values
                .into_iter()
                .map(|v| FlexType::Vector(Arc::from(v)))
                .collect())
        }
        FlexTypeEnum::List | FlexTypeEnum::Dict | FlexTypeEnum::DateTime => {
            // These types use raw archive serialization
            let mut cursor = Cursor::new(data);
            let mut result = Vec::with_capacity(num_elements);
            for _ in 0..num_elements {
                result.push(read_flex_type(&mut cursor)?);
            }
            Ok(result)
        }
        FlexTypeEnum::Undefined => {
            Ok(vec![FlexType::Undefined; num_elements])
        }
    }
}
