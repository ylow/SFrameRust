//! GraphLab-compatible archive serialization and deserialization.
//!
//! The C++ oarchive/iarchive serializes primitives as little-endian POD:
//! - Integers: raw LE bytes (u8/u16/u32/u64/i64)
//! - Doubles: raw LE 8 bytes
//! - Strings: 8-byte LE length prefix + raw bytes
//! - Vectors: 8-byte LE length prefix + elements
//! - FlexType: 1-byte tag (128 + type_enum) + value data

use std::io::{Read, Write};
use std::sync::Arc;

use crate::error::{Result, SFrameError};
use crate::flex_type::{FlexDateTime, FlexType, FlexTypeEnum};

// --- Primitive readers ---

pub fn read_u8(reader: &mut (impl Read + ?Sized)) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub fn read_u16(reader: &mut (impl Read + ?Sized)) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

pub fn read_u32(reader: &mut (impl Read + ?Sized)) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn read_u64(reader: &mut (impl Read + ?Sized)) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub fn read_i64(reader: &mut (impl Read + ?Sized)) -> Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

pub fn read_f64(reader: &mut (impl Read + ?Sized)) -> Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

pub fn read_i32(reader: &mut (impl Read + ?Sized)) -> Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

pub fn read_bytes(reader: &mut impl Read, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

// --- Primitive writers ---

pub fn write_u8(writer: &mut (impl Write + ?Sized), val: u8) -> Result<()> {
    writer.write_all(&[val])?;
    Ok(())
}

pub fn write_u16(writer: &mut (impl Write + ?Sized), val: u16) -> Result<()> {
    writer.write_all(&val.to_le_bytes())?;
    Ok(())
}

pub fn write_u32(writer: &mut (impl Write + ?Sized), val: u32) -> Result<()> {
    writer.write_all(&val.to_le_bytes())?;
    Ok(())
}

pub fn write_u64(writer: &mut (impl Write + ?Sized), val: u64) -> Result<()> {
    writer.write_all(&val.to_le_bytes())?;
    Ok(())
}

pub fn write_i64(writer: &mut (impl Write + ?Sized), val: i64) -> Result<()> {
    writer.write_all(&val.to_le_bytes())?;
    Ok(())
}

pub fn write_f64(writer: &mut (impl Write + ?Sized), val: f64) -> Result<()> {
    writer.write_all(&val.to_le_bytes())?;
    Ok(())
}

/// Write a GraphLab-serialized string: 8-byte LE length + raw bytes.
pub fn write_string(writer: &mut (impl Write + ?Sized), s: &str) -> Result<()> {
    write_u64(writer, s.len() as u64)?;
    writer.write_all(s.as_bytes())?;
    Ok(())
}

/// Write a GraphLab-serialized Vec<f64>: 8-byte LE length + raw f64 bytes.
pub fn write_vec_f64(writer: &mut (impl Write + ?Sized), v: &[f64]) -> Result<()> {
    write_u64(writer, v.len() as u64)?;
    for &val in v {
        write_f64(writer, val)?;
    }
    Ok(())
}

/// Serialize a FlexType to GraphLab archive format.
pub fn write_flex_type(writer: &mut (impl Write + ?Sized), val: &FlexType) -> Result<()> {
    match val {
        FlexType::Integer(v) => {
            write_u8(writer, FLEX_TYPE_TAG_OFFSET + FlexTypeEnum::Integer as u8)?;
            write_i64(writer, *v)?;
        }
        FlexType::Float(v) => {
            write_u8(writer, FLEX_TYPE_TAG_OFFSET + FlexTypeEnum::Float as u8)?;
            write_f64(writer, *v)?;
        }
        FlexType::String(s) => {
            write_u8(writer, FLEX_TYPE_TAG_OFFSET + FlexTypeEnum::String as u8)?;
            write_string(writer, s)?;
        }
        FlexType::Vector(v) => {
            write_u8(writer, FLEX_TYPE_TAG_OFFSET + FlexTypeEnum::Vector as u8)?;
            write_vec_f64(writer, v)?;
        }
        FlexType::List(items) => {
            write_u8(writer, FLEX_TYPE_TAG_OFFSET + FlexTypeEnum::List as u8)?;
            write_u64(writer, items.len() as u64)?;
            for item in items.iter() {
                write_flex_type(writer, item)?;
            }
        }
        FlexType::Dict(pairs) => {
            write_u8(writer, FLEX_TYPE_TAG_OFFSET + FlexTypeEnum::Dict as u8)?;
            write_u64(writer, pairs.len() as u64)?;
            for (k, v) in pairs.iter() {
                write_flex_type(writer, k)?;
                write_flex_type(writer, v)?;
            }
        }
        FlexType::DateTime(dt) => {
            write_u8(writer, FLEX_TYPE_TAG_OFFSET + FlexTypeEnum::DateTime as u8)?;
            write_flex_datetime(writer, dt)?;
        }
        FlexType::Undefined => {
            write_u8(writer, FLEX_TYPE_TAG_OFFSET + FlexTypeEnum::Undefined as u8)?;
        }
    }
    Ok(())
}

/// Serialize flex_date_time (new format with microseconds).
pub fn write_flex_datetime(writer: &mut (impl Write + ?Sized), dt: &FlexDateTime) -> Result<()> {
    let ts_low = dt.posix_timestamp as u32;
    let ts_high = ((dt.posix_timestamp as u64) >> 32) as u32 & 0x00FF_FFFF;

    // New format: apply LEGACY_TZ_SHIFT to indicate microsecond field follows
    let tz_shifted = if dt.tz_offset_quarter_hours >= 0 {
        (dt.tz_offset_quarter_hours as i32) + LEGACY_TZ_SHIFT
    } else {
        (dt.tz_offset_quarter_hours as i32) - LEGACY_TZ_SHIFT
    };
    let tz_byte = tz_shifted as u8;

    let ts_high_and_tz = ts_high | ((tz_byte as u32) << 24);
    write_u32(writer, ts_low)?;
    write_u32(writer, ts_high_and_tz)?;
    write_u32(writer, dt.microsecond)?;
    Ok(())
}

/// Read a GraphLab-serialized string: 8-byte LE length + raw bytes.
pub fn read_string(reader: &mut (impl Read + ?Sized)) -> Result<String> {
    let len = read_u64(reader)? as usize;
    if len > 256 * 1024 * 1024 {
        return Err(SFrameError::Format(format!(
            "String length {} exceeds sanity limit",
            len
        )));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| SFrameError::Format(format!("Invalid UTF-8: {}", e)))
}

/// Read a GraphLab-serialized Vec<f64>: 8-byte LE length + raw f64 bytes (POD).
pub fn read_vec_f64(reader: &mut (impl Read + ?Sized)) -> Result<Vec<f64>> {
    let len = read_u64(reader)? as usize;
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        result.push(read_f64(reader)?);
    }
    Ok(result)
}

// --- FlexType deserialization ---

/// C++ tag format: tag_byte = 128 + flex_type_enum value.
const FLEX_TYPE_TAG_OFFSET: u8 = 128;

/// Legacy timezone shift constant from C++ flex_date_time.
const LEGACY_TZ_SHIFT: i32 = 25;

/// Deserialize a FlexType from GraphLab archive format.
pub fn read_flex_type(reader: &mut (impl Read + ?Sized)) -> Result<FlexType> {
    let tag = read_u8(reader)?;

    if tag < FLEX_TYPE_TAG_OFFSET {
        return Err(SFrameError::Format(format!(
            "Legacy FlexType tag {} not supported",
            tag
        )));
    }

    let type_id = tag - FLEX_TYPE_TAG_OFFSET;
    let type_enum = FlexTypeEnum::try_from(type_id)?;

    match type_enum {
        FlexTypeEnum::Integer => Ok(FlexType::Integer(read_i64(reader)?)),
        FlexTypeEnum::Float => Ok(FlexType::Float(read_f64(reader)?)),
        FlexTypeEnum::String => Ok(FlexType::String(Arc::from(read_string(reader)?))),
        FlexTypeEnum::Vector => {
            let v = read_vec_f64(reader)?;
            Ok(FlexType::Vector(Arc::from(v)))
        }
        FlexTypeEnum::List => {
            let len = read_u64(reader)? as usize;
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(read_flex_type(reader)?);
            }
            Ok(FlexType::List(Arc::from(items)))
        }
        FlexTypeEnum::Dict => {
            let len = read_u64(reader)? as usize;
            let mut pairs = Vec::with_capacity(len);
            for _ in 0..len {
                let key = read_flex_type(reader)?;
                let value = read_flex_type(reader)?;
                pairs.push((key, value));
            }
            Ok(FlexType::Dict(Arc::from(pairs)))
        }
        FlexTypeEnum::DateTime => {
            let dt = read_flex_datetime(reader)?;
            Ok(FlexType::DateTime(dt))
        }
        FlexTypeEnum::Undefined => Ok(FlexType::Undefined),
    }
}

/// Deserialize flex_date_time.
///
/// C++ struct layout (12 bytes total):
///   bytes 0-3: low 32 bits of posix timestamp
///   bytes 4-6: high 24 bits of posix timestamp
///   byte 7: tz_offset (with legacy shift encoding)
///   bytes 8-11: microsecond (only present in new format)
///
/// Legacy detection: if tz_offset (raw byte 7) is in range
/// (-LEGACY_TZ_SHIFT, LEGACY_TZ_SHIFT), it's old format (no microseconds).
pub fn read_flex_datetime(reader: &mut (impl Read + ?Sized)) -> Result<FlexDateTime> {
    let mut first8 = [0u8; 8];
    reader.read_exact(&mut first8)?;

    let ts_low = u32::from_le_bytes([first8[0], first8[1], first8[2], first8[3]]);
    let ts_high_and_tz = u32::from_le_bytes([first8[4], first8[5], first8[6], first8[7]]);

    let ts_high = ts_high_and_tz & 0x00FF_FFFF;
    let tz_raw = (ts_high_and_tz >> 24) as i8;

    let posix_timestamp = ((ts_high as i64) << 32) | (ts_low as i64);

    let tz_raw_i32 = tz_raw as i32;
    let (tz_offset, microsecond) = if tz_raw_i32 > -LEGACY_TZ_SHIFT && tz_raw_i32 < LEGACY_TZ_SHIFT
    {
        // Old format: no microsecond field
        (tz_raw, 0u32)
    } else {
        // New format: tz has shift applied, microsecond follows
        let tz = if tz_raw_i32 >= LEGACY_TZ_SHIFT {
            (tz_raw_i32 - LEGACY_TZ_SHIFT) as i8
        } else {
            (tz_raw_i32 + LEGACY_TZ_SHIFT) as i8
        };
        let microsecond = read_u32(reader)?;
        (tz, microsecond)
    };

    Ok(FlexDateTime {
        posix_timestamp,
        tz_offset_quarter_hours: tz_offset,
        microsecond,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_u64() {
        let bytes = 42u64.to_le_bytes();
        let mut cursor = Cursor::new(&bytes);
        assert_eq!(read_u64(&mut cursor).unwrap(), 42);
    }

    #[test]
    fn test_read_i64() {
        let bytes = (-1i64).to_le_bytes();
        let mut cursor = Cursor::new(&bytes);
        assert_eq!(read_i64(&mut cursor).unwrap(), -1);
    }

    #[test]
    fn test_read_f64() {
        let bytes = 3.14f64.to_le_bytes();
        let mut cursor = Cursor::new(&bytes);
        let val = read_f64(&mut cursor).unwrap();
        assert!((val - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_string() {
        let s = "hello";
        let mut data = (s.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(s.as_bytes());
        let mut cursor = Cursor::new(&data);
        assert_eq!(read_string(&mut cursor).unwrap(), "hello");
    }

    #[test]
    fn test_read_string_empty() {
        let data = 0u64.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(&data);
        assert_eq!(read_string(&mut cursor).unwrap(), "");
    }

    #[test]
    fn test_read_vec_f64() {
        let values = vec![1.0f64, 2.0, 3.0];
        let mut data = (values.len() as u64).to_le_bytes().to_vec();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut cursor = Cursor::new(&data);
        assert_eq!(read_vec_f64(&mut cursor).unwrap(), values);
    }

    #[test]
    fn test_read_flex_type_integer() {
        let mut data = vec![128u8]; // tag = 128 + 0 (INTEGER)
        data.extend_from_slice(&42i64.to_le_bytes());
        let mut cursor = Cursor::new(&data);
        assert_eq!(read_flex_type(&mut cursor).unwrap(), FlexType::Integer(42));
    }

    #[test]
    fn test_read_flex_type_float() {
        let mut data = vec![129u8]; // tag = 128 + 1 (FLOAT)
        data.extend_from_slice(&3.14f64.to_le_bytes());
        let mut cursor = Cursor::new(&data);
        match read_flex_type(&mut cursor).unwrap() {
            FlexType::Float(f) => assert!((f - 3.14).abs() < f64::EPSILON),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_read_flex_type_string() {
        let mut data = vec![130u8]; // tag = 128 + 2 (STRING)
        let s = "hello";
        data.extend_from_slice(&(s.len() as u64).to_le_bytes());
        data.extend_from_slice(s.as_bytes());
        let mut cursor = Cursor::new(&data);
        assert_eq!(
            read_flex_type(&mut cursor).unwrap(),
            FlexType::String(Arc::from("hello"))
        );
    }

    #[test]
    fn test_read_flex_type_undefined() {
        let data = vec![135u8]; // tag = 128 + 7 (UNDEFINED)
        let mut cursor = Cursor::new(&data);
        assert_eq!(read_flex_type(&mut cursor).unwrap(), FlexType::Undefined);
    }

    #[test]
    fn test_read_flex_type_vector() {
        let mut data = vec![131u8]; // tag = 128 + 3 (VECTOR)
        let vals = vec![1.0f64, 2.0, 3.0];
        data.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut cursor = Cursor::new(&data);
        match read_flex_type(&mut cursor).unwrap() {
            FlexType::Vector(v) => assert_eq!(v.as_ref(), &[1.0, 2.0, 3.0]),
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn test_read_flex_type_list() {
        let mut data = vec![132u8]; // tag = 128 + 4 (LIST)
        data.extend_from_slice(&2u64.to_le_bytes()); // list length
        data.push(128); // INTEGER tag
        data.extend_from_slice(&1i64.to_le_bytes());
        data.push(130); // STRING tag
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(b"hi");
        let mut cursor = Cursor::new(&data);
        match read_flex_type(&mut cursor).unwrap() {
            FlexType::List(l) => {
                assert_eq!(l.len(), 2);
                assert_eq!(l[0], FlexType::Integer(1));
                assert_eq!(l[1], FlexType::String(Arc::from("hi")));
            }
            other => panic!("Expected List, got {:?}", other),
        }
    }

    #[test]
    fn test_read_flex_type_dict() {
        let mut data = vec![133u8]; // tag = 128 + 5 (DICT)
        data.extend_from_slice(&1u64.to_le_bytes()); // dict length = 1 pair
        // key: String("key")
        data.push(130);
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"key");
        // value: Integer(99)
        data.push(128);
        data.extend_from_slice(&99i64.to_le_bytes());
        let mut cursor = Cursor::new(&data);
        match read_flex_type(&mut cursor).unwrap() {
            FlexType::Dict(d) => {
                assert_eq!(d.len(), 1);
                assert_eq!(d[0].0, FlexType::String(Arc::from("key")));
                assert_eq!(d[0].1, FlexType::Integer(99));
            }
            other => panic!("Expected Dict, got {:?}", other),
        }
    }
}
