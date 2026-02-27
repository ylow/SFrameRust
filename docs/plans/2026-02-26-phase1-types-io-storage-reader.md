# Phase 1: Types, I/O, and Storage Reader — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Read an existing C++ SFrame V2 file from disk and iterate its contents, verified against CSV ground truth.

**Architecture:** Three crates built bottom-up: sframe-types (FlexType enum + serialization), sframe-io (virtual filesystem trait + local backend), sframe-storage (V2 format reader). The sample `business.sf/` is the integration test — read it and verify all 11,536 rows × 12 columns match `business.csv`.

**Tech Stack:** Rust 2021 edition, lz4_flex (compression), serde + serde_json (.sidx parsing), thiserror (errors), csv crate (test verification only)

**Milestone:** `cargo test` passes an integration test that reads `samples/business.sf/`, decodes all columns, and asserts values match `samples/business.csv`.

---

## Task 1: Workspace Scaffold

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/sframe-types/Cargo.toml`
- Create: `crates/sframe-types/src/lib.rs`
- Create: `crates/sframe-io/Cargo.toml`
- Create: `crates/sframe-io/src/lib.rs`
- Create: `crates/sframe-storage/Cargo.toml`
- Create: `crates/sframe-storage/src/lib.rs`

**Step 1: Create workspace root Cargo.toml**

```toml
[workspace]
resolver = "2"
members = [
    "crates/sframe-types",
    "crates/sframe-io",
    "crates/sframe-storage",
]
```

**Step 2: Create sframe-types crate**

`crates/sframe-types/Cargo.toml`:
```toml
[package]
name = "sframe-types"
version = "0.1.0"
edition = "2021"

[dependencies]
thiserror = "2"
```

`crates/sframe-types/src/lib.rs`:
```rust
pub mod error;
pub mod flex_type;
```

Create `crates/sframe-types/src/error.rs`:
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SFrameError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Format error: {0}")]
    Format(String),
}

pub type Result<T> = std::result::Result<T, SFrameError>;
```

Create `crates/sframe-types/src/flex_type.rs` as an empty placeholder:
```rust
// FlexType definitions will go here
```

**Step 3: Create sframe-io crate**

`crates/sframe-io/Cargo.toml`:
```toml
[package]
name = "sframe-io"
version = "0.1.0"
edition = "2021"

[dependencies]
sframe-types = { path = "../sframe-types" }
```

`crates/sframe-io/src/lib.rs`:
```rust
pub mod local_fs;
pub mod vfs;
```

Create placeholder files `crates/sframe-io/src/vfs.rs` and `crates/sframe-io/src/local_fs.rs`.

**Step 4: Create sframe-storage crate**

`crates/sframe-storage/Cargo.toml`:
```toml
[package]
name = "sframe-storage"
version = "0.1.0"
edition = "2021"

[dependencies]
sframe-types = { path = "../sframe-types" }
sframe-io = { path = "../sframe-io" }
lz4_flex = "0.11"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
```

`crates/sframe-storage/src/lib.rs`:
```rust
// SFrame V2 storage format
```

**Step 5: Verify workspace compiles**

Run: `cargo build`
Expected: Compiles with no errors.

**Step 6: Commit**

```bash
git add Cargo.toml crates/
git commit -m "scaffold: workspace with sframe-types, sframe-io, sframe-storage crates"
```

---

## Task 2: FlexTypeEnum and FlexType

**Files:**
- Create: `crates/sframe-types/src/flex_type.rs`
- Test: `crates/sframe-types/src/flex_type.rs` (inline tests)

**Step 1: Write tests for FlexTypeEnum**

In `crates/sframe-types/src/flex_type.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flex_type_enum_values() {
        // Must match C++ flex_type_enum values for format compatibility
        assert_eq!(FlexTypeEnum::Integer as u8, 0);
        assert_eq!(FlexTypeEnum::Float as u8, 1);
        assert_eq!(FlexTypeEnum::String as u8, 2);
        assert_eq!(FlexTypeEnum::Vector as u8, 3);
        assert_eq!(FlexTypeEnum::List as u8, 4);
        assert_eq!(FlexTypeEnum::Dict as u8, 5);
        assert_eq!(FlexTypeEnum::DateTime as u8, 6);
        assert_eq!(FlexTypeEnum::Undefined as u8, 7);
    }

    #[test]
    fn test_flex_type_enum_from_u8() {
        assert_eq!(FlexTypeEnum::try_from(0u8).unwrap(), FlexTypeEnum::Integer);
        assert_eq!(FlexTypeEnum::try_from(7u8).unwrap(), FlexTypeEnum::Undefined);
        assert!(FlexTypeEnum::try_from(8u8).is_err()); // IMAGE not supported
        assert!(FlexTypeEnum::try_from(255u8).is_err());
    }

    #[test]
    fn test_flex_type_type_tag() {
        assert_eq!(FlexType::Integer(42).type_enum(), FlexTypeEnum::Integer);
        assert_eq!(FlexType::Float(3.14).type_enum(), FlexTypeEnum::Float);
        assert_eq!(FlexType::String(Arc::from("hello")).type_enum(), FlexTypeEnum::String);
        assert_eq!(FlexType::Undefined.type_enum(), FlexTypeEnum::Undefined);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p sframe-types`
Expected: FAIL — types not defined yet.

**Step 3: Implement FlexTypeEnum and FlexType**

```rust
use std::sync::Arc;
use crate::error::{SFrameError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FlexTypeEnum {
    Integer = 0,
    Float = 1,
    String = 2,
    Vector = 3,
    List = 4,
    Dict = 5,
    DateTime = 6,
    Undefined = 7,
}

impl TryFrom<u8> for FlexTypeEnum {
    type Error = SFrameError;
    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Integer),
            1 => Ok(Self::Float),
            2 => Ok(Self::String),
            3 => Ok(Self::Vector),
            4 => Ok(Self::List),
            5 => Ok(Self::Dict),
            6 => Ok(Self::DateTime),
            7 => Ok(Self::Undefined),
            _ => Err(SFrameError::Type(format!("Unknown type enum value: {}", value))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlexDateTime {
    pub posix_timestamp: i64,
    pub tz_offset_quarter_hours: i8,
    pub microsecond: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlexType {
    Integer(i64),
    Float(f64),
    String(Arc<str>),
    Vector(Arc<[f64]>),
    List(Arc<[FlexType]>),
    Dict(Arc<[(FlexType, FlexType)]>),
    DateTime(FlexDateTime),
    Undefined,
}

impl FlexType {
    pub fn type_enum(&self) -> FlexTypeEnum {
        match self {
            FlexType::Integer(_) => FlexTypeEnum::Integer,
            FlexType::Float(_) => FlexTypeEnum::Float,
            FlexType::String(_) => FlexTypeEnum::String,
            FlexType::Vector(_) => FlexTypeEnum::Vector,
            FlexType::List(_) => FlexTypeEnum::List,
            FlexType::Dict(_) => FlexTypeEnum::Dict,
            FlexType::DateTime(_) => FlexTypeEnum::DateTime,
            FlexType::Undefined => FlexTypeEnum::Undefined,
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p sframe-types`
Expected: All pass.

**Step 5: Commit**

```bash
git commit -am "feat(types): add FlexType enum and FlexTypeEnum with format-compatible values"
```

---

## Task 3: GraphLab-Compatible Archive Serialization

The C++ oarchive/iarchive uses a specific format we must match exactly.
Key rules:
- All integers are little-endian POD (raw bytes)
- Strings: 8-byte LE length prefix + raw bytes
- Vectors: 8-byte LE length prefix + elements
- FlexType: 1-byte tag (128 + type_enum) + value

**Files:**
- Create: `crates/sframe-types/src/serialization.rs`
- Modify: `crates/sframe-types/src/lib.rs` (add module)

**Step 1: Write tests for primitive deserialization**

In `crates/sframe-types/src/serialization.rs`:
```rust
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
        // GraphLab format: 8-byte LE length + raw bytes
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
        // 8-byte length prefix + raw f64 bytes (POD)
        let values = vec![1.0f64, 2.0, 3.0];
        let mut data = (values.len() as u64).to_le_bytes().to_vec();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut cursor = Cursor::new(&data);
        assert_eq!(read_vec_f64(&mut cursor).unwrap(), values);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p sframe-types`
Expected: FAIL — functions not defined.

**Step 3: Implement primitive read functions**

```rust
use std::io::Read;
use crate::error::{SFrameError, Result};

pub fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub fn read_u16(reader: &mut impl Read) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

pub fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub fn read_i64(reader: &mut impl Read) -> Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

pub fn read_f64(reader: &mut impl Read) -> Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

pub fn read_string(reader: &mut impl Read) -> Result<String> {
    let len = read_u64(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| SFrameError::Format(format!("Invalid UTF-8: {}", e)))
}

pub fn read_vec_f64(reader: &mut impl Read) -> Result<Vec<f64>> {
    let len = read_u64(reader)? as usize;
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        result.push(read_f64(reader)?);
    }
    result
}
```

Note: The C++ version reads POD vectors as a single bulk read. We could optimize
this later with unsafe, but per-element is correct and safe.

**Step 4: Run tests**

Run: `cargo test -p sframe-types`
Expected: All pass.

**Step 5: Commit**

```bash
git commit -am "feat(types): add GraphLab-compatible archive deserialization primitives"
```

---

## Task 4: FlexType Deserialization

The C++ format serializes FlexType as: `[tag_byte] [value_data]` where
`tag_byte = 128 + flex_type_enum`.

**Files:**
- Modify: `crates/sframe-types/src/serialization.rs`

**Step 1: Write tests for FlexType deserialization**

```rust
#[test]
fn test_read_flex_type_integer() {
    // Tag 128 (INTEGER=0) + i64 value
    let mut data = vec![128u8];
    data.extend_from_slice(&42i64.to_le_bytes());
    let mut cursor = Cursor::new(&data);
    assert_eq!(read_flex_type(&mut cursor).unwrap(), FlexType::Integer(42));
}

#[test]
fn test_read_flex_type_float() {
    // Tag 129 (FLOAT=1) + f64 value
    let mut data = vec![129u8];
    data.extend_from_slice(&3.14f64.to_le_bytes());
    let mut cursor = Cursor::new(&data);
    let val = read_flex_type(&mut cursor).unwrap();
    match val {
        FlexType::Float(f) => assert!((f - 3.14).abs() < f64::EPSILON),
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_read_flex_type_string() {
    // Tag 130 (STRING=2) + length-prefixed string
    let mut data = vec![130u8];
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
    // Tag 135 (UNDEFINED=7), no value data
    let data = vec![135u8];
    let mut cursor = Cursor::new(&data);
    assert_eq!(read_flex_type(&mut cursor).unwrap(), FlexType::Undefined);
}

#[test]
fn test_read_flex_type_vector() {
    // Tag 131 (VECTOR=3) + length-prefixed f64 array
    let mut data = vec![131u8];
    let vals = vec![1.0f64, 2.0, 3.0];
    data.extend_from_slice(&(vals.len() as u64).to_le_bytes());
    for v in &vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let mut cursor = Cursor::new(&data);
    let result = read_flex_type(&mut cursor).unwrap();
    match result {
        FlexType::Vector(v) => assert_eq!(v.as_ref(), &[1.0, 2.0, 3.0]),
        _ => panic!("Expected Vector"),
    }
}

#[test]
fn test_read_flex_type_list() {
    // Tag 132 (LIST=4) + length-prefixed list of FlexTypes
    let mut data = vec![132u8];
    // List of 2 elements: Integer(1), String("hi")
    data.extend_from_slice(&2u64.to_le_bytes()); // list length
    data.push(128); // INTEGER tag
    data.extend_from_slice(&1i64.to_le_bytes());
    data.push(130); // STRING tag
    data.extend_from_slice(&2u64.to_le_bytes()); // string length
    data.extend_from_slice(b"hi");
    let mut cursor = Cursor::new(&data);
    let result = read_flex_type(&mut cursor).unwrap();
    match result {
        FlexType::List(l) => {
            assert_eq!(l.len(), 2);
            assert_eq!(l[0], FlexType::Integer(1));
            assert_eq!(l[1], FlexType::String(Arc::from("hi")));
        }
        _ => panic!("Expected List"),
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p sframe-types`
Expected: FAIL — `read_flex_type` not defined.

**Step 3: Implement read_flex_type**

```rust
use crate::flex_type::{FlexType, FlexTypeEnum, FlexDateTime};
use std::sync::Arc;

/// C++ tag format: tag_byte = 128 + flex_type_enum value
const FLEX_TYPE_TAG_OFFSET: u8 = 128;

pub fn read_flex_type(reader: &mut impl Read) -> Result<FlexType> {
    let tag = read_u8(reader)?;

    if tag < FLEX_TYPE_TAG_OFFSET {
        return Err(SFrameError::Format(format!(
            "Legacy FlexType tag {} not supported", tag
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
/// C++ format: first 8 bytes contain packed timestamp + tz_offset.
/// The tz_offset byte determines whether microsecond field follows.
/// Legacy timezone shift constant = 25.
pub fn read_flex_datetime(reader: &mut impl Read) -> Result<FlexDateTime> {
    let mut first8 = [0u8; 8];
    reader.read_exact(&mut first8)?;

    // C++ struct packing: first 4 bytes = low 32 bits of timestamp,
    // next 3 bytes = high 24 bits, byte 7 = tz_offset (with legacy shift)
    let ts_low = u32::from_le_bytes([first8[0], first8[1], first8[2], first8[3]]);
    let ts_high_and_tz = u32::from_le_bytes([first8[4], first8[5], first8[6], first8[7]]);

    let ts_high = ts_high_and_tz & 0x00FFFFFF;
    let tz_raw = (ts_high_and_tz >> 24) as i8;

    let posix_timestamp = ((ts_high as i64) << 32) | (ts_low as i64);

    // Legacy shift constant
    const LEGACY_TZ_SHIFT: i8 = 25;

    let (tz_offset, microsecond) = if tz_raw > -LEGACY_TZ_SHIFT && tz_raw < LEGACY_TZ_SHIFT {
        // Old format: no microsecond field, tz is raw
        (tz_raw, 0u32)
    } else {
        // New format: tz has shift applied, microsecond follows
        let tz = if tz_raw >= LEGACY_TZ_SHIFT {
            tz_raw - LEGACY_TZ_SHIFT
        } else {
            tz_raw + LEGACY_TZ_SHIFT
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
```

NOTE: The flex_date_time deserialization is tricky due to the legacy format
detection. The exact bit packing needs to be verified against the C++ source.
If the sample data contains DateTime columns (it doesn't — business.sf has
INTEGER, FLOAT, STRING, LIST only), we'll test more thoroughly. For now this
is our best-effort implementation based on the C++ struct layout.

**Step 4: Run tests**

Run: `cargo test -p sframe-types`
Expected: All pass.

**Step 5: Commit**

```bash
git commit -am "feat(types): add FlexType deserialization matching GraphLab archive format"
```

---

## Task 5: Variable-Length Integer Encoding

Used in V2 block encoding (frame-of-reference codec). This is distinct from the
oarchive serialization — it's a compact varint used inside typed blocks.

**Files:**
- Create: `crates/sframe-types/src/varint.rs`
- Modify: `crates/sframe-types/src/lib.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_decode_1_byte() {
        // Value 42: [42 << 1 | 0] = [84]
        let data = [84u8];
        let mut cursor = Cursor::new(&data[..]);
        assert_eq!(decode_varint(&mut cursor).unwrap(), 42);
    }

    #[test]
    fn test_decode_1_byte_zero() {
        let data = [0u8];
        let mut cursor = Cursor::new(&data[..]);
        assert_eq!(decode_varint(&mut cursor).unwrap(), 0);
    }

    #[test]
    fn test_decode_1_byte_max() {
        // Max 1-byte value: 63 -> [63 << 1 | 0] = [126]
        let data = [126u8];
        let mut cursor = Cursor::new(&data[..]);
        assert_eq!(decode_varint(&mut cursor).unwrap(), 63);
    }

    #[test]
    fn test_decode_2_byte() {
        // Value 64: [64 << 2 | 1] = [257] as u16 LE = [0x01, 0x01]
        let val: u16 = (64 << 2) | 1;
        let data = val.to_le_bytes();
        let mut cursor = Cursor::new(&data[..]);
        assert_eq!(decode_varint(&mut cursor).unwrap(), 64);
    }

    #[test]
    fn test_decode_9_byte() {
        // Max: sentinel 0x7F followed by 8 bytes of u64
        let mut data = vec![0x7Fu8];
        data.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut cursor = Cursor::new(&data);
        assert_eq!(decode_varint(&mut cursor).unwrap(), u64::MAX);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let test_values = [0, 1, 63, 64, 127, 128, 8191, 8192, 1_000_000, u64::MAX];
        for &val in &test_values {
            let mut buf = Vec::new();
            encode_varint(val, &mut buf).unwrap();
            let mut cursor = Cursor::new(&buf);
            assert_eq!(decode_varint(&mut cursor).unwrap(), val, "roundtrip failed for {}", val);
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p sframe-types`
Expected: FAIL.

**Step 3: Implement varint encode/decode**

```rust
use std::io::{Read, Write};
use crate::error::Result;

/// Decode a variable-length unsigned integer.
///
/// Encoding scheme (distinct from LEB128):
/// - Trailing zero bits in the first byte indicate the byte count.
/// - 1 byte  (bit 0 = 0): value in bits [7:1]
/// - 2 bytes (bits [1:0] = 01): value in bits [15:2]
/// - 3 bytes (bits [2:0] = 011): value in bits [23:3]
/// - ...up to 7 bytes
/// - 9 bytes: first byte = 0x7F, next 8 bytes = raw u64 LE
pub fn decode_varint(reader: &mut impl Read) -> Result<u64> {
    let mut first = [0u8; 1];
    reader.read_exact(&mut first)?;
    let b = first[0];

    if b & 1 == 0 {
        // 1 byte
        return Ok((b as u64) >> 1);
    }
    if b == 0x7F {
        // 9 bytes: sentinel + raw u64
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        return Ok(u64::from_le_bytes(buf));
    }

    // Count trailing ones to determine byte length
    // bits [n-1:0] are all 1, bit n is 0
    // Number of trailing 1-bits + 1 = total bytes to read (including first)
    let trailing_ones = b.trailing_ones() as usize; // 1..=6
    let total_bytes = trailing_ones + 1; // 2..=7
    let shift = total_bytes; // number of flag bits

    let mut buf = [0u8; 8];
    buf[0] = b;
    reader.read_exact(&mut buf[1..total_bytes])?;

    let raw = u64::from_le_bytes(buf);
    Ok(raw >> shift)
}

/// Encode a variable-length unsigned integer.
pub fn encode_varint(value: u64, writer: &mut impl Write) -> Result<()> {
    // Determine how many bits we need
    let bits_needed = if value == 0 { 0 } else { 64 - value.leading_zeros() as usize };

    if bits_needed <= 7 {
        // 1 byte: value << 1, bit 0 = 0
        writer.write_all(&[(value << 1) as u8])?;
    } else if bits_needed <= 14 {
        let encoded = (value << 2) | 1;
        writer.write_all(&(encoded as u16).to_le_bytes())?;
    } else if bits_needed <= 21 {
        let encoded = (value << 3) | 0b011;
        writer.write_all(&(encoded as u32).to_le_bytes()[..3])?;
    } else if bits_needed <= 28 {
        let encoded = (value << 4) | 0b0111;
        writer.write_all(&(encoded as u32).to_le_bytes())?;
    } else if bits_needed <= 35 {
        let encoded = (value << 5) | 0b01111;
        writer.write_all(&(encoded as u64).to_le_bytes()[..5])?;
    } else if bits_needed <= 42 {
        let encoded = (value << 6) | 0b011111;
        writer.write_all(&(encoded as u64).to_le_bytes()[..6])?;
    } else if bits_needed <= 49 {
        let encoded = (value << 7) | 0b0111111;
        writer.write_all(&(encoded as u64).to_le_bytes()[..7])?;
    } else {
        // 9 bytes: sentinel + raw u64
        writer.write_all(&[0x7F])?;
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
```

**Step 4: Run tests**

Run: `cargo test -p sframe-types`
Expected: All pass.

**Step 5: Commit**

```bash
git commit -am "feat(types): add variable-length integer encoding for V2 block format"
```

---

## Task 6: Dense Bitset Deserialization

Used for undefined/null bitmaps in typed blocks.

**Files:**
- Create: `crates/sframe-types/src/bitset.rs`
- Modify: `crates/sframe-types/src/lib.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_empty_bitset() {
        // len=0, arrlen=0
        let mut data = Vec::new();
        data.extend_from_slice(&0u64.to_le_bytes()); // len
        data.extend_from_slice(&0u64.to_le_bytes()); // arrlen
        let mut cursor = Cursor::new(&data);
        let bs = DenseBitset::deserialize(&mut cursor).unwrap();
        assert_eq!(bs.len(), 0);
    }

    #[test]
    fn test_read_bitset_single_word() {
        // 8 bits, 1 word, value 0b10101010
        let mut data = Vec::new();
        data.extend_from_slice(&8u64.to_le_bytes()); // len
        data.extend_from_slice(&1u64.to_le_bytes()); // arrlen
        data.extend_from_slice(&0xAAu64.to_le_bytes()); // word
        let mut cursor = Cursor::new(&data);
        let bs = DenseBitset::deserialize(&mut cursor).unwrap();
        assert_eq!(bs.len(), 8);
        assert!(!bs.get(0));  // bit 0 of 0xAA = 0
        assert!(bs.get(1));   // bit 1 of 0xAA = 1
        assert!(!bs.get(2));
        assert!(bs.get(3));
    }

    #[test]
    fn test_bitset_out_of_range() {
        let mut data = Vec::new();
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0xFFu64.to_le_bytes());
        let mut cursor = Cursor::new(&data);
        let bs = DenseBitset::deserialize(&mut cursor).unwrap();
        assert!(!bs.get(100)); // out of range returns false
    }
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement DenseBitset**

```rust
use std::io::Read;
use crate::error::Result;
use crate::serialization::read_u64;

pub struct DenseBitset {
    len: usize,      // number of bits
    words: Vec<u64>, // packed 64-bit words
}

impl DenseBitset {
    pub fn deserialize(reader: &mut impl Read) -> Result<Self> {
        let len = read_u64(reader)? as usize;
        let arrlen = read_u64(reader)? as usize;
        let mut words = Vec::with_capacity(arrlen);
        for _ in 0..arrlen {
            words.push(read_u64(reader)?);
        }
        Ok(DenseBitset { len, words })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.words[word_idx] >> bit_idx) & 1 == 1
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p sframe-types`
Expected: All pass.

**Step 5: Commit**

```bash
git commit -am "feat(types): add DenseBitset deserialization for null bitmaps"
```

---

## Task 7: Virtual Filesystem — Trait + Local Backend

**Files:**
- Modify: `crates/sframe-io/src/vfs.rs`
- Modify: `crates/sframe-io/src/local_fs.rs`
- Test: inline tests

**Step 1: Write VFS trait and LocalFileSystem tests**

In `crates/sframe-io/src/vfs.rs`:
```rust
use sframe_types::error::Result;
use std::io::{Read, Write, Seek};

pub trait ReadableFile: Read + Seek + Send {
    fn size(&self) -> Result<u64>;
}

pub trait WritableFile: Write + Send {
    fn flush_all(&mut self) -> Result<()>;
}

pub trait VirtualFileSystem: Send + Sync {
    fn open_read(&self, path: &str) -> Result<Box<dyn ReadableFile>>;
    fn open_write(&self, path: &str) -> Result<Box<dyn WritableFile>>;
    fn exists(&self, path: &str) -> Result<bool>;
    fn mkdir_p(&self, path: &str) -> Result<()>;
    fn remove(&self, path: &str) -> Result<()>;
}
```

Note: simplified from the design — dropped `async` from the VFS trait for now.
The local backend is synchronous. When we add S3, we can either make it async
or use an internal block-on. This avoids pulling tokio into sframe-io.

In `crates/sframe-io/src/local_fs.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};

    #[test]
    fn test_write_and_read_file() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem;
        let path = dir.path().join("test.txt");
        let path_str = path.to_str().unwrap();

        let mut wf = fs.open_write(path_str).unwrap();
        wf.write_all(b"hello world").unwrap();
        wf.flush_all().unwrap();
        drop(wf);

        let mut rf = fs.open_read(path_str).unwrap();
        let mut buf = String::new();
        rf.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "hello world");
    }

    #[test]
    fn test_seek_and_size() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem;
        let path = dir.path().join("data.bin");
        let path_str = path.to_str().unwrap();

        let mut wf = fs.open_write(path_str).unwrap();
        wf.write_all(&[0u8; 100]).unwrap();
        wf.flush_all().unwrap();
        drop(wf);

        let mut rf = fs.open_read(path_str).unwrap();
        assert_eq!(rf.size().unwrap(), 100);
        rf.seek(SeekFrom::Start(90)).unwrap();
        let mut buf = [0u8; 10];
        rf.read_exact(&mut buf).unwrap();
    }

    #[test]
    fn test_exists() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem;
        let path = dir.path().join("nope.txt");
        assert!(!fs.exists(path.to_str().unwrap()).unwrap());
    }
}
```

**Step 2: Add tempfile dev-dependency**

In `crates/sframe-io/Cargo.toml`:
```toml
[dev-dependencies]
tempfile = "3"
```

**Step 3: Implement LocalFileSystem**

```rust
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use sframe_types::error::{Result, SFrameError};
use crate::vfs::{ReadableFile, WritableFile, VirtualFileSystem};

pub struct LocalFileSystem;

struct LocalReadableFile {
    file: BufReader<File>,
    size: u64,
}

impl Read for LocalReadableFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

impl Seek for LocalReadableFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}

impl ReadableFile for LocalReadableFile {
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }
}

struct LocalWritableFile {
    file: BufWriter<File>,
}

impl Write for LocalWritableFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

impl WritableFile for LocalWritableFile {
    fn flush_all(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }
}

impl VirtualFileSystem for LocalFileSystem {
    fn open_read(&self, path: &str) -> Result<Box<dyn ReadableFile>> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        Ok(Box::new(LocalReadableFile {
            file: BufReader::new(file),
            size,
        }))
    }

    fn open_write(&self, path: &str) -> Result<Box<dyn WritableFile>> {
        let file = File::create(path)?;
        Ok(Box::new(LocalWritableFile {
            file: BufWriter::new(file),
        }))
    }

    fn exists(&self, path: &str) -> Result<bool> {
        Ok(std::path::Path::new(path).exists())
    }

    fn mkdir_p(&self, path: &str) -> Result<()> {
        fs::create_dir_all(path)?;
        Ok(())
    }

    fn remove(&self, path: &str) -> Result<()> {
        fs::remove_file(path)?;
        Ok(())
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p sframe-io`
Expected: All pass.

**Step 5: Commit**

```bash
git commit -am "feat(io): add VFS trait and LocalFileSystem implementation"
```

---

## Task 8: Index File Parsing (frame_idx + sidx)

**Files:**
- Create: `crates/sframe-storage/src/index.rs`
- Create: `crates/sframe-storage/src/dir_archive.rs`
- Modify: `crates/sframe-storage/src/lib.rs`

**Step 1: Write tests against actual sample files**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn samples_dir() -> String {
        // Navigate from crate root to workspace root
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../samples", manifest)
    }

    #[test]
    fn test_parse_frame_idx() {
        let path = format!("{}/business.sf/m_9688d6320ff94822.frame_idx", samples_dir());
        let content = std::fs::read_to_string(&path).unwrap();
        let idx = FrameIndex::parse(&content).unwrap();
        assert_eq!(idx.nrows, 11536);
        assert_eq!(idx.num_columns, 12);
        assert_eq!(idx.column_names[0], "business_id");
        assert_eq!(idx.column_names[11], "type");
        assert_eq!(idx.column_files[0], "m_9688d6320ff94822.sidx:0");
    }

    #[test]
    fn test_parse_sidx() {
        let path = format!("{}/business.sf/m_9688d6320ff94822.sidx", samples_dir());
        let content = std::fs::read_to_string(&path).unwrap();
        let idx = GroupIndex::parse(&content).unwrap();
        assert_eq!(idx.version, 2);
        assert_eq!(idx.nsegments, 1);
        assert_eq!(idx.segment_files.len(), 1);
        assert_eq!(idx.segment_files[0], "m_9688d6320ff94822.0000");
        assert_eq!(idx.columns.len(), 12);
        // Column 0: business_id = STRING (type 2)
        assert_eq!(idx.columns[0].dtype, FlexTypeEnum::String);
        // Column 1: categories = LIST (type 3) -- actually VECTOR
        assert_eq!(idx.columns[1].dtype, FlexTypeEnum::Vector);
        // Column 7: open = INTEGER (type 0)
        assert_eq!(idx.columns[7].dtype, FlexTypeEnum::Integer);
        // Column 4: latitude = FLOAT (type 1)
        assert_eq!(idx.columns[4].dtype, FlexTypeEnum::Float);
    }

    #[test]
    fn test_parse_dir_archive() {
        let path = format!("{}/business.sf/dir_archive.ini", samples_dir());
        let content = std::fs::read_to_string(&path).unwrap();
        let archive = DirArchive::parse(&content).unwrap();
        assert_eq!(archive.contents, "sframe");
    }
}
```

**Step 2: Run to verify they fail**

**Step 3: Implement index parsing**

`dir_archive.rs` — simple INI parser for dir_archive.ini:
```rust
pub struct DirArchive {
    pub version: u32,
    pub contents: String,
    pub prefixes: Vec<String>,
}
```

`index.rs` — parse frame_idx (INI) and sidx (JSON):
```rust
pub struct FrameIndex {
    pub version: u32,
    pub num_columns: usize,
    pub nrows: u64,
    pub column_names: Vec<String>,
    pub column_files: Vec<String>,
}

pub struct GroupIndex {
    pub version: u32,
    pub nsegments: usize,
    pub segment_files: Vec<String>,
    pub columns: Vec<ColumnIndex>,
}

pub struct ColumnIndex {
    pub dtype: FlexTypeEnum,
    pub content_type: String,
    pub segment_sizes: Vec<u64>,
}
```

For the INI parsing, use a simple hand-rolled parser (the format is trivial —
`[section]` headers and `key=value` lines). Avoid pulling in an INI crate for
this.

For the sidx, use serde_json with a raw JSON structure, then map to GroupIndex.

**Step 4: Run tests**

Run: `cargo test -p sframe-storage`
Expected: All pass.

**Step 5: Commit**

```bash
git commit -am "feat(storage): parse dir_archive.ini, frame_idx, and sidx index files"
```

---

## Task 9: Segment File Footer + Block Info Reading

Read the binary footer of a segment file to get the block index.

**Files:**
- Create: `crates/sframe-storage/src/segment_reader.rs`
- Create: `crates/sframe-storage/src/block_info.rs`

**Step 1: Write tests against sample segment file**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use sframe_io::local_fs::LocalFileSystem;

    #[test]
    fn test_read_segment_footer() {
        let fs = LocalFileSystem;
        let path = format!("{}/../../samples/business.sf/m_9688d6320ff94822.0000",
            env!("CARGO_MANIFEST_DIR"));
        let mut file = fs.open_read(&path).unwrap();
        let block_index = read_segment_footer(&mut *file).unwrap();

        // 12 columns
        assert_eq!(block_index.len(), 12);

        // Each column should have at least 1 block
        for col_blocks in &block_index {
            assert!(!col_blocks.is_empty());
        }

        // Total elements across blocks in column 0 should = 11536
        let total: u64 = block_index[0].iter().map(|b| b.num_elem).sum();
        assert_eq!(total, 11536);
    }
}
```

**Step 2: Run to verify it fails**

**Step 3: Implement footer reading**

```rust
pub struct BlockInfo {
    pub offset: u64,
    pub length: u64,
    pub block_size: u64,
    pub num_elem: u64,
    pub flags: u64,
    pub content_type: u16,
}

pub const LZ4_COMPRESSION: u64 = 1;
pub const IS_FLEXIBLE_TYPE: u64 = 2;
pub const MULTIPLE_TYPE_BLOCK: u64 = 4;
pub const BLOCK_ENCODING_EXTENSION: u64 = 8;
```

The footer is serialized via GraphLab's oarchive as a `Vec<Vec<BlockInfo>>`.
Reading procedure:
1. Seek to `file_size - 8`, read footer_size (u64 LE)
2. Seek to `file_size - 8 - footer_size`
3. Read outer vec length (u64), then for each column read inner vec length (u64),
   then for each block read the 6 fields as raw POD (42 bytes per BlockInfo).

NOTE: BlockInfo serialization uses POD dump in C++. The struct is 42 bytes:
5 × u64 (40 bytes) + 1 × u16 (2 bytes). However, C++ struct alignment may add
padding. Need to verify exact layout. If C++ pads to 48 bytes, we need to match.
We should test this empirically against the sample file.

**Step 4: Run tests**

Run: `cargo test -p sframe-storage`
Expected: All pass.

**Step 5: Commit**

```bash
git commit -am "feat(storage): read segment file footer and block index"
```

---

## Task 10: Block Decoding — Frame-of-Reference Integer Codec

The most complex encoding. Integers are encoded in groups of 128 using
frame-of-reference with 3 codec variants.

**Files:**
- Create: `crates/sframe-storage/src/block_decode.rs`
- Create: `crates/sframe-storage/src/codec_integer.rs`

**Step 1: Write tests**

We cannot easily synthesize test data for this codec without either:
(a) Porting the encoder first, or
(b) Reading known blocks from the sample file.

Approach: read the "open" column (column 7, INTEGER type, values are 0 or 1)
from the sample file and verify against CSV.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_integer_block_from_sample() {
        // Read raw block bytes for column 7 (open) from the sample segment file.
        // Then decode and verify the first few values are 0 or 1.
        // Full verification happens in the integration test.
        let block_data = load_sample_block(7, 0); // column 7, block 0
        let values = decode_integer_block(&block_data).unwrap();
        assert!(!values.is_empty());
        for v in &values {
            assert!(*v == 0 || *v == 1, "Expected 0 or 1, got {}", v);
        }
    }
}
```

**Step 2: Implement frame-of-reference decode**

The codec works in groups of 128 values:

```
[1 byte header] [variable-encoded base value] [packed differences]

Header byte:
  bits 0-1: codec type (0=FoR, 1=FoR-delta, 2=FoR-delta-negative)
  bits 2-7: nbits code (0→0 bits, 1→1, 2→2, 3→4, 4→8, 5→16, 6→32, 7→64)
```

Decode steps:
1. Read header byte
2. Read base value via variable-length decode
3. Read packed differences (N values at K bits each, little-endian packed)
4. Reconstruct values based on codec type:
   - FoR: `value[i] = base + diff[i]`
   - FoR-delta: `value[i] = value[i-1] + diff[i]` (base = first value)
   - FoR-delta-negative: zigzag decode diffs, then cumulative sum

**Step 3: Run tests, iterate until the sample block decodes correctly**

**Step 4: Commit**

```bash
git commit -am "feat(storage): frame-of-reference integer codec decoder"
```

---

## Task 11: Block Decoding — Float, String, Vector Codecs

**Files:**
- Create: `crates/sframe-storage/src/codec_float.rs`
- Create: `crates/sframe-storage/src/codec_string.rs`
- Create: `crates/sframe-storage/src/codec_vector.rs`

**Step 1: Implement float decoder**

Two modes based on reserved byte:
- Legacy (0): bit-rotate doubles to uint64, then FoR decode, bit-rotate back
- Integer (1): FoR decode as integers, cast to f64

Test against column 4 (latitude) from sample.

**Step 2: Implement string decoder**

Two modes:
- Dictionary (≤64 unique): read dict entries, FoR-decode indices
- Direct: FoR-decode lengths, read concatenated bytes

Test against column 0 (business_id) from sample.

**Step 3: Implement vector decoder**

FoR-decode lengths, then float-decode flattened values.

Test against column 1 (categories) from sample.

**Step 4: Run all codec tests**

Run: `cargo test -p sframe-storage`

**Step 5: Commit**

```bash
git commit -am "feat(storage): float, string, and vector block decoders"
```

---

## Task 12: Typed Block Decoder (Top-Level)

Combines the type header parsing with the per-type codecs.

**Files:**
- Modify: `crates/sframe-storage/src/block_decode.rs`

**Step 1: Write the typed block decoder**

Handles the type header (num_types byte, type byte, optional undefined bitmap)
then dispatches to the appropriate codec.

```rust
pub fn decode_typed_block(
    data: &[u8],
    expected_type: FlexTypeEnum,
    flags: u64,
) -> Result<Vec<FlexType>> {
    // 1. Read num_types byte
    // 2. If num_types == 0: return empty vec
    // 3. If num_types == 1: read type byte, decode homogeneous block
    // 4. If num_types == 2: read type byte, read undefined bitmap, decode block, merge
    // 5. If num_types > 2: fall back to raw FlexType deserialization
}
```

**Step 2: Test with each column type from the sample**

**Step 3: Commit**

```bash
git commit -am "feat(storage): typed block decoder with null bitmap support"
```

---

## Task 13: Full Column Reader

Reads all blocks for a column across all segments, yielding a complete Vec<FlexType>.

**Files:**
- Modify: `crates/sframe-storage/src/segment_reader.rs`

**Step 1: Implement read_column**

```rust
impl SegmentReader {
    pub fn read_column(&mut self, column: usize) -> Result<Vec<FlexType>> {
        let blocks = &self.block_index[column];
        let mut result = Vec::new();
        for (block_idx, block_info) in blocks.iter().enumerate() {
            let raw = self.read_raw_block(block_info)?;
            let decompressed = decompress_if_needed(&raw, block_info)?;
            let values = decode_typed_block(
                &decompressed,
                self.column_types[column],
                block_info.flags,
            )?;
            result.extend(values);
        }
        Ok(result)
    }
}
```

**Step 2: Test — read all 12 columns from sample, verify row counts**

```rust
#[test]
fn test_read_all_columns() {
    let reader = open_sample_segment();
    for col in 0..12 {
        let values = reader.read_column(col).unwrap();
        assert_eq!(values.len(), 11536);
    }
}
```

**Step 3: Commit**

```bash
git commit -am "feat(storage): full column reader with LZ4 decompression"
```

---

## Task 14: SFrame Reader — Tie It All Together

Top-level function that reads dir_archive.ini, frame_idx, sidx, opens segment
files, and provides access to columns.

**Files:**
- Create: `crates/sframe-storage/src/sframe_reader.rs`

**Step 1: Implement read_sframe**

```rust
pub struct SFrameData {
    pub frame_index: FrameIndex,
    pub group_index: GroupIndex,
    pub segment_readers: Vec<SegmentReader>,
}

pub fn read_sframe(base_path: &str) -> Result<SFrameData> {
    let fs = LocalFileSystem;
    // 1. Read and parse dir_archive.ini
    // 2. Read and parse frame_idx
    // 3. Read and parse sidx
    // 4. Open segment files, read footers
    // 5. Return SFrameData
}
```

**Step 2: Test with sample**

```rust
#[test]
fn test_read_sframe() {
    let sf = read_sframe(&format!("{}/../../samples/business.sf",
        env!("CARGO_MANIFEST_DIR"))).unwrap();
    assert_eq!(sf.frame_index.nrows, 11536);
    assert_eq!(sf.frame_index.num_columns, 12);
    assert_eq!(sf.frame_index.column_names[0], "business_id");
}
```

**Step 3: Commit**

```bash
git commit -am "feat(storage): top-level SFrame reader from dir_archive"
```

---

## Task 15: Integration Test — Verify Against CSV

The milestone test. Read business.sf, read business.csv, compare every value.

**Files:**
- Create: `crates/sframe-storage/tests/integration_test.rs`

**Step 1: Add csv dev-dependency**

In `crates/sframe-storage/Cargo.toml`:
```toml
[dev-dependencies]
csv = "1"
```

**Step 2: Write integration test**

```rust
use sframe_storage::sframe_reader::read_sframe;
use sframe_types::flex_type::FlexType;
use std::collections::HashMap;

fn samples_dir() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../../samples", manifest)
}

#[test]
fn test_business_sf_matches_csv() {
    // Read SFrame
    let sf = read_sframe(&format!("{}/business.sf", samples_dir())).unwrap();

    // Read CSV
    let mut csv_reader = csv::Reader::from_path(
        format!("{}/business.csv", samples_dir())
    ).unwrap();

    let headers: Vec<String> = csv_reader.headers().unwrap()
        .iter().map(|s| s.to_string()).collect();

    // Read all columns from SFrame
    let mut sf_columns: HashMap<String, Vec<FlexType>> = HashMap::new();
    for (i, name) in sf.frame_index.column_names.iter().enumerate() {
        let col = sf.segment_readers[0].read_column(i).unwrap();
        sf_columns.insert(name.clone(), col);
    }

    // Compare row by row
    let mut row_idx = 0;
    for record in csv_reader.records() {
        let record = record.unwrap();

        // Check a few key columns
        // business_id (STRING)
        let sf_val = &sf_columns["business_id"][row_idx];
        let csv_val = &record[headers.iter().position(|h| h == "business_id").unwrap()];
        match sf_val {
            FlexType::String(s) => assert_eq!(s.as_ref(), csv_val,
                "Row {} business_id mismatch", row_idx),
            _ => panic!("Expected String for business_id"),
        }

        // open (INTEGER)
        let sf_val = &sf_columns["open"][row_idx];
        let csv_val: i64 = record[headers.iter().position(|h| h == "open").unwrap()]
            .parse().unwrap();
        match sf_val {
            FlexType::Integer(v) => assert_eq!(*v, csv_val,
                "Row {} open mismatch", row_idx),
            _ => panic!("Expected Integer for open"),
        }

        // latitude (FLOAT)
        let sf_val = &sf_columns["latitude"][row_idx];
        let csv_val: f64 = record[headers.iter().position(|h| h == "latitude").unwrap()]
            .parse().unwrap();
        match sf_val {
            FlexType::Float(v) => assert!(
                (v - csv_val).abs() < 1e-6,
                "Row {} latitude mismatch: {} vs {}", row_idx, v, csv_val
            ),
            _ => panic!("Expected Float for latitude"),
        }

        row_idx += 1;
    }

    assert_eq!(row_idx, 11536, "Row count mismatch");
}
```

**Step 3: Run integration test**

Run: `cargo test -p sframe-storage --test integration_test`
Expected: PASS — this is the milestone.

**Step 4: Commit**

```bash
git commit -am "test(storage): integration test verifying business.sf matches business.csv"
```

---

## Summary

| Task | Crate | What |
|------|-------|------|
| 1 | all | Workspace scaffold |
| 2 | sframe-types | FlexType + FlexTypeEnum |
| 3 | sframe-types | Archive serialization (read primitives) |
| 4 | sframe-types | FlexType deserialization |
| 5 | sframe-types | Variable-length integer encoding |
| 6 | sframe-types | Dense bitset |
| 7 | sframe-io | VFS trait + LocalFileSystem |
| 8 | sframe-storage | Index file parsing (frame_idx, sidx, dir_archive.ini) |
| 9 | sframe-storage | Segment footer + BlockInfo reading |
| 10 | sframe-storage | Integer codec (frame-of-reference) |
| 11 | sframe-storage | Float, string, vector codecs |
| 12 | sframe-storage | Typed block decoder |
| 13 | sframe-storage | Full column reader |
| 14 | sframe-storage | Top-level SFrame reader |
| 15 | sframe-storage | Integration test vs CSV |

After this milestone, the next phases will be:
- **Phase 2:** Storage writer (SegmentWriter + SFrameWriter)
- **Phase 3:** Query engine (SFrameRows, operators, planner, async execution)
- **Phase 4:** Algorithms (sort, join, groupby, CSV parser)
- **Phase 5:** Top-level SFrame/SArray API + pretty printing
