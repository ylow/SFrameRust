//! Variable-length unsigned integer encoding used in V2 block format.
//!
//! This is distinct from the oarchive serialization format and from LEB128.
//! The encoding uses trailing bits in the first byte to indicate byte count:
//!
//! - 1 byte  (bit 0 = 0): value in bits [7:1], range 0..63
//! - 2 bytes (bits 1:0 = 01): value in bits [15:2], range 0..8191
//! - 3 bytes (bits 2:0 = 011): value in bits [23:3]
//! - ...up to 7 bytes (bits 5:0 = 0b011111)
//! - 9 bytes: first byte = 0x7F, next 8 bytes = raw u64 LE

use std::io::{Read, Write};

use crate::error::Result;

/// Decode a variable-length unsigned integer.
pub fn decode_varint(reader: &mut (impl Read + ?Sized)) -> Result<u64> {
    let mut first = [0u8; 1];
    reader.read_exact(&mut first)?;
    let b = first[0];

    if b & 1 == 0 {
        // 1 byte: value in bits [7:1]
        return Ok((b as u64) >> 1);
    }

    if b == 0x7F {
        // 9 bytes: sentinel + raw u64
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        return Ok(u64::from_le_bytes(buf));
    }

    // Count trailing ones to determine byte length.
    // The pattern is: N trailing 1-bits followed by a 0-bit.
    // Total bytes = trailing_ones + 1.
    let trailing_ones = b.trailing_ones() as usize; // 1..=6
    let total_bytes = trailing_ones + 1; // 2..=7
    let shift = total_bytes; // number of flag bits to strip

    let mut buf = [0u8; 8];
    buf[0] = b;
    reader.read_exact(&mut buf[1..total_bytes])?;

    let raw = u64::from_le_bytes(buf);
    Ok(raw >> shift)
}

/// Encode a variable-length unsigned integer.
pub fn encode_varint(value: u64, writer: &mut impl Write) -> Result<()> {
    let bits_needed = if value == 0 {
        0
    } else {
        64 - value.leading_zeros() as usize
    };

    if bits_needed <= 7 {
        writer.write_all(&[(value << 1) as u8])?;
    } else if bits_needed <= 14 {
        let encoded = (value << 2) | 1;
        writer.write_all(&(encoded as u16).to_le_bytes())?;
    } else if bits_needed <= 21 {
        let encoded = (value << 3) | 0b011;
        let bytes = (encoded as u32).to_le_bytes();
        writer.write_all(&bytes[..3])?;
    } else if bits_needed <= 28 {
        let encoded = (value << 4) | 0b0111;
        writer.write_all(&(encoded as u32).to_le_bytes())?;
    } else if bits_needed <= 35 {
        let encoded = (value << 5) | 0b0_1111;
        let bytes = (encoded as u64).to_le_bytes();
        writer.write_all(&bytes[..5])?;
    } else if bits_needed <= 42 {
        let encoded = (value << 6) | 0b01_1111;
        let bytes = (encoded as u64).to_le_bytes();
        writer.write_all(&bytes[..6])?;
    } else if bits_needed <= 49 {
        let encoded = (value << 7) | 0b011_1111;
        let bytes = (encoded as u64).to_le_bytes();
        writer.write_all(&bytes[..7])?;
    } else {
        // 9 bytes: sentinel + raw u64
        writer.write_all(&[0x7F])?;
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_decode_1_byte_zero() {
        let data = [0u8]; // value 0: [0 << 1 | 0] = [0]
        let mut cursor = Cursor::new(&data[..]);
        assert_eq!(decode_varint(&mut cursor).unwrap(), 0);
    }

    #[test]
    fn test_decode_1_byte() {
        // Value 42: [42 << 1 | 0] = [84]
        let data = [84u8];
        let mut cursor = Cursor::new(&data[..]);
        assert_eq!(decode_varint(&mut cursor).unwrap(), 42);
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
        // Value 64: [64 << 2 | 1] = 257 as u16 LE
        let val: u16 = (64 << 2) | 1;
        let data = val.to_le_bytes();
        let mut cursor = Cursor::new(&data[..]);
        assert_eq!(decode_varint(&mut cursor).unwrap(), 64);
    }

    #[test]
    fn test_decode_2_byte_max() {
        // Max 2-byte value: 8191
        let val: u16 = (8191 << 2) | 1;
        let data = val.to_le_bytes();
        let mut cursor = Cursor::new(&data[..]);
        assert_eq!(decode_varint(&mut cursor).unwrap(), 8191);
    }

    #[test]
    fn test_decode_9_byte() {
        let mut data = vec![0x7Fu8];
        data.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut cursor = Cursor::new(&data);
        assert_eq!(decode_varint(&mut cursor).unwrap(), u64::MAX);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let test_values = [
            0,
            1,
            63,
            64,
            127,
            128,
            8191,
            8192,
            (1 << 21) - 1,
            1 << 21,
            (1 << 28) - 1,
            1 << 28,
            (1 << 35) - 1,
            1 << 35,
            (1 << 42) - 1,
            1 << 42,
            (1 << 49) - 1,
            1 << 49,
            1_000_000,
            u64::MAX / 2,
            u64::MAX,
        ];
        for &val in &test_values {
            let mut buf = Vec::new();
            encode_varint(val, &mut buf).unwrap();
            let mut cursor = Cursor::new(&buf);
            let decoded = decode_varint(&mut cursor).unwrap();
            assert_eq!(decoded, val, "roundtrip failed for {}", val);
        }
    }

    #[test]
    fn test_encoding_sizes() {
        // 1 byte for values 0..=127 (7 bits available in bits [7:1])
        for val in [0u64, 1, 42, 63, 64, 127] {
            let mut buf = Vec::new();
            encode_varint(val, &mut buf).unwrap();
            assert_eq!(buf.len(), 1, "expected 1 byte for {}", val);
        }
        // 2 bytes for values 128..=8191 (14 bits available)
        for val in [128u64, 256, 8191] {
            let mut buf = Vec::new();
            encode_varint(val, &mut buf).unwrap();
            assert_eq!(buf.len(), 2, "expected 2 bytes for {}", val);
        }
        // 9 bytes for large values
        let mut buf = Vec::new();
        encode_varint(u64::MAX, &mut buf).unwrap();
        assert_eq!(buf.len(), 9);
    }
}
