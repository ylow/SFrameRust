//! Frame-of-reference integer codec for V2 block format.
//!
//! Integers are encoded in groups of up to 128 values using one of 3 codecs:
//!
//! - **Frame of Reference (FoR):** diff[i] = value[i] - min_value
//! - **FoR Delta:** diff[i] = value[i] - value[i-1] (monotonically increasing)
//! - **FoR Delta Negative:** diff[i] = zigzag(value[i] - value[i-1]) (signed deltas)
//!
//! Each group is encoded as:
//!   [1 byte header] [variable-encoded base value] [packed differences]
//!
//! Header byte (from C++ integer_pack.hpp):
//!   bits 0-1: codec type (0=FoR, 1=FoR-delta, 2=FoR-delta-negative)
//!   bits 2-7: shiftpos = 1 + log2(nbits); if 0, nbits=0 (all identical)
//!             nbits = 1 << (shiftpos - 1) when shiftpos > 0

use std::io::{Cursor, Read, Write};

use sframe_types::error::{Result, SFrameError};
use sframe_types::varint::{decode_varint, encode_varint};

/// Codec type encoded in header bits 0-1.
const CODEC_FOR: u8 = 0;
const CODEC_FOR_DELTA: u8 = 1;
const CODEC_FOR_DELTA_NEGATIVE: u8 = 2;

/// Maximum group size for frame-of-reference encoding.
const FOR_GROUP_SIZE: usize = 128;

/// Decode a frame-of-reference encoded integer stream from a byte slice.
pub fn decode_integers_for(data: &[u8], num_elements: usize) -> Result<Vec<i64>> {
    let mut cursor = Cursor::new(data);
    decode_integers_for_reader(&mut cursor, num_elements)
}

/// Decode a frame-of-reference encoded integer stream from a reader.
/// After return, the reader is positioned just past the consumed data.
pub fn decode_integers_for_reader(
    reader: &mut (impl Read + ?Sized),
    num_elements: usize,
) -> Result<Vec<i64>> {
    let mut result = Vec::with_capacity(num_elements);

    while result.len() < num_elements {
        let remaining = num_elements - result.len();
        let group_size = remaining.min(FOR_GROUP_SIZE);
        decode_group(reader, &mut result, group_size)?;
    }

    Ok(result)
}

/// Decode a single group of up to 128 values.
fn decode_group(
    reader: &mut (impl Read + ?Sized),
    output: &mut Vec<i64>,
    group_size: usize,
) -> Result<()> {
    if group_size == 0 {
        return Ok(());
    }

    // Read header byte
    let mut header_buf = [0u8; 1];
    reader.read_exact(&mut header_buf)?;
    let header = header_buf[0];

    let codec = header & 0x03; // bits 0-1
    let shiftpos = (header >> 2) as usize; // bits 2-7

    // Compute bit width from shiftpos
    let nbits = if shiftpos == 0 {
        0usize
    } else {
        1usize << (shiftpos - 1)
    };

    // Read base value (variable-length encoded)
    // For FoR: base = min_value
    // For FoR-delta / FoR-delta-negative: base = first output value
    let base = decode_varint(reader)?;

    if nbits == 0 {
        // All values are identical
        for _ in 0..group_size {
            output.push(base as i64);
        }
        return Ok(());
    }

    match codec {
        CODEC_FOR => {
            // Read group_size packed differences, then add min_value
            let diffs = read_packed_values(reader, group_size, nbits)?;
            for d in diffs {
                output.push(base.wrapping_add(d) as i64);
            }
        }
        CODEC_FOR_DELTA => {
            // base is first value
            output.push(base as i64);
            if group_size > 1 {
                let diffs = read_packed_values(reader, group_size - 1, nbits)?;
                for d in diffs {
                    let prev = *output.last().unwrap() as u64;
                    output.push(prev.wrapping_add(d) as i64);
                }
            }
        }
        CODEC_FOR_DELTA_NEGATIVE => {
            // base is first value, then zigzag-encoded signed deltas
            output.push(base as i64);
            if group_size > 1 {
                let diffs = read_packed_values(reader, group_size - 1, nbits)?;
                for d in diffs {
                    let signed_delta = zigzag_decode(d);
                    let prev = *output.last().unwrap();
                    output.push(prev.wrapping_add(signed_delta));
                }
            }
        }
        _ => {
            return Err(SFrameError::Format(format!(
                "Unknown codec type: {}",
                codec
            )));
        }
    }

    Ok(())
}

/// Read `count` values packed at `bit_width` bits each from the reader.
/// Values are packed little-endian within bytes.
fn read_packed_values(
    reader: &mut (impl Read + ?Sized),
    count: usize,
    bit_width: usize,
) -> Result<Vec<u64>> {
    if bit_width == 0 || count == 0 {
        return Ok(vec![0u64; count]);
    }

    // Special case for 64-bit: just read raw u64 values
    if bit_width == 64 {
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            result.push(u64::from_le_bytes(buf));
        }
        return Ok(result);
    }

    // Calculate bytes needed
    let total_bits = count * bit_width;
    let total_bytes = (total_bits + 7) / 8;

    let mut packed = vec![0u8; total_bytes];
    reader.read_exact(&mut packed)?;

    let mut result = Vec::with_capacity(count);
    let mask: u64 = (1u64 << bit_width) - 1;

    // C++ pack functions (Duff's device) align values to the MSB side when
    // the count doesn't fill a complete byte group, leaving padding zeros
    // in the LSB positions. We must skip those padding bits when unpacking.
    let pad_bits = if bit_width < 8 {
        (8 - (count * bit_width) % 8) % 8
    } else {
        0
    };

    for i in 0..count {
        let bit_offset = pad_bits + i * bit_width;
        let byte_offset = bit_offset / 8;
        let bit_shift = bit_offset % 8;

        // Read enough bytes to cover the value
        let mut val = 0u64;
        let bytes_needed = ((bit_shift + bit_width) + 7) / 8;

        for j in 0..bytes_needed.min(8) {
            if byte_offset + j < packed.len() {
                val |= (packed[byte_offset + j] as u64) << (j * 8);
            }
        }

        val >>= bit_shift;
        val &= mask;
        result.push(val);
    }

    Ok(result)
}

/// Zigzag decode: maps unsigned to signed.
/// C++ calls this shifted_integer_decode.
/// 0 → 0, 1 → -1, 2 → 1, 3 → -2, 4 → 2, ...
fn zigzag_decode(n: u64) -> i64 {
    ((n >> 1) as i64) ^ -((n & 1) as i64)
}

// ==========================================================================
// Encoding
// ==========================================================================

/// Encode integers using frame-of-reference encoding, writing to the given writer.
pub fn encode_integers_for(writer: &mut (impl Write + ?Sized), values: &[i64]) -> Result<()> {
    let mut offset = 0;
    while offset < values.len() {
        let end = (offset + FOR_GROUP_SIZE).min(values.len());
        encode_group(writer, &values[offset..end])?;
        offset = end;
    }
    Ok(())
}

/// Encode a single group of up to 128 values, choosing the best codec.
fn encode_group(writer: &mut (impl Write + ?Sized), values: &[i64]) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }

    // Reinterpret as u64 for FoR encoding (matching C++ behavior)
    let uvals: Vec<u64> = values.iter().map(|&v| v as u64).collect();

    // Compute metrics for each codec
    let (for_nbits, for_min) = compute_for_metrics(&uvals);
    let (delta_nbits, is_monotonic) = compute_delta_metrics(&uvals);
    let delta_neg_nbits = compute_delta_neg_metrics(values);

    // Select best codec (smallest bit width wins)
    if for_nbits <= delta_nbits && for_nbits <= delta_neg_nbits {
        // FRAME_OF_REFERENCE: base = min, diffs = value - min
        let nbits = round_up_nbits(for_nbits);
        let shiftpos = nbits_to_shiftpos(nbits);
        let header = (shiftpos << 2) as u8 | CODEC_FOR;
        writer.write_all(&[header])?;
        encode_varint(for_min, writer)?;
        if nbits > 0 {
            let diffs: Vec<u64> = uvals.iter().map(|&v| v.wrapping_sub(for_min)).collect();
            write_packed_values(writer, &diffs, nbits)?;
        }
    } else if is_monotonic && delta_nbits <= delta_neg_nbits {
        // FRAME_OF_REFERENCE_DELTA: base = first, diffs = value[i] - value[i-1]
        let nbits = round_up_nbits(delta_nbits);
        let shiftpos = nbits_to_shiftpos(nbits);
        let header = (shiftpos << 2) as u8 | CODEC_FOR_DELTA;
        writer.write_all(&[header])?;
        encode_varint(uvals[0], writer)?;
        if nbits > 0 && uvals.len() > 1 {
            let diffs: Vec<u64> = uvals
                .windows(2)
                .map(|w| w[1].wrapping_sub(w[0]))
                .collect();
            write_packed_values(writer, &diffs, nbits)?;
        }
    } else {
        // FRAME_OF_REFERENCE_DELTA_NEGATIVE: base = first, diffs = zigzag(delta)
        let nbits = round_up_nbits(delta_neg_nbits);
        let shiftpos = nbits_to_shiftpos(nbits);
        let header = (shiftpos << 2) as u8 | CODEC_FOR_DELTA_NEGATIVE;
        writer.write_all(&[header])?;
        encode_varint(uvals[0], writer)?;
        if nbits > 0 && values.len() > 1 {
            let diffs: Vec<u64> = values
                .windows(2)
                .map(|w| zigzag_encode(w[1].wrapping_sub(w[0])))
                .collect();
            write_packed_values(writer, &diffs, nbits)?;
        }
    }

    Ok(())
}

/// Compute bits needed for plain FoR (value - min). Returns (nbits, min_value).
fn compute_for_metrics(values: &[u64]) -> (usize, u64) {
    let min_val = *values.iter().min().unwrap();
    let max_diff = values.iter().map(|&v| v.wrapping_sub(min_val)).max().unwrap();
    (bits_required(max_diff), min_val)
}

/// Compute bits needed for delta encoding. Returns (nbits, is_monotonic).
fn compute_delta_metrics(values: &[u64]) -> (usize, bool) {
    if values.len() <= 1 {
        return (0, true);
    }
    let mut is_monotonic = true;
    let mut max_diff: u64 = 0;
    for w in values.windows(2) {
        if w[1] < w[0] {
            is_monotonic = false;
        }
        let diff = w[1].wrapping_sub(w[0]);
        max_diff = max_diff.max(diff);
    }
    (bits_required(max_diff), is_monotonic)
}

/// Compute bits needed for zigzag delta encoding.
fn compute_delta_neg_metrics(values: &[i64]) -> usize {
    if values.len() <= 1 {
        return 0;
    }
    let mut max_encoded: u64 = 0;
    for w in values.windows(2) {
        let delta = w[1].wrapping_sub(w[0]);
        let encoded = zigzag_encode(delta);
        max_encoded = max_encoded.max(encoded);
    }
    bits_required(max_encoded)
}

/// Number of bits needed to represent a value.
fn bits_required(val: u64) -> usize {
    if val == 0 {
        0
    } else {
        64 - val.leading_zeros() as usize
    }
}

/// Round up to the nearest power-of-2 bit width: 0, 1, 2, 4, 8, 16, 32, 64.
fn round_up_nbits(nbits: usize) -> usize {
    match nbits {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 4,
        5..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

/// Convert nbits to shiftpos for the header byte.
fn nbits_to_shiftpos(nbits: usize) -> usize {
    match nbits {
        0 => 0,
        1 => 1,
        2 => 2,
        4 => 3,
        8 => 4,
        16 => 5,
        32 => 6,
        64 => 7,
        _ => unreachable!("nbits must be power of 2: {}", nbits),
    }
}

/// Zigzag encode: maps signed to unsigned.
/// C++ calls this shifted_integer_encode.
/// 0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...
fn zigzag_encode(n: i64) -> u64 {
    ((n >> 63) as u64) ^ ((n as u64) << 1)
}

/// Write `count` values packed at `bit_width` bits each.
/// Uses MSB-first alignment for sub-byte widths matching C++ pack functions.
fn write_packed_values(
    writer: &mut (impl Write + ?Sized),
    values: &[u64],
    bit_width: usize,
) -> Result<()> {
    if bit_width == 0 || values.is_empty() {
        return Ok(());
    }

    // Special case for 64-bit: write raw u64 values
    if bit_width == 64 {
        for &v in values {
            writer.write_all(&v.to_le_bytes())?;
        }
        return Ok(());
    }

    let count = values.len();
    let total_bits = count * bit_width;
    let total_bytes = (total_bits + 7) / 8;

    let mut packed = vec![0u8; total_bytes];
    let mask: u64 = (1u64 << bit_width) - 1;

    // MSB-first padding for sub-byte bit widths (matching C++ Duff's device)
    let pad_bits = if bit_width < 8 {
        (8 - (count * bit_width) % 8) % 8
    } else {
        0
    };

    for (i, &val) in values.iter().enumerate() {
        let v = val & mask;
        let bit_offset = pad_bits + i * bit_width;
        let byte_offset = bit_offset / 8;
        let bit_shift = bit_offset % 8;

        // Write the value bits into the packed buffer
        if bit_width <= 8 {
            packed[byte_offset] |= (v as u8) << bit_shift;
            if bit_shift + bit_width > 8 && byte_offset + 1 < packed.len() {
                packed[byte_offset + 1] |= (v >> (8 - bit_shift)) as u8;
            }
        } else {
            // For multi-byte values, write into the buffer little-endian
            let mut remaining = v;
            let mut shift = bit_shift;
            let mut bo = byte_offset;
            let mut bits_left = bit_width;
            while bits_left > 0 && bo < packed.len() {
                let bits_in_byte = (8 - shift).min(bits_left);
                packed[bo] |= ((remaining & ((1u64 << bits_in_byte) - 1)) as u8) << shift;
                remaining >>= bits_in_byte;
                bits_left -= bits_in_byte;
                shift = 0;
                bo += 1;
            }
        }
    }

    writer.write_all(&packed)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_decode() {
        assert_eq!(zigzag_decode(0), 0);
        assert_eq!(zigzag_decode(1), -1);
        assert_eq!(zigzag_decode(2), 1);
        assert_eq!(zigzag_decode(3), -2);
        assert_eq!(zigzag_decode(4), 2);
    }

    #[test]
    fn test_read_packed_1bit_full_byte() {
        // 8 values at 1 bit each: [1, 0, 1, 0, 1, 1, 0, 0]
        // C++ pack_1 with srclen=8 (full byte, no padding):
        //   value[0]→bit0, value[1]→bit1, ..., value[7]→bit7
        //   byte = 0b00110101 = 0x35
        let data = [0x35u8];
        let mut cursor = Cursor::new(&data[..]);
        let vals = read_packed_values(&mut cursor, 8, 1).unwrap();
        assert_eq!(vals, vec![1, 0, 1, 0, 1, 1, 0, 0]);
    }

    #[test]
    fn test_read_packed_1bit_partial() {
        // 4 values at 1 bit each: [1, 0, 1, 0]
        // C++ pack_1 with srclen=4 (partial byte, 4 pad bits in LSB):
        //   enters at case 4: value[0]→bit4, value[1]→bit5, value[2]→bit6, value[3]→bit7
        //   byte = 0b01010000 = 0x50
        let data = [0x50u8];
        let mut cursor = Cursor::new(&data[..]);
        let vals = read_packed_values(&mut cursor, 4, 1).unwrap();
        assert_eq!(vals, vec![1, 0, 1, 0]);
    }

    #[test]
    fn test_read_packed_8bit() {
        let data = [10u8, 20, 30];
        let mut cursor = Cursor::new(&data[..]);
        let vals = read_packed_values(&mut cursor, 3, 8).unwrap();
        assert_eq!(vals, vec![10, 20, 30]);
    }

    #[test]
    fn test_read_packed_4bit_even() {
        // 4 values at 4 bits: [3, 5, 7, 1] (even count, no padding)
        // C++ pack_4 with srclen=4: normal pairs
        //   byte0 = (5<<4) | 3 = 0x53, byte1 = (1<<4) | 7 = 0x17
        let data = [0x53u8, 0x17];
        let mut cursor = Cursor::new(&data[..]);
        let vals = read_packed_values(&mut cursor, 4, 4).unwrap();
        assert_eq!(vals, vec![3, 5, 7, 1]);
    }

    #[test]
    fn test_read_packed_4bit_odd() {
        // 3 values at 4 bits: [5, 3, 7] (odd count, 4 pad bits in LSB)
        // C++ pack_4 with srclen=3: enters at case 3
        //   byte0 = value[0] << 4 = 0x50 (padding in LOW nibble)
        //   byte1 = (value[2] << 4) | value[1] = 0x73
        let data = [0x50u8, 0x73];
        let mut cursor = Cursor::new(&data[..]);
        let vals = read_packed_values(&mut cursor, 3, 4).unwrap();
        assert_eq!(vals, vec![5, 3, 7]);
    }

    #[test]
    fn test_read_packed_2bit_partial() {
        // 3 values at 2 bits each: [2, 1, 3] (3%4=3, pad=2 bits)
        // C++ pack_2 with srclen=3: enters at case 3
        //   In pack_2, the second half of the byte group handles cases 4-1
        //   srclen%8=3 → case 3: c |= src[0] << 2; case 2: c |= src[1] << 4;
        //   case 1: c |= src[2] << 6; *out = c
        //   byte = (2<<2) | (1<<4) | (3<<6) = 0x08 | 0x10 | 0xC0 = 0xD8
        let data = [0xD8u8];
        let mut cursor = Cursor::new(&data[..]);
        let vals = read_packed_values(&mut cursor, 3, 2).unwrap();
        assert_eq!(vals, vec![2, 1, 3]);
    }

    #[test]
    fn test_zigzag_encode_roundtrip() {
        for v in [-100i64, -2, -1, 0, 1, 2, 100, i64::MIN, i64::MAX] {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v, "zigzag roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_pack_unpack_roundtrip_4bit() {
        let values: Vec<u64> = vec![3, 5, 7, 1, 0, 15, 8, 2];
        let mut buf = Vec::new();
        write_packed_values(&mut buf, &values, 4).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        let decoded = read_packed_values(&mut cursor, values.len(), 4).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_pack_unpack_roundtrip_4bit_odd() {
        let values: Vec<u64> = vec![3, 5, 7];
        let mut buf = Vec::new();
        write_packed_values(&mut buf, &values, 4).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        let decoded = read_packed_values(&mut cursor, values.len(), 4).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_pack_unpack_roundtrip_1bit() {
        let values: Vec<u64> = vec![1, 0, 1, 0, 1];
        let mut buf = Vec::new();
        write_packed_values(&mut buf, &values, 1).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        let decoded = read_packed_values(&mut cursor, values.len(), 1).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_pack_unpack_roundtrip_16bit() {
        let values: Vec<u64> = vec![1000, 2000, 3000, 65535];
        let mut buf = Vec::new();
        write_packed_values(&mut buf, &values, 16).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        let decoded = read_packed_values(&mut cursor, values.len(), 16).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_encode_decode_integers_roundtrip() {
        // Test with various patterns
        let test_cases: Vec<Vec<i64>> = vec![
            vec![0; 200],                                   // all zeros
            (0..300).collect(),                              // monotonic
            vec![100, 50, 200, 10, 90, 300, 5, 250],       // random
            vec![1, -1, 2, -2, 3, -3],                      // negative deltas
            (0..128).collect(),                              // exactly one group
            (0..129).collect(),                              // two groups
            vec![i64::MIN, 0, i64::MAX],                    // extremes
        ];

        for (i, values) in test_cases.iter().enumerate() {
            let mut encoded = Vec::new();
            encode_integers_for(&mut encoded, values).unwrap();
            let decoded = decode_integers_for(&encoded, values.len()).unwrap();
            assert_eq!(&decoded, values, "roundtrip failed for test case {}", i);
        }
    }

    #[test]
    fn test_shiftpos_to_nbits() {
        // Verify the shiftpos → nbits mapping
        // shiftpos=0 → nbits=0
        // shiftpos=1 → nbits=1 (1<<0)
        // shiftpos=2 → nbits=2 (1<<1)
        // shiftpos=3 → nbits=4 (1<<2)
        // shiftpos=4 → nbits=8 (1<<3)
        // shiftpos=5 → nbits=16
        // shiftpos=6 → nbits=32
        // shiftpos=7 → nbits=64
        let expected = [0, 1, 2, 4, 8, 16, 32, 64];
        for (sp, &exp) in expected.iter().enumerate() {
            let nbits = if sp == 0 { 0 } else { 1usize << (sp - 1) };
            assert_eq!(nbits, exp, "shiftpos {} → nbits {}, expected {}", sp, nbits, exp);
        }
    }
}
