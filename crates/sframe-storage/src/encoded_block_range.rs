//! Streaming block decoder using genawaiter coroutines.
//!
//! `EncodedBlockRange` holds decompressed block bytes and a generator
//! coroutine that decodes values incrementally in response to `Decode(n)`
//! and `Skip(n)` commands. This avoids materializing the full block when
//! only a sub-range of rows is needed.

use std::future::Future;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::pin::Pin;

use genawaiter::sync::{Co, Gen};
use genawaiter::GeneratorState;

use sframe_types::bitset::DenseBitset;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};
use sframe_types::flex_wrappers::{FlexString, FlexVec};
use sframe_types::serialization::read_flex_type;
use sframe_types::varint::decode_varint;

use crate::block_decode::decode_typed_values;
use crate::block_info::BlockInfo;
use crate::codec_float::unrotate_double_bits;
use crate::codec_integer::{decode_group, decode_integers_for_reader, skip_group};

/// Maximum group size for frame-of-reference encoding.
const FOR_GROUP_SIZE: usize = 128;

/// Commands sent to the decoder generator.
#[derive(Debug, Clone)]
pub enum DecoderCmd {
    /// Bootstrap handshake: sent once at the beginning.
    Start,
    /// Decode the next `n` values.
    Decode(usize),
    /// Skip the next `n` values.
    Skip(usize),
    /// Release the generator (early termination).
    Release,
}

/// Responses yielded by the decoder generator.
#[derive(Debug)]
pub enum DecoderResponse {
    /// Generator is ready for commands.
    Ready,
    /// Decoded values.
    Values(Vec<FlexType>),
    /// Number of values actually skipped.
    Skipped(usize),
}

/// Streaming block decoder backed by a genawaiter coroutine.
pub struct EncodedBlockRange {
    gen: Gen<DecoderResponse, DecoderCmd, Pin<Box<dyn Future<Output = ()> + Send>>>,
    current_row: usize,
    total_rows: usize,
}

impl EncodedBlockRange {
    /// Create a new streaming decoder for a block.
    ///
    /// `data` is the decompressed block bytes. `block_info` provides
    /// metadata (element count, flags, etc.).
    pub fn new(data: Vec<u8>, block_info: BlockInfo) -> Self {
        let total_rows = block_info.num_elem as usize;

        let mut gen: Gen<DecoderResponse, DecoderCmd, Pin<Box<dyn Future<Output = ()> + Send>>> =
            Gen::new(|co| {
                Box::pin(decode_block_stream(co, data, block_info))
                    as Pin<Box<dyn Future<Output = ()> + Send>>
            });

        // Bootstrap handshake: first resume_with value is dropped by genawaiter.
        // The generator runs to its first yield_(Ready) and we receive Ready back.
        let state = gen.resume_with(DecoderCmd::Start);
        debug_assert!(matches!(
            state,
            GeneratorState::Yielded(DecoderResponse::Ready)
        ));

        EncodedBlockRange {
            gen,
            current_row: 0,
            total_rows,
        }
    }

    /// Decode the next `n` values from the block.
    pub fn decode_next(&mut self, n: usize) -> Vec<FlexType> {
        if n == 0 || self.is_exhausted() {
            return Vec::new();
        }
        let n = n.min(self.total_rows - self.current_row);
        let state = self.gen.resume_with(DecoderCmd::Decode(n));
        match state {
            GeneratorState::Yielded(DecoderResponse::Values(values)) => {
                self.current_row += values.len();
                values
            }
            _ => Vec::new(),
        }
    }

    /// Skip the next `n` values without decoding them.
    pub fn skip(&mut self, n: usize) {
        if n == 0 || self.is_exhausted() {
            return;
        }
        let n = n.min(self.total_rows - self.current_row);
        let state = self.gen.resume_with(DecoderCmd::Skip(n));
        if let GeneratorState::Yielded(DecoderResponse::Skipped(actual)) = state {
            self.current_row += actual;
        }
    }

    /// Number of remaining rows in this block.
    pub fn remaining(&self) -> usize {
        self.total_rows - self.current_row
    }

    /// Current row index within this block.
    pub fn current_row(&self) -> usize {
        self.current_row
    }

    /// Total number of rows in this block.
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Whether all rows have been consumed.
    pub fn is_exhausted(&self) -> bool {
        self.current_row >= self.total_rows
    }
}

// ============================================================================
// Generator body
// ============================================================================

/// Main generator entry point. Mirrors the structure of `decode_typed_block`.
///
/// Protocol: the generator yields `Ready`, then the `.await` returns the
/// first real command (genawaiter drops the value from the bootstrap
/// `resume_with`). Each codec streamer receives this first command as a
/// parameter so there is no command/response offset.
async fn decode_block_stream(
    co: Co<DecoderResponse, DecoderCmd>,
    data: Vec<u8>,
    block_info: BlockInfo,
) {
    let num_elements = block_info.num_elem as usize;

    // Bootstrap: yield Ready. The .await returns the first real command
    // (the Start value from the constructor's resume_with is dropped by
    // genawaiter — the first .await receives the NEXT resume_with value).
    let first_cmd = co.yield_(DecoderResponse::Ready).await;

    if num_elements == 0 || data.is_empty() {
        return;
    }

    let num_types = data[0];
    match num_types {
        0 => {}
        1 => {
            let dtype = match FlexTypeEnum::try_from(data[1]) {
                Ok(d) => d,
                Err(_) => return,
            };
            let type_data = &data[2..];
            stream_homogeneous(&co, type_data, num_elements, dtype, &block_info, first_cmd)
                .await;
        }
        2 => {
            let dtype = match FlexTypeEnum::try_from(data[1]) {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut cursor = Cursor::new(&data[2..]);
            let bitmap = match DenseBitset::deserialize(&mut cursor) {
                Ok(b) => b,
                Err(_) => return,
            };
            let pos = 2 + cursor.position() as usize;
            let type_data = &data[pos..];
            let defined_count = (0..num_elements).filter(|i| !bitmap.get(*i)).count();
            stream_with_undefineds(
                &co,
                type_data,
                num_elements,
                defined_count,
                dtype,
                &block_info,
                &bitmap,
                first_cmd,
            )
            .await;
        }
        _ => {
            stream_mixed(&co, &data[1..], num_elements, first_cmd).await;
        }
    }
}

/// Dispatch to the correct codec streamer for a homogeneous block.
async fn stream_homogeneous(
    co: &Co<DecoderResponse, DecoderCmd>,
    data: &[u8],
    num_elements: usize,
    dtype: FlexTypeEnum,
    block_info: &BlockInfo,
    first_cmd: DecoderCmd,
) {
    match dtype {
        FlexTypeEnum::Integer => {
            stream_integers(co, data, num_elements, FlexType::Integer, first_cmd).await;
        }
        FlexTypeEnum::Float => {
            stream_floats(co, data, num_elements, block_info.has_encoding_extension(), first_cmd)
                .await;
        }
        FlexTypeEnum::String => {
            stream_strings(co, data, num_elements, first_cmd).await;
        }
        FlexTypeEnum::Vector => {
            stream_vectors(co, data, num_elements, block_info.has_encoding_extension(), first_cmd)
                .await;
        }
        FlexTypeEnum::Undefined => {
            stream_constant(co, num_elements, first_cmd).await;
        }
        FlexTypeEnum::List | FlexTypeEnum::Dict | FlexTypeEnum::DateTime => {
            let mut cursor = Cursor::new(data);
            let mut buffer = Vec::with_capacity(num_elements);
            for _ in 0..num_elements {
                match read_flex_type(&mut cursor) {
                    Ok(v) => buffer.push(v),
                    Err(_) => break,
                }
            }
            serve_from_buffer(co, buffer, first_cmd).await;
        }
    }
}

// ============================================================================
// Codec streaming functions
// ============================================================================

/// Stream FoR-encoded integers, applying a transform to produce FlexType values.
///
/// Groups of up to 128 are decoded or skipped atomically. A small internal
/// buffer holds any partially-consumed group values.
async fn stream_integers(
    co: &Co<DecoderResponse, DecoderCmd>,
    data: &[u8],
    num_elements: usize,
    transform: impl Fn(i64) -> FlexType,
    first_cmd: DecoderCmd,
) {
    let mut cursor = Cursor::new(data);
    let mut global_pos = 0usize;
    let mut buf: Vec<i64> = Vec::new();
    let mut buf_pos = 0usize;
    let mut stream_pos = 0usize;

    let mut cmd = first_cmd;

    while global_pos < num_elements {
        match cmd {
            DecoderCmd::Decode(n) => {
                let n = n.min(num_elements - global_pos);
                let mut values = Vec::with_capacity(n);
                let mut need = n;

                while need > 0 {
                    if buf_pos < buf.len() {
                        let avail = buf.len() - buf_pos;
                        let take = need.min(avail);
                        for &v in &buf[buf_pos..buf_pos + take] {
                            values.push(transform(v));
                        }
                        buf_pos += take;
                        need -= take;
                        continue;
                    }

                    buf.clear();
                    buf_pos = 0;
                    let stream_left = num_elements - stream_pos;
                    if stream_left == 0 {
                        break;
                    }
                    let group_size = stream_left.min(FOR_GROUP_SIZE);
                    if decode_group(&mut cursor, &mut buf, group_size).is_err() {
                        break;
                    }
                    stream_pos += group_size;
                }

                global_pos += values.len();
                cmd = co.yield_(DecoderResponse::Values(values)).await;
            }
            DecoderCmd::Skip(n) => {
                let n = n.min(num_elements - global_pos);
                let mut remaining = n;

                if buf_pos < buf.len() {
                    let avail = buf.len() - buf_pos;
                    let skip = remaining.min(avail);
                    buf_pos += skip;
                    remaining -= skip;
                }

                while remaining > 0 {
                    let stream_left = num_elements - stream_pos;
                    if stream_left == 0 {
                        break;
                    }
                    let group_size = stream_left.min(FOR_GROUP_SIZE);

                    if remaining >= group_size {
                        if skip_group(&mut cursor, group_size).is_err() {
                            break;
                        }
                        stream_pos += group_size;
                        remaining -= group_size;
                    } else {
                        buf.clear();
                        buf_pos = 0;
                        if decode_group(&mut cursor, &mut buf, group_size).is_err() {
                            break;
                        }
                        stream_pos += group_size;
                        buf_pos = remaining;
                        remaining = 0;
                    }
                }

                let actual = n - remaining;
                global_pos += actual;
                cmd = co.yield_(DecoderResponse::Skipped(actual)).await;
            }
            DecoderCmd::Release => return,
            DecoderCmd::Start => {
                cmd = co.yield_(DecoderResponse::Ready).await;
            }
        }
    }
}

/// Stream float values.
///
/// Handles the encoding extension byte, then delegates to integer streaming
/// with an appropriate transform.
async fn stream_floats(
    co: &Co<DecoderResponse, DecoderCmd>,
    data: &[u8],
    num_elements: usize,
    has_encoding_extension: bool,
    first_cmd: DecoderCmd,
) {
    if has_encoding_extension {
        if data.is_empty() {
            return;
        }
        let reserved = data[0];
        let float_data = &data[1..];
        match reserved {
            0 => {
                stream_integers(co, float_data, num_elements, |v| {
                    let bits = v as u64;
                    FlexType::Float(f64::from_bits(unrotate_double_bits(bits)))
                }, first_cmd)
                .await;
            }
            1 => {
                stream_integers(co, float_data, num_elements, |v| FlexType::Float(v as f64), first_cmd)
                    .await;
            }
            _ => {}
        }
    } else {
        stream_integers(co, data, num_elements, |v| {
            let bits = v as u64;
            FlexType::Float(f64::from_bits(unrotate_double_bits(bits)))
        }, first_cmd)
        .await;
    }
}

/// Stream string values.
///
/// Two sub-paths:
/// - Dictionary: decode dict upfront, delegate to integer streaming with lookup.
/// - Direct: decode lengths upfront, then stream string bodies.
async fn stream_strings(
    co: &Co<DecoderResponse, DecoderCmd>,
    data: &[u8],
    num_elements: usize,
    first_cmd: DecoderCmd,
) {
    if data.is_empty() {
        return;
    }

    let use_dict = data[0] != 0;
    let mut cursor = Cursor::new(&data[1..]);

    if use_dict {
        let dict_len = match decode_varint(&mut cursor) {
            Ok(v) => v as usize,
            Err(_) => return,
        };

        let mut dict: Vec<FlexString> = Vec::with_capacity(dict_len);
        for _ in 0..dict_len {
            let str_len = match decode_varint(&mut cursor) {
                Ok(v) => v as usize,
                Err(_) => break,
            };
            let mut buf = vec![0u8; str_len];
            if cursor.read_exact(&mut buf).is_err() {
                break;
            }
            let s = String::from_utf8_lossy(&buf);
            dict.push(FlexString::from(s.as_ref()));
        }

        let pos = cursor.position() as usize;
        let index_data = &data[1 + pos..];

        let dict_ref = &dict;
        stream_integers(co, index_data, num_elements, move |v| {
            let idx = v as usize;
            if idx < dict_ref.len() {
                FlexType::String(dict_ref[idx].clone())
            } else {
                FlexType::String(FlexString::from(""))
            }
        }, first_cmd)
        .await;
    } else {
        // Direct encoding: decode lengths, then stream bodies.
        let lengths = match decode_integers_for_reader(&mut cursor, num_elements) {
            Ok(v) => v,
            Err(_) => return,
        };

        let body_start = 1 + cursor.position() as usize;
        let body_data = &data[body_start..];
        let mut body_cursor = Cursor::new(body_data);

        let mut pos = 0usize;
        let mut cmd = first_cmd;

        while pos < num_elements {
            match cmd {
                DecoderCmd::Decode(n) => {
                    let n = n.min(num_elements - pos);
                    let mut values = Vec::with_capacity(n);
                    for i in 0..n {
                        let len = lengths[pos + i] as usize;
                        let mut buf = vec![0u8; len];
                        if body_cursor.read_exact(&mut buf).is_err() {
                            break;
                        }
                        let s = String::from_utf8_lossy(&buf);
                        values.push(FlexType::String(FlexString::from(s.as_ref())));
                    }
                    pos += values.len();
                    cmd = co.yield_(DecoderResponse::Values(values)).await;
                }
                DecoderCmd::Skip(n) => {
                    let n = n.min(num_elements - pos);
                    let mut skip_bytes: i64 = 0;
                    for i in 0..n {
                        skip_bytes += lengths[pos + i];
                    }
                    if body_cursor
                        .seek(SeekFrom::Current(skip_bytes))
                        .is_err()
                    {
                        return;
                    }
                    pos += n;
                    cmd = co.yield_(DecoderResponse::Skipped(n)).await;
                }
                DecoderCmd::Release => return,
                DecoderCmd::Start => {
                    cmd = co.yield_(DecoderResponse::Ready).await;
                }
            }
        }
    }
}

/// Stream vector values.
///
/// Decodes lengths and all float values upfront (the flattened FoR stream
/// cannot be partially skipped at vector boundaries), then serves from buffer.
async fn stream_vectors(
    co: &Co<DecoderResponse, DecoderCmd>,
    data: &[u8],
    num_elements: usize,
    has_encoding_extension: bool,
    first_cmd: DecoderCmd,
) {
    if data.is_empty() {
        return;
    }

    let mut cursor = Cursor::new(data);
    let mut reserved = [0u8; 1];
    if cursor.read_exact(&mut reserved).is_err() {
        return;
    }

    let lengths = match decode_integers_for_reader(&mut cursor, num_elements) {
        Ok(v) => v,
        Err(_) => return,
    };

    let values_offset = cursor.position() as usize;
    let values_data = &data[values_offset..];

    let total_values: i64 = lengths.iter().sum();
    if total_values < 0 {
        return;
    }

    let flat_values = match crate::codec_float::decode_floats(
        values_data,
        total_values as usize,
        has_encoding_extension,
    ) {
        Ok(v) => v,
        Err(_) => return,
    };

    let mut buffer: Vec<FlexType> = Vec::with_capacity(num_elements);
    let mut offset = 0usize;
    for &len in &lengths {
        let len = len as usize;
        let end = (offset + len).min(flat_values.len());
        buffer.push(FlexType::Vector(FlexVec::from(&flat_values[offset..end])));
        offset = end;
    }

    serve_from_buffer(co, buffer, first_cmd).await;
}

/// Stream a block with undefineds.
///
/// Decodes all defined values upfront, interleaves with undefineds from the
/// bitmap, then serves from buffer.
// Parameters are tightly coupled to the block decoding protocol; splitting would add complexity.
#[allow(clippy::too_many_arguments)]
async fn stream_with_undefineds(
    co: &Co<DecoderResponse, DecoderCmd>,
    type_data: &[u8],
    num_elements: usize,
    defined_count: usize,
    dtype: FlexTypeEnum,
    block_info: &BlockInfo,
    bitmap: &DenseBitset,
    first_cmd: DecoderCmd,
) {
    let defined_values = match decode_typed_values(type_data, defined_count, dtype, block_info) {
        Ok(v) => v,
        Err(_) => return,
    };

    let mut buffer = Vec::with_capacity(num_elements);
    let mut def_idx = 0;
    for i in 0..num_elements {
        if bitmap.get(i) {
            buffer.push(FlexType::Undefined);
        } else if def_idx < defined_values.len() {
            buffer.push(defined_values[def_idx].clone());
            def_idx += 1;
        } else {
            buffer.push(FlexType::Undefined);
        }
    }

    serve_from_buffer(co, buffer, first_cmd).await;
}

/// Stream mixed-type blocks. Decodes all upfront, serves from buffer.
async fn stream_mixed(
    co: &Co<DecoderResponse, DecoderCmd>,
    data: &[u8],
    num_elements: usize,
    first_cmd: DecoderCmd,
) {
    let mut cursor = Cursor::new(data);
    let mut buffer = Vec::with_capacity(num_elements);
    for _ in 0..num_elements {
        match read_flex_type(&mut cursor) {
            Ok(v) => buffer.push(v),
            Err(_) => break,
        }
    }
    serve_from_buffer(co, buffer, first_cmd).await;
}

/// Stream all-Undefined blocks without any codec.
async fn stream_constant(
    co: &Co<DecoderResponse, DecoderCmd>,
    num_elements: usize,
    first_cmd: DecoderCmd,
) {
    let mut pos = 0usize;
    let mut cmd = first_cmd;

    while pos < num_elements {
        match cmd {
            DecoderCmd::Decode(n) => {
                let n = n.min(num_elements - pos);
                let values = vec![FlexType::Undefined; n];
                pos += n;
                cmd = co.yield_(DecoderResponse::Values(values)).await;
            }
            DecoderCmd::Skip(n) => {
                let n = n.min(num_elements - pos);
                pos += n;
                cmd = co.yield_(DecoderResponse::Skipped(n)).await;
            }
            DecoderCmd::Release => return,
            DecoderCmd::Start => {
                cmd = co.yield_(DecoderResponse::Ready).await;
            }
        }
    }
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Serve pre-decoded values from a buffer in response to commands.
async fn serve_from_buffer(
    co: &Co<DecoderResponse, DecoderCmd>,
    buffer: Vec<FlexType>,
    first_cmd: DecoderCmd,
) {
    let total = buffer.len();
    let mut pos = 0usize;
    let mut cmd = first_cmd;

    while pos < total {
        match cmd {
            DecoderCmd::Decode(n) => {
                let n = n.min(total - pos);
                let values = buffer[pos..pos + n].to_vec();
                pos += n;
                cmd = co.yield_(DecoderResponse::Values(values)).await;
            }
            DecoderCmd::Skip(n) => {
                let n = n.min(total - pos);
                pos += n;
                cmd = co.yield_(DecoderResponse::Skipped(n)).await;
            }
            DecoderCmd::Release => return,
            DecoderCmd::Start => {
                cmd = co.yield_(DecoderResponse::Ready).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_decode::decode_typed_block;
    use crate::segment_reader::SegmentReader;
    use crate::sframe_reader::SFrameReader;

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{manifest}/../../samples")
    }

    /// Helper: get decompressed block data and block info for every block
    /// across all columns in the first segment.
    fn all_blocks() -> Vec<(Vec<u8>, BlockInfo, String)> {
        let mut sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
        let column_names = sf.column_names().to_vec();
        let seg = &mut sf.segment_readers[0];
        let mut blocks = Vec::new();
        for col in 0..seg.num_columns() {
            for blk in 0..seg.num_blocks(col) {
                let info = seg.block_index[col][blk].clone();
                let raw = seg.read_raw_block(&info).unwrap();
                let decompressed = SegmentReader::decompress_block(&raw, &info).unwrap();
                let label = format!("col={}({}) blk={}", col, column_names[col], blk);
                blocks.push((decompressed, info, label));
            }
        }
        blocks
    }

    #[test]
    fn test_full_decode_matches_bulk() {
        for (data, info, label) in all_blocks() {
            let expected = decode_typed_block(&data, &info).unwrap();
            let total = info.num_elem as usize;

            let mut stream = EncodedBlockRange::new(data, info);
            assert_eq!(stream.total_rows(), total, "{label}");

            let got = stream.decode_next(total);
            assert_eq!(got.len(), expected.len(), "length mismatch: {label}");
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                assert_eq!(a, b, "row {i} mismatch: {label}");
            }
            assert!(stream.is_exhausted(), "not exhausted: {label}");
            assert_eq!(stream.remaining(), 0, "remaining != 0: {label}");
        }
    }

    #[test]
    fn test_chunked_decode_matches_bulk() {
        let chunk_size = 100;
        for (data, info, label) in all_blocks() {
            let expected = decode_typed_block(&data, &info).unwrap();
            let total = info.num_elem as usize;

            let mut stream = EncodedBlockRange::new(data, info);
            let mut got = Vec::with_capacity(total);

            while !stream.is_exhausted() {
                got.extend(stream.decode_next(chunk_size));
            }

            assert_eq!(got.len(), expected.len(), "length mismatch: {label}");
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                assert_eq!(a, b, "row {i} mismatch: {label}");
            }
        }
    }

    #[test]
    fn test_skip_then_decode_second_half() {
        for (data, info, label) in all_blocks() {
            let expected = decode_typed_block(&data, &info).unwrap();
            let total = info.num_elem as usize;
            if total == 0 {
                continue;
            }
            let half = total / 2;

            let mut stream = EncodedBlockRange::new(data, info);
            stream.skip(half);
            assert_eq!(stream.current_row(), half, "current_row after skip: {label}");
            assert_eq!(stream.remaining(), total - half, "remaining after skip: {label}");

            let got = stream.decode_next(total - half);
            assert_eq!(got.len(), expected[half..].len(), "length mismatch: {label}");
            for (i, (a, b)) in expected[half..].iter().zip(got.iter()).enumerate() {
                assert_eq!(a, b, "row {i} (global {}) mismatch: {label}", half + i);
            }
            assert!(stream.is_exhausted(), "not exhausted: {label}");
        }
    }

    #[test]
    fn test_skip_all() {
        for (data, info, label) in all_blocks() {
            let total = info.num_elem as usize;
            let mut stream = EncodedBlockRange::new(data, info);
            stream.skip(total);
            assert!(stream.is_exhausted(), "not exhausted after skip all: {label}");
            assert_eq!(stream.remaining(), 0, "remaining != 0: {label}");
            // Decoding after exhaustion should return empty
            let got = stream.decode_next(10);
            assert!(got.is_empty(), "decode after exhaustion non-empty: {label}");
        }
    }

    #[test]
    fn test_interleaved_skip_decode() {
        for (data, info, label) in all_blocks() {
            let expected = decode_typed_block(&data, &info).unwrap();
            let total = info.num_elem as usize;
            if total < 180 {
                continue; // need enough rows for skip 50, decode 10, skip 100, decode 20
            }

            let mut stream = EncodedBlockRange::new(data, info);

            // Skip 50
            stream.skip(50);
            assert_eq!(stream.current_row(), 50, "after skip 50: {label}");

            // Decode 10
            let chunk1 = stream.decode_next(10);
            assert_eq!(chunk1.len(), 10, "chunk1 len: {label}");
            for (i, (a, b)) in expected[50..60].iter().zip(chunk1.iter()).enumerate() {
                assert_eq!(a, b, "chunk1 row {i} (global {}): {label}", 50 + i);
            }
            assert_eq!(stream.current_row(), 60, "after decode 10: {label}");

            // Skip 100
            stream.skip(100);
            assert_eq!(stream.current_row(), 160, "after skip 100: {label}");

            // Decode 20
            let chunk2 = stream.decode_next(20);
            assert_eq!(chunk2.len(), 20, "chunk2 len: {label}");
            for (i, (a, b)) in expected[160..180].iter().zip(chunk2.iter()).enumerate() {
                assert_eq!(a, b, "chunk2 row {i} (global {}): {label}", 160 + i);
            }
            assert_eq!(stream.current_row(), 180, "after decode 20: {label}");
        }
    }

    #[test]
    fn test_decode_zero_returns_empty() {
        for (data, info, label) in all_blocks().into_iter().take(3) {
            let mut stream = EncodedBlockRange::new(data, info);
            let got = stream.decode_next(0);
            assert!(got.is_empty(), "decode(0) non-empty: {label}");
            assert_eq!(stream.current_row(), 0, "current_row moved: {label}");
        }
    }

    #[test]
    fn test_skip_zero_is_noop() {
        for (data, info, label) in all_blocks().into_iter().take(3) {
            let mut stream = EncodedBlockRange::new(data, info);
            stream.skip(0);
            assert_eq!(stream.current_row(), 0, "current_row moved: {label}");
        }
    }

    #[test]
    fn test_decode_more_than_remaining_clamps() {
        for (data, info, label) in all_blocks() {
            let expected = decode_typed_block(&data, &info).unwrap();
            let total = info.num_elem as usize;

            let mut stream = EncodedBlockRange::new(data, info);
            // Request way more than available
            let got = stream.decode_next(total + 1000);
            assert_eq!(got.len(), expected.len(), "should clamp to total: {label}");
            for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                assert_eq!(a, b, "row {i} mismatch: {label}");
            }
            assert!(stream.is_exhausted(), "not exhausted: {label}");
        }
    }

    #[test]
    fn test_column_type_coverage() {
        // Verify we test all expected column types from business.sf
        let sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
        let types: Vec<FlexTypeEnum> = sf.group_index.columns.iter().map(|c| c.dtype).collect();

        let has_int = types.contains(&FlexTypeEnum::Integer);
        let has_float = types.contains(&FlexTypeEnum::Float);
        let has_string = types.contains(&FlexTypeEnum::String);
        let has_vector = types.contains(&FlexTypeEnum::Vector);

        assert!(has_int, "business.sf should have integer columns");
        assert!(has_float, "business.sf should have float columns");
        assert!(has_string, "business.sf should have string columns");
        assert!(has_vector, "business.sf should have vector columns");
    }
}
