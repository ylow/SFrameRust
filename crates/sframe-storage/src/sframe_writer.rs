//! Top-level SFrame writer.
//!
//! Writes an SFrame as a dir_archive: dir_archive.ini, frame_idx,
//! sidx, and segment files.

use std::io::BufWriter;

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::segment_writer::SegmentWriter;

/// Target block size in bytes (64KB like C++).
const TARGET_BLOCK_SIZE: usize = 64 * 1024;

/// Minimum rows per block.
const MIN_ROWS_PER_BLOCK: usize = 8;

/// Maximum rows per block.
const MAX_ROWS_PER_BLOCK: usize = 256 * 1024;

/// Default number of rows per segment before auto-splitting.
const DEFAULT_ROWS_PER_SEGMENT: u64 = 1_000_000;

/// Write an SFrame to a directory.
///
/// `base_path` is the directory to create (e.g. "output.sf").
/// `column_names` and `column_types` describe the schema.
/// `rows` is an iterator of rows, where each row is a Vec<FlexType> with
/// one value per column.
pub fn write_sframe(
    base_path: &str,
    column_names: &[&str],
    column_types: &[FlexTypeEnum],
    rows: &[Vec<FlexType>],
) -> Result<()> {
    let num_columns = column_names.len();
    if num_columns != column_types.len() {
        return Err(SFrameError::Format(
            "column_names and column_types length mismatch".to_string(),
        ));
    }

    let nrows = rows.len() as u64;

    // Generate a unique hash prefix for file names
    let hash = generate_hash(base_path);
    let data_prefix = format!("m_{}", hash);

    // Create directory
    std::fs::create_dir_all(base_path)
        .map_err(|e| SFrameError::Io(e))?;

    // Write segment file
    let segment_file = format!("{}.0000", data_prefix);
    let segment_path = format!("{}/{}", base_path, segment_file);
    let segment_sizes = write_segment(
        &segment_path,
        num_columns,
        column_types,
        rows,
    )?;

    // Write .sidx (JSON)
    let sidx_file = format!("{}.sidx", data_prefix);
    let sidx_path = format!("{}/{}", base_path, sidx_file);
    write_sidx(
        &sidx_path,
        &[segment_file],
        column_types,
        &[segment_sizes],
    )?;

    // Write .frame_idx (INI)
    let frame_idx_file = format!("{}.frame_idx", data_prefix);
    let frame_idx_path = format!("{}/{}", base_path, frame_idx_file);
    write_frame_idx(
        &frame_idx_path,
        column_names,
        &sidx_file,
        nrows,
        1,
    )?;

    // Write dir_archive.ini
    write_dir_archive_ini(base_path, &data_prefix)?;

    // Write empty objects.bin (legacy)
    std::fs::write(format!("{}/objects.bin", base_path), b"")
        .map_err(|e| SFrameError::Io(e))?;

    Ok(())
}

/// Write a single segment file containing all rows.
fn write_segment(
    path: &str,
    num_columns: usize,
    column_types: &[FlexTypeEnum],
    rows: &[Vec<FlexType>],
) -> Result<Vec<u64>> {
    let file = std::fs::File::create(path).map_err(|e| SFrameError::Io(e))?;
    let buf_writer = BufWriter::new(file);
    let mut seg_writer = SegmentWriter::new(buf_writer, num_columns);

    if rows.is_empty() {
        return seg_writer.finish();
    }

    // Dynamic block sizing: start with a small block, adjust based on encoded size
    let mut rows_per_block = MIN_ROWS_PER_BLOCK.max(16);

    let mut row_offset = 0;
    while row_offset < rows.len() {
        let block_end = (row_offset + rows_per_block).min(rows.len());

        // Write one block per column
        for col in 0..num_columns {
            let values: Vec<FlexType> = rows[row_offset..block_end]
                .iter()
                .map(|row| row[col].clone())
                .collect();
            seg_writer.write_column_block(col, &values, column_types[col])?;
        }

        // Adjust rows_per_block: estimate bytes per row from first block
        if row_offset == 0 && block_end > row_offset {
            rows_per_block = TARGET_BLOCK_SIZE / estimate_row_size(column_types);
            rows_per_block = rows_per_block.max(MIN_ROWS_PER_BLOCK).min(MAX_ROWS_PER_BLOCK);
        }

        row_offset = block_end;
    }

    seg_writer.finish()
}

/// Rough estimate of bytes per row for block sizing.
fn estimate_row_size(column_types: &[FlexTypeEnum]) -> usize {
    let mut size = 0;
    for &t in column_types {
        size += match t {
            FlexTypeEnum::Integer => 8,
            FlexTypeEnum::Float => 8,
            FlexTypeEnum::String => 32,
            FlexTypeEnum::Vector => 64,
            FlexTypeEnum::List => 64,
            FlexTypeEnum::Dict => 64,
            FlexTypeEnum::DateTime => 12,
            FlexTypeEnum::Undefined => 1,
        };
    }
    size.max(1)
}

/// Write the .sidx file (JSON format) with support for multiple segments.
fn write_sidx(
    path: &str,
    segment_files: &[String],
    column_types: &[FlexTypeEnum],
    all_segment_sizes: &[Vec<u64>],
) -> Result<()> {
    use serde_json::{json, Map, Value};

    let num_segments = segment_files.len();

    let mut seg_files_map = Map::new();
    for (i, seg_file) in segment_files.iter().enumerate() {
        seg_files_map.insert(format!("{:04}", i), Value::String(seg_file.clone()));
    }

    let mut columns = Vec::new();
    for (col_idx, &dtype) in column_types.iter().enumerate() {
        let mut seg_sizes_map = Map::new();
        for (seg_idx, sizes) in all_segment_sizes.iter().enumerate() {
            seg_sizes_map.insert(
                format!("{:04}", seg_idx),
                Value::String(sizes[col_idx].to_string()),
            );
        }

        let mut metadata = Map::new();
        metadata.insert("__type__".to_string(), Value::String((dtype as u8).to_string()));

        columns.push(json!({
            "content_type": "",
            "metadata": metadata,
            "segment_sizes": seg_sizes_map,
        }));
    }

    let sidx = json!({
        "sarray": {
            "version": 2,
            "num_segments": num_segments,
        },
        "segment_files": seg_files_map,
        "columns": columns,
    });

    let content = serde_json::to_string_pretty(&sidx)
        .map_err(|e| SFrameError::Format(format!("JSON serialization error: {}", e)))?;
    std::fs::write(path, content).map_err(|e| SFrameError::Io(e))?;

    Ok(())
}

/// Write the .frame_idx file (INI format).
fn write_frame_idx(
    path: &str,
    column_names: &[&str],
    sidx_file: &str,
    nrows: u64,
    num_segments: usize,
) -> Result<()> {
    let mut content = String::new();

    content.push_str("[sframe]\n");
    content.push_str("version=2\n");
    content.push_str(&format!("num_segments={}\n", num_segments));
    content.push_str(&format!("num_columns={}\n", column_names.len()));
    content.push_str(&format!("nrows={}\n", nrows));

    content.push_str("\n[column_names]\n");
    for (i, name) in column_names.iter().enumerate() {
        content.push_str(&format!("{}={}\n", i, name));
    }

    content.push_str("\n[column_files]\n");
    for i in 0..column_names.len() {
        content.push_str(&format!("{}={}:{}\n", i, sidx_file, i));
    }

    content.push_str("\n[metadata]\n");

    std::fs::write(path, content).map_err(|e| SFrameError::Io(e))?;

    Ok(())
}

/// Write dir_archive.ini.
fn write_dir_archive_ini(base_path: &str, data_prefix: &str) -> Result<()> {
    let content = format!(
        "[archive]\n\
         version=1\n\
         num_prefixes=3\n\
         \n\
         [metadata]\n\
         contents=sframe\n\
         \n\
         [prefixes]\n\
         0000=dir_archive.ini\n\
         0001=objects.bin\n\
         0002={}\n",
        data_prefix
    );

    std::fs::write(format!("{}/dir_archive.ini", base_path), content)
        .map_err(|e| SFrameError::Io(e))?;

    Ok(())
}

/// Incremental SFrame writer that accepts column data batch-by-batch.
///
/// Unlike [`write_sframe`] which requires all rows in memory at once, this
/// writer accepts batches incrementally, enabling streaming pipelines that
/// never hold the entire dataset in memory.
///
/// Incoming data is buffered internally and flushed to disk in blocks of
/// `rows_per_block` rows. When a segment reaches `rows_per_segment` rows,
/// it is finalized and a new segment is created automatically.
///
/// # Example
/// ```ignore
/// let mut writer = SFrameWriter::new("output.sf", &["id", "name"], &[Integer, String])?;
/// writer.write_columns(&batch1_columns)?;
/// writer.write_columns(&batch2_columns)?;
/// writer.finish()?;
/// ```
pub struct SFrameWriter {
    base_path: String,
    column_names: Vec<String>,
    column_types: Vec<FlexTypeEnum>,
    data_prefix: String,
    rows_per_block: usize,
    rows_per_segment: u64,
    num_columns: usize,

    // Current segment state
    seg_writer: SegmentWriter<BufWriter<std::fs::File>>,
    current_segment_idx: usize,
    rows_in_current_segment: u64,

    // Accumulated metadata across all finished segments
    segment_files: Vec<String>,
    all_segment_sizes: Vec<Vec<u64>>,

    // Cross-batch buffering
    column_buffers: Vec<Vec<FlexType>>,
    buffered_rows: usize,
    total_rows: u64,
}

impl SFrameWriter {
    /// Create a new writer targeting the given directory.
    pub fn new(
        base_path: &str,
        column_names: &[&str],
        column_types: &[FlexTypeEnum],
    ) -> Result<Self> {
        Self::with_segment_size(base_path, column_names, column_types, DEFAULT_ROWS_PER_SEGMENT)
    }

    /// Create a new writer with a custom rows-per-segment threshold.
    ///
    /// When the current segment accumulates this many rows, it is
    /// finalized and a new segment is created.
    pub fn with_segment_size(
        base_path: &str,
        column_names: &[&str],
        column_types: &[FlexTypeEnum],
        rows_per_segment: u64,
    ) -> Result<Self> {
        let num_columns = column_names.len();
        if num_columns != column_types.len() {
            return Err(SFrameError::Format(
                "column_names and column_types length mismatch".to_string(),
            ));
        }

        let hash = generate_hash(base_path);
        let data_prefix = format!("m_{}", hash);

        std::fs::create_dir_all(base_path).map_err(SFrameError::Io)?;

        let rows_per_block = TARGET_BLOCK_SIZE / estimate_row_size(column_types);
        let rows_per_block = rows_per_block
            .max(MIN_ROWS_PER_BLOCK)
            .min(MAX_ROWS_PER_BLOCK)
            .min(rows_per_segment as usize); // Never exceed segment capacity

        let (seg_writer, _segment_file) =
            create_segment_writer(base_path, &data_prefix, 0, num_columns)?;

        Ok(SFrameWriter {
            base_path: base_path.to_string(),
            column_names: column_names.iter().map(|s| s.to_string()).collect(),
            column_types: column_types.to_vec(),
            data_prefix,
            rows_per_block,
            rows_per_segment,
            num_columns,
            seg_writer,
            current_segment_idx: 0,
            rows_in_current_segment: 0,
            segment_files: Vec::new(),
            all_segment_sizes: Vec::new(),
            column_buffers: vec![Vec::new(); num_columns],
            buffered_rows: 0,
            total_rows: 0,
        })
    }

    /// Write a batch of column data. Each element of `columns` contains
    /// the values for one column. All columns must have the same length.
    ///
    /// Data is buffered internally and flushed in blocks of `rows_per_block`.
    /// Small batches are accumulated; large batches are split into blocks.
    /// When a segment fills up, a new segment is created automatically.
    pub fn write_columns(&mut self, columns: &[Vec<FlexType>]) -> Result<()> {
        if columns.is_empty() {
            return Ok(());
        }

        let num_rows = columns[0].len();
        if num_rows == 0 {
            return Ok(());
        }

        if columns.len() != self.num_columns {
            return Err(SFrameError::Format(format!(
                "Expected {} columns, got {}",
                self.num_columns,
                columns.len()
            )));
        }

        // Append incoming data to internal buffers
        for (buf, col_data) in self.column_buffers.iter_mut().zip(columns.iter()) {
            buf.extend_from_slice(col_data);
        }
        self.buffered_rows += num_rows;
        self.total_rows += num_rows as u64;

        // Flush complete blocks
        self.flush_full_blocks()?;

        Ok(())
    }

    /// Flush all complete blocks from the internal buffers.
    /// Handles segment splitting when the current segment reaches capacity.
    fn flush_full_blocks(&mut self) -> Result<()> {
        while self.buffered_rows >= self.rows_per_block {
            // Check if we need to start a new segment
            if self.rows_in_current_segment >= self.rows_per_segment {
                self.finish_current_segment()?;
                self.start_new_segment()?;
            }

            for col_idx in 0..self.num_columns {
                let block: Vec<FlexType> = self.column_buffers[col_idx]
                    .drain(..self.rows_per_block)
                    .collect();
                self.seg_writer.write_column_block(
                    col_idx,
                    &block,
                    self.column_types[col_idx],
                )?;
            }
            self.buffered_rows -= self.rows_per_block;
            self.rows_in_current_segment += self.rows_per_block as u64;
        }
        Ok(())
    }

    /// Flush any remaining buffered rows as a final partial block.
    fn flush_remaining(&mut self) -> Result<()> {
        if self.buffered_rows > 0 {
            for col_idx in 0..self.num_columns {
                let block: Vec<FlexType> = self.column_buffers[col_idx].drain(..).collect();
                self.seg_writer.write_column_block(
                    col_idx,
                    &block,
                    self.column_types[col_idx],
                )?;
            }
            self.rows_in_current_segment += self.buffered_rows as u64;
            self.buffered_rows = 0;
        }
        Ok(())
    }

    /// Finish the current segment: write its footer and record its metadata.
    fn finish_current_segment(&mut self) -> Result<()> {
        // Take the current seg_writer and finish it.
        // We swap in a dummy that will be replaced by start_new_segment.
        let seg_writer = std::mem::replace(
            &mut self.seg_writer,
            SegmentWriter::new(BufWriter::new(
                tempfile_placeholder()?,
            ), self.num_columns),
        );

        let segment_sizes = seg_writer.finish()?;
        let segment_file = segment_filename(&self.data_prefix, self.current_segment_idx);
        self.segment_files.push(segment_file);
        self.all_segment_sizes.push(segment_sizes);
        Ok(())
    }

    /// Start writing to a new segment.
    fn start_new_segment(&mut self) -> Result<()> {
        self.current_segment_idx += 1;
        self.rows_in_current_segment = 0;

        let (seg_writer, _segment_file) = create_segment_writer(
            &self.base_path,
            &self.data_prefix,
            self.current_segment_idx,
            self.num_columns,
        )?;
        self.seg_writer = seg_writer;
        Ok(())
    }

    /// Finalize the SFrame: flush remaining data, write segment footer
    /// and all metadata files.
    pub fn finish(mut self) -> Result<()> {
        // Flush any remaining buffered rows as a partial block
        self.flush_remaining()?;

        // Finish the current (last) segment
        let segment_sizes = self.seg_writer.finish()?;
        let segment_file = segment_filename(&self.data_prefix, self.current_segment_idx);
        self.segment_files.push(segment_file);
        self.all_segment_sizes.push(segment_sizes);

        let col_names: Vec<&str> = self.column_names.iter().map(|s| s.as_str()).collect();
        let num_segments = self.segment_files.len();

        // Write .sidx
        let sidx_file = format!("{}.sidx", self.data_prefix);
        let sidx_path = format!("{}/{}", self.base_path, sidx_file);
        write_sidx(
            &sidx_path,
            &self.segment_files,
            &self.column_types,
            &self.all_segment_sizes,
        )?;

        // Write .frame_idx
        let frame_idx_file = format!("{}.frame_idx", self.data_prefix);
        let frame_idx_path = format!("{}/{}", self.base_path, frame_idx_file);
        write_frame_idx(
            &frame_idx_path,
            &col_names,
            &sidx_file,
            self.total_rows,
            num_segments,
        )?;

        // Write dir_archive.ini
        write_dir_archive_ini(&self.base_path, &self.data_prefix)?;

        // Write empty objects.bin (legacy)
        std::fs::write(format!("{}/objects.bin", self.base_path), b"")
            .map_err(SFrameError::Io)?;

        Ok(())
    }
}

/// Create a segment file name like "m_xxx.0000".
fn segment_filename(data_prefix: &str, segment_idx: usize) -> String {
    format!("{}.{:04}", data_prefix, segment_idx)
}

/// Create a new SegmentWriter for the given segment index.
fn create_segment_writer(
    base_path: &str,
    data_prefix: &str,
    segment_idx: usize,
    num_columns: usize,
) -> Result<(SegmentWriter<BufWriter<std::fs::File>>, String)> {
    let segment_file = segment_filename(data_prefix, segment_idx);
    let segment_path = format!("{}/{}", base_path, segment_file);
    let file = std::fs::File::create(&segment_path).map_err(SFrameError::Io)?;
    let buf_writer = BufWriter::new(file);
    Ok((SegmentWriter::new(buf_writer, num_columns), segment_file))
}

/// Create a placeholder file for std::mem::replace in finish_current_segment.
/// This file is never written to — it exists only to satisfy the type system.
fn tempfile_placeholder() -> Result<std::fs::File> {
    // Use /dev/null (unix) as a throwaway write target
    #[cfg(unix)]
    {
        std::fs::File::create("/dev/null").map_err(SFrameError::Io)
    }
    #[cfg(not(unix))]
    {
        // On non-unix, create a temp file
        let tmp = std::env::temp_dir().join(".sframe_writer_placeholder");
        std::fs::File::create(&tmp).map_err(SFrameError::Io)
    }
}

/// Generate a deterministic hash prefix from the path.
fn generate_hash(path: &str) -> String {
    // Simple hash - not cryptographic, just for unique file naming
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in path.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    format!("{:016x}", hash)
}
