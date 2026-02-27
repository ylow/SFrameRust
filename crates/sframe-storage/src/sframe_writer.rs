//! Top-level SFrame writer.
//!
//! Writes an SFrame as a dir_archive: dir_archive.ini, frame_idx,
//! sidx, and segment files.

use std::io::{BufWriter, Write};
use std::sync::Arc;

use sframe_io::local_fs::LocalFileSystem;
use sframe_io::vfs::{VirtualFileSystem, WritableFile};
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

/// No-op writer used as a placeholder during segment swaps.
struct NullWriter;

impl Write for NullWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl WritableFile for NullWriter {
    fn flush_all(&mut self) -> Result<()> {
        Ok(())
    }
}

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
    write_sidx_to_fs(
        &sidx_path,
        &[segment_file],
        column_types,
        &[segment_sizes],
    )?;

    // Write .frame_idx (INI)
    let frame_idx_file = format!("{}.frame_idx", data_prefix);
    let frame_idx_path = format!("{}/{}", base_path, frame_idx_file);
    write_frame_idx_to_fs(
        &frame_idx_path,
        column_names,
        &sidx_file,
        nrows,
        1,
        &std::collections::HashMap::new(),
    )?;

    // Write dir_archive.ini
    write_dir_archive_ini_to_fs(base_path, &data_prefix)?;

    // Write empty objects.bin (legacy)
    std::fs::write(format!("{}/objects.bin", base_path), b"")
        .map_err(|e| SFrameError::Io(e))?;

    Ok(())
}

/// Write a single segment file containing all rows.
///
/// Each column is written independently with its own block size. The block
/// size starts from a static type-based estimate and is refined online after
/// each block write using the actual compressed size, targeting ~64KB blocks.
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

    let nrows = rows.len();

    // Write each column independently with adaptive block sizing
    for col in 0..num_columns {
        let bpv = estimate_bytes_per_value(column_types[col]).max(1);
        let mut rows_per_block = (TARGET_BLOCK_SIZE / bpv)
            .max(MIN_ROWS_PER_BLOCK)
            .min(MAX_ROWS_PER_BLOCK);

        let mut total_encoded_bytes: u64 = 0;
        let mut total_encoded_values: u64 = 0;

        let mut row_offset = 0;
        while row_offset < nrows {
            let block_end = (row_offset + rows_per_block).min(nrows);
            let values: Vec<FlexType> = rows[row_offset..block_end]
                .iter()
                .map(|row| row[col].clone())
                .collect();
            let on_disk_bytes = seg_writer.write_column_block(
                col,
                &values,
                column_types[col],
            )?;

            // Update online estimate
            total_encoded_bytes += on_disk_bytes;
            total_encoded_values += (block_end - row_offset) as u64;
            let avg_bpv = total_encoded_bytes as f64 / total_encoded_values as f64;
            if avg_bpv > 0.0 {
                rows_per_block = (TARGET_BLOCK_SIZE as f64 / avg_bpv) as usize;
                rows_per_block = rows_per_block
                    .max(MIN_ROWS_PER_BLOCK)
                    .min(MAX_ROWS_PER_BLOCK);
            }

            row_offset = block_end;
        }
    }

    seg_writer.finish()
}

/// Rough estimate of bytes per value for a given column type, used for block sizing.
fn estimate_bytes_per_value(dtype: FlexTypeEnum) -> usize {
    match dtype {
        FlexTypeEnum::Integer => 8,
        FlexTypeEnum::Float => 8,
        FlexTypeEnum::String => 32,
        FlexTypeEnum::Vector => 64,
        FlexTypeEnum::List => 64,
        FlexTypeEnum::Dict => 64,
        FlexTypeEnum::DateTime => 12,
        FlexTypeEnum::Undefined => 1,
    }
}

/// Build the .sidx JSON content string.
fn build_sidx_content(
    segment_files: &[String],
    column_types: &[FlexTypeEnum],
    all_segment_sizes: &[Vec<u64>],
) -> Result<String> {
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

    serde_json::to_string_pretty(&sidx)
        .map_err(|e| SFrameError::Format(format!("JSON serialization error: {}", e)))
}

/// Build the .frame_idx INI content string.
fn build_frame_idx_content(
    column_names: &[&str],
    sidx_file: &str,
    nrows: u64,
    num_segments: usize,
    metadata: &std::collections::HashMap<String, String>,
) -> String {
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
    for (key, value) in metadata {
        content.push_str(&format!("{}={}\n", key, value));
    }

    content
}

/// Build the dir_archive.ini content string.
fn build_dir_archive_ini_content(data_prefix: &str) -> String {
    format!(
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
    )
}

/// Write .sidx to a VFS path.
fn write_sidx(
    vfs: &dyn VirtualFileSystem,
    path: &str,
    segment_files: &[String],
    column_types: &[FlexTypeEnum],
    all_segment_sizes: &[Vec<u64>],
) -> Result<()> {
    let content = build_sidx_content(segment_files, column_types, all_segment_sizes)?;
    vfs.write_string(path, &content)
}

/// Write .frame_idx to a VFS path.
fn write_frame_idx(
    vfs: &dyn VirtualFileSystem,
    path: &str,
    column_names: &[&str],
    sidx_file: &str,
    nrows: u64,
    num_segments: usize,
    metadata: &std::collections::HashMap<String, String>,
) -> Result<()> {
    let content = build_frame_idx_content(column_names, sidx_file, nrows, num_segments, metadata);
    vfs.write_string(path, &content)
}

/// Write dir_archive.ini to a VFS path.
fn write_dir_archive_ini(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    data_prefix: &str,
) -> Result<()> {
    let content = build_dir_archive_ini_content(data_prefix);
    vfs.write_string(&format!("{}/dir_archive.ini", base_path), &content)
}

// Legacy std::fs versions used by write_sframe standalone function.

fn write_sidx_to_fs(
    path: &str,
    segment_files: &[String],
    column_types: &[FlexTypeEnum],
    all_segment_sizes: &[Vec<u64>],
) -> Result<()> {
    let content = build_sidx_content(segment_files, column_types, all_segment_sizes)?;
    std::fs::write(path, content).map_err(SFrameError::Io)?;
    Ok(())
}

fn write_frame_idx_to_fs(
    path: &str,
    column_names: &[&str],
    sidx_file: &str,
    nrows: u64,
    num_segments: usize,
    metadata: &std::collections::HashMap<String, String>,
) -> Result<()> {
    let content = build_frame_idx_content(column_names, sidx_file, nrows, num_segments, metadata);
    std::fs::write(path, content).map_err(SFrameError::Io)?;
    Ok(())
}

fn write_dir_archive_ini_to_fs(base_path: &str, data_prefix: &str) -> Result<()> {
    let content = build_dir_archive_ini_content(data_prefix);
    std::fs::write(format!("{}/dir_archive.ini", base_path), content)
        .map_err(SFrameError::Io)?;
    Ok(())
}

/// Incremental SFrame writer that accepts column data batch-by-batch.
///
/// Unlike [`write_sframe`] which requires all rows in memory at once, this
/// writer accepts batches incrementally, enabling streaming pipelines that
/// never hold the entire dataset in memory.
///
/// Incoming data is buffered internally and flushed in blocks of
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
    vfs: Arc<dyn VirtualFileSystem>,
    base_path: String,
    column_names: Vec<String>,
    column_types: Vec<FlexTypeEnum>,
    data_prefix: String,
    /// Per-column block sizes, adapted online from actual compressed sizes.
    /// Initialized from a static type-based estimate, then refined after
    /// each block write using cumulative average bytes-per-value.
    per_column_rows_per_block: Vec<usize>,
    rows_per_segment: u64,
    num_columns: usize,

    // Current segment state
    seg_writer: SegmentWriter<Box<dyn WritableFile>>,
    current_segment_idx: usize,
    /// Total rows received (appended to buffers) for the current segment.
    rows_in_current_segment: u64,

    // Accumulated metadata across all finished segments
    segment_files: Vec<String>,
    all_segment_sizes: Vec<Vec<u64>>,

    // Cross-batch buffering (per-column, may have different lengths after flushing)
    column_buffers: Vec<Vec<FlexType>>,
    total_rows: u64,

    // Online block-size estimation: cumulative encoded bytes and values per column
    col_encoded_bytes: Vec<u64>,
    col_encoded_values: Vec<u64>,

    // User metadata
    metadata: std::collections::HashMap<String, String>,
}

impl SFrameWriter {
    /// Create a new writer targeting the given directory (local filesystem).
    pub fn new(
        base_path: &str,
        column_names: &[&str],
        column_types: &[FlexTypeEnum],
    ) -> Result<Self> {
        Self::build(
            Arc::new(LocalFileSystem),
            base_path,
            column_names,
            column_types,
            DEFAULT_ROWS_PER_SEGMENT,
        )
    }

    /// Create a new writer using a specific VFS backend.
    pub fn with_vfs(
        vfs: Arc<dyn VirtualFileSystem>,
        base_path: &str,
        column_names: &[&str],
        column_types: &[FlexTypeEnum],
    ) -> Result<Self> {
        Self::build(vfs, base_path, column_names, column_types, DEFAULT_ROWS_PER_SEGMENT)
    }

    /// Create a new writer with a custom rows-per-segment threshold.
    pub fn with_segment_size(
        base_path: &str,
        column_names: &[&str],
        column_types: &[FlexTypeEnum],
        rows_per_segment: u64,
    ) -> Result<Self> {
        Self::build(
            Arc::new(LocalFileSystem),
            base_path,
            column_names,
            column_types,
            rows_per_segment,
        )
    }

    fn build(
        vfs: Arc<dyn VirtualFileSystem>,
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

        vfs.mkdir_p(base_path)?;

        let per_column_rows_per_block: Vec<usize> = column_types
            .iter()
            .map(|&dtype| {
                let bpv = estimate_bytes_per_value(dtype).max(1);
                (TARGET_BLOCK_SIZE / bpv)
                    .max(MIN_ROWS_PER_BLOCK)
                    .min(MAX_ROWS_PER_BLOCK)
                    .min(rows_per_segment as usize)
            })
            .collect();

        let seg_writer = create_segment_writer_vfs(
            &*vfs,
            base_path,
            &data_prefix,
            0,
            num_columns,
        )?;

        Ok(SFrameWriter {
            vfs,
            base_path: base_path.to_string(),
            column_names: column_names.iter().map(|s| s.to_string()).collect(),
            column_types: column_types.to_vec(),
            data_prefix,
            per_column_rows_per_block,
            rows_per_segment,
            num_columns,
            seg_writer,
            current_segment_idx: 0,
            rows_in_current_segment: 0,
            segment_files: Vec::new(),
            all_segment_sizes: Vec::new(),
            column_buffers: vec![Vec::new(); num_columns],
            total_rows: 0,
            col_encoded_bytes: vec![0; num_columns],
            col_encoded_values: vec![0; num_columns],
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Set a metadata key-value pair to be persisted in the frame_idx file.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Write a batch of column data. Each element of `columns` contains
    /// the values for one column. All columns must have the same length.
    ///
    /// Data is buffered internally. Each column flushes blocks independently
    /// based on its per-column block size (determined by type). When a segment
    /// fills up, all remaining buffers are flushed and a new segment starts.
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

        let mut offset = 0;
        while offset < num_rows {
            // How many rows can the current segment still accept?
            let segment_remaining = (self.rows_per_segment - self.rows_in_current_segment) as usize;
            if segment_remaining == 0 {
                // Segment is full — flush remaining buffers and split
                self.flush_all_buffers()?;
                self.finish_current_segment()?;
                self.start_new_segment()?;
                continue;
            }

            let chunk_size = (num_rows - offset).min(segment_remaining);

            // Append chunk to buffers
            for (buf, col_data) in self.column_buffers.iter_mut().zip(columns.iter()) {
                buf.extend_from_slice(&col_data[offset..offset + chunk_size]);
            }
            self.rows_in_current_segment += chunk_size as u64;
            self.total_rows += chunk_size as u64;
            offset += chunk_size;

            // Flush ready blocks per column
            self.flush_ready_columns()?;
        }

        Ok(())
    }

    /// Flush all complete blocks from each column's buffer independently.
    /// Each column has its own block size, so columns flush at different rates.
    /// After each block write, the per-column block size is updated based on
    /// the actual compressed size (online estimation).
    fn flush_ready_columns(&mut self) -> Result<()> {
        for col_idx in 0..self.num_columns {
            // Re-read block_size each iteration since it may be updated
            while self.column_buffers[col_idx].len() >= self.per_column_rows_per_block[col_idx] {
                let block_size = self.per_column_rows_per_block[col_idx];
                let block: Vec<FlexType> = self.column_buffers[col_idx]
                    .drain(..block_size)
                    .collect();
                let on_disk_bytes = self.seg_writer.write_column_block(
                    col_idx,
                    &block,
                    self.column_types[col_idx],
                )?;
                self.update_block_size(col_idx, block_size as u64, on_disk_bytes);
            }
        }
        Ok(())
    }

    /// Flush all remaining buffered data for every column as partial blocks.
    fn flush_all_buffers(&mut self) -> Result<()> {
        for col_idx in 0..self.num_columns {
            if !self.column_buffers[col_idx].is_empty() {
                let n = self.column_buffers[col_idx].len();
                let block: Vec<FlexType> = self.column_buffers[col_idx].drain(..).collect();
                let on_disk_bytes = self.seg_writer.write_column_block(
                    col_idx,
                    &block,
                    self.column_types[col_idx],
                )?;
                self.update_block_size(col_idx, n as u64, on_disk_bytes);
            }
        }
        Ok(())
    }

    /// Update the per-column block size estimate using the actual compressed
    /// size from the most recent block write (cumulative average).
    fn update_block_size(&mut self, col_idx: usize, num_values: u64, on_disk_bytes: u64) {
        self.col_encoded_bytes[col_idx] += on_disk_bytes;
        self.col_encoded_values[col_idx] += num_values;
        let avg_bpv = self.col_encoded_bytes[col_idx] as f64
            / self.col_encoded_values[col_idx] as f64;
        if avg_bpv > 0.0 {
            let new_rpb = (TARGET_BLOCK_SIZE as f64 / avg_bpv) as usize;
            self.per_column_rows_per_block[col_idx] = new_rpb
                .max(MIN_ROWS_PER_BLOCK)
                .min(MAX_ROWS_PER_BLOCK)
                .min(self.rows_per_segment as usize);
        }
    }

    /// Finish the current segment: write its footer and record its metadata.
    fn finish_current_segment(&mut self) -> Result<()> {
        // Swap in a NullWriter placeholder while we finish the current segment.
        let seg_writer = std::mem::replace(
            &mut self.seg_writer,
            SegmentWriter::new(
                Box::new(NullWriter) as Box<dyn WritableFile>,
                self.num_columns,
            ),
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

        let seg_writer = create_segment_writer_vfs(
            &*self.vfs,
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
    ///
    /// Returns the total number of rows written.
    pub fn finish(mut self) -> Result<u64> {
        // Flush any remaining buffered rows as a partial block
        self.flush_all_buffers()?;

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
            &*self.vfs,
            &sidx_path,
            &self.segment_files,
            &self.column_types,
            &self.all_segment_sizes,
        )?;

        // Write .frame_idx
        let frame_idx_file = format!("{}.frame_idx", self.data_prefix);
        let frame_idx_path = format!("{}/{}", self.base_path, frame_idx_file);
        write_frame_idx(
            &*self.vfs,
            &frame_idx_path,
            &col_names,
            &sidx_file,
            self.total_rows,
            num_segments,
            &self.metadata,
        )?;

        // Write dir_archive.ini
        write_dir_archive_ini(&*self.vfs, &self.base_path, &self.data_prefix)?;

        // Write empty objects.bin (legacy)
        self.vfs.write_string(&format!("{}/objects.bin", self.base_path), "")?;

        Ok(self.total_rows)
    }
}

/// Create a segment file name like "m_xxx.0000".
fn segment_filename(data_prefix: &str, segment_idx: usize) -> String {
    format!("{}.{:04}", data_prefix, segment_idx)
}

/// Create a new SegmentWriter via VFS for the given segment index.
fn create_segment_writer_vfs(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    data_prefix: &str,
    segment_idx: usize,
    num_columns: usize,
) -> Result<SegmentWriter<Box<dyn WritableFile>>> {
    let segment_file = segment_filename(data_prefix, segment_idx);
    let segment_path = format!("{}/{}", base_path, segment_file);
    let file = vfs.open_write(&segment_path)?;
    Ok(SegmentWriter::new(file, num_columns))
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
