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
        &segment_file,
        column_types,
        &segment_sizes,
    )?;

    // Write .frame_idx (INI)
    let frame_idx_file = format!("{}.frame_idx", data_prefix);
    let frame_idx_path = format!("{}/{}", base_path, frame_idx_file);
    write_frame_idx(
        &frame_idx_path,
        column_names,
        &sidx_file,
        nrows,
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
            // After the first block, we could refine, but for simplicity
            // use a fixed size that's reasonable for most data
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

/// Write the .sidx file (JSON format).
fn write_sidx(
    path: &str,
    segment_file: &str,
    column_types: &[FlexTypeEnum],
    segment_sizes: &[u64],
) -> Result<()> {
    use serde_json::{json, Map, Value};

    let mut segment_files = Map::new();
    segment_files.insert("0000".to_string(), Value::String(segment_file.to_string()));

    let mut columns = Vec::new();
    for (i, &dtype) in column_types.iter().enumerate() {
        let mut seg_sizes_map = Map::new();
        seg_sizes_map.insert(
            "0000".to_string(),
            Value::String(segment_sizes[i].to_string()),
        );

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
            "num_segments": 1,
        },
        "segment_files": segment_files,
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
) -> Result<()> {
    let mut content = String::new();

    content.push_str("[sframe]\n");
    content.push_str("version=2\n");
    content.push_str(&format!("num_segments=1\n"));
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
/// `rows_per_block` rows. This means many small batches are coalesced into
/// efficient blocks, and a single large batch is automatically split.
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
    seg_writer: SegmentWriter<BufWriter<std::fs::File>>,
    segment_file: String,
    data_prefix: String,
    total_rows: u64,
    rows_per_block: usize,
    /// Internal column buffers for cross-batch accumulation.
    /// Each inner Vec holds the buffered values for one column.
    column_buffers: Vec<Vec<FlexType>>,
    /// Number of buffered rows (same across all column buffers).
    buffered_rows: usize,
}

impl SFrameWriter {
    /// Create a new writer targeting the given directory.
    pub fn new(
        base_path: &str,
        column_names: &[&str],
        column_types: &[FlexTypeEnum],
    ) -> Result<Self> {
        let num_columns = column_names.len();
        if num_columns != column_types.len() {
            return Err(SFrameError::Format(
                "column_names and column_types length mismatch".to_string(),
            ));
        }

        let hash = generate_hash(base_path);
        let data_prefix = format!("m_{}", hash);
        let segment_file = format!("{}.0000", data_prefix);
        let segment_path = format!("{}/{}", base_path, segment_file);

        std::fs::create_dir_all(base_path).map_err(SFrameError::Io)?;

        let file = std::fs::File::create(&segment_path).map_err(SFrameError::Io)?;
        let buf_writer = BufWriter::new(file);
        let seg_writer = SegmentWriter::new(buf_writer, num_columns);

        let rows_per_block = TARGET_BLOCK_SIZE / estimate_row_size(column_types);
        let rows_per_block = rows_per_block.max(MIN_ROWS_PER_BLOCK).min(MAX_ROWS_PER_BLOCK);

        let column_buffers = vec![Vec::new(); num_columns];

        Ok(SFrameWriter {
            base_path: base_path.to_string(),
            column_names: column_names.iter().map(|s| s.to_string()).collect(),
            column_types: column_types.to_vec(),
            seg_writer,
            segment_file,
            data_prefix,
            total_rows: 0,
            rows_per_block,
            column_buffers,
            buffered_rows: 0,
        })
    }

    /// Write a batch of column data. Each element of `columns` contains
    /// the values for one column. All columns must have the same length.
    ///
    /// Data is buffered internally and flushed in blocks of `rows_per_block`.
    /// Small batches are accumulated; large batches are split into blocks.
    pub fn write_columns(&mut self, columns: &[Vec<FlexType>]) -> Result<()> {
        if columns.is_empty() {
            return Ok(());
        }

        let num_rows = columns[0].len();
        if num_rows == 0 {
            return Ok(());
        }

        if columns.len() != self.column_buffers.len() {
            return Err(SFrameError::Format(format!(
                "Expected {} columns, got {}",
                self.column_buffers.len(),
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
    /// Leaves any partial block (< rows_per_block) in the buffer.
    fn flush_full_blocks(&mut self) -> Result<()> {
        while self.buffered_rows >= self.rows_per_block {
            for col_idx in 0..self.column_buffers.len() {
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
        }
        Ok(())
    }

    /// Flush any remaining buffered rows as a final partial block.
    fn flush_remaining(&mut self) -> Result<()> {
        if self.buffered_rows > 0 {
            for col_idx in 0..self.column_buffers.len() {
                let block: Vec<FlexType> = self.column_buffers[col_idx].drain(..).collect();
                self.seg_writer.write_column_block(
                    col_idx,
                    &block,
                    self.column_types[col_idx],
                )?;
            }
            self.buffered_rows = 0;
        }
        Ok(())
    }

    /// Finalize the SFrame: flush remaining data, write segment footer
    /// and all metadata files.
    pub fn finish(mut self) -> Result<()> {
        // Flush any remaining buffered rows as a partial block
        self.flush_remaining()?;

        let segment_sizes = self.seg_writer.finish()?;

        let col_names: Vec<&str> = self.column_names.iter().map(|s| s.as_str()).collect();

        // Write .sidx
        let sidx_file = format!("{}.sidx", self.data_prefix);
        let sidx_path = format!("{}/{}", self.base_path, sidx_file);
        write_sidx(
            &sidx_path,
            &self.segment_file,
            &self.column_types,
            &segment_sizes,
        )?;

        // Write .frame_idx
        let frame_idx_file = format!("{}.frame_idx", self.data_prefix);
        let frame_idx_path = format!("{}/{}", self.base_path, frame_idx_file);
        write_frame_idx(&frame_idx_path, &col_names, &sidx_file, self.total_rows)?;

        // Write dir_archive.ini
        write_dir_archive_ini(&self.base_path, &self.data_prefix)?;

        // Write empty objects.bin (legacy)
        std::fs::write(format!("{}/objects.bin", self.base_path), b"")
            .map_err(SFrameError::Io)?;

        Ok(())
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
