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

use crate::segment_writer::BufferedSegmentWriter;

/// Default number of rows per segment before auto-splitting.
/// Set to u64::MAX so segments are 1:1 with writer threads; block-level
/// flushing already bounds memory.
const DEFAULT_ROWS_PER_SEGMENT: u64 = u64::MAX;

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
    let data_prefix = format!("m_{hash}");

    // Create directory
    std::fs::create_dir_all(base_path)
        .map_err(SFrameError::Io)?;

    // Write segment file
    let segment_file = format!("{data_prefix}.0000");
    let segment_path = format!("{base_path}/{segment_file}");
    let segment_sizes = write_segment(
        &segment_path,
        num_columns,
        column_types,
        rows,
    )?;

    // Write .sidx (JSON)
    let sidx_file = format!("{data_prefix}.sidx");
    let sidx_path = format!("{base_path}/{sidx_file}");
    write_sidx_to_fs(
        &sidx_path,
        &[segment_file],
        column_types,
        &[segment_sizes],
    )?;

    // Write .frame_idx (INI)
    let frame_idx_file = format!("{data_prefix}.frame_idx");
    let frame_idx_path = format!("{base_path}/{frame_idx_file}");
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
    std::fs::write(format!("{base_path}/objects.bin"), b"")
        .map_err(SFrameError::Io)?;

    Ok(())
}

/// Write a single segment file containing all rows.
///
/// Each column is written independently with adaptive block sizing
/// handled by `BufferedSegmentWriter`.
fn write_segment(
    path: &str,
    _num_columns: usize,
    column_types: &[FlexTypeEnum],
    rows: &[Vec<FlexType>],
) -> Result<Vec<u64>> {
    let file = std::fs::File::create(path).map_err(SFrameError::Io)?;
    let buf_writer = BufWriter::new(file);
    let mut seg_writer = BufferedSegmentWriter::new(buf_writer, column_types);

    if rows.is_empty() {
        return seg_writer.finish();
    }

    let nrows = rows.len();

    // Write each column independently
    for col in 0..column_types.len() {
        let values: Vec<FlexType> = rows[0..nrows]
            .iter()
            .map(|row| row[col].clone())
            .collect();
        seg_writer.write_column_block(col, &values, column_types[col])?;
    }

    seg_writer.finish()
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
        seg_files_map.insert(format!("{i:04}"), Value::String(seg_file.clone()));
    }

    let mut columns = Vec::new();
    for (col_idx, &dtype) in column_types.iter().enumerate() {
        let mut seg_sizes_map = Map::new();
        for (seg_idx, sizes) in all_segment_sizes.iter().enumerate() {
            seg_sizes_map.insert(
                format!("{seg_idx:04}"),
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
        .map_err(|e| SFrameError::Format(format!("JSON serialization error: {e}")))
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
    content.push_str(&format!("num_segments={num_segments}\n"));
    content.push_str(&format!("num_columns={}\n", column_names.len()));
    content.push_str(&format!("nrows={nrows}\n"));

    content.push_str("\n[column_names]\n");
    for (i, name) in column_names.iter().enumerate() {
        content.push_str(&format!("{i}={name}\n"));
    }

    content.push_str("\n[column_files]\n");
    for i in 0..column_names.len() {
        content.push_str(&format!("{i}={sidx_file}:{i}\n"));
    }

    content.push_str("\n[metadata]\n");
    for (key, value) in metadata {
        content.push_str(&format!("{key}={value}\n"));
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
         0002={data_prefix}\n"
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
    vfs.write_string(&format!("{base_path}/dir_archive.ini"), &content)
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
    std::fs::write(format!("{base_path}/dir_archive.ini"), content)
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
    rows_per_segment: u64,
    num_columns: usize,

    // Current segment state
    seg_writer: BufferedSegmentWriter<Box<dyn WritableFile>>,
    current_segment_idx: usize,
    /// Total rows received for the current segment.
    rows_in_current_segment: u64,

    // Accumulated metadata across all finished segments
    segment_files: Vec<String>,
    all_segment_sizes: Vec<Vec<u64>>,

    total_rows: u64,

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
        let data_prefix = format!("m_{hash}");

        vfs.mkdir_p(base_path)?;

        let seg_writer = create_buffered_segment_writer_vfs(
            &*vfs,
            base_path,
            &data_prefix,
            0,
            column_types,
        )?;

        Ok(SFrameWriter {
            vfs,
            base_path: base_path.to_string(),
            column_names: column_names.iter().map(|s| s.to_string()).collect(),
            column_types: column_types.to_vec(),
            data_prefix,
            rows_per_segment,
            num_columns,
            seg_writer,
            current_segment_idx: 0,
            rows_in_current_segment: 0,
            segment_files: Vec::new(),
            all_segment_sizes: Vec::new(),
            total_rows: 0,
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
    /// Data is buffered internally by `BufferedSegmentWriter` and flushed
    /// in blocks of approximately 64KB. When a segment fills up, it is
    /// finalized and a new segment starts.
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
                // Segment is full — finish and start a new one
                self.finish_current_segment()?;
                self.start_new_segment()?;
                continue;
            }

            let chunk_size = (num_rows - offset).min(segment_remaining);

            for (col_idx, col_data) in columns.iter().enumerate() {
                self.seg_writer.write_column_block(
                    col_idx,
                    &col_data[offset..offset + chunk_size],
                    self.column_types[col_idx],
                )?;
            }

            self.rows_in_current_segment += chunk_size as u64;
            self.total_rows += chunk_size as u64;
            offset += chunk_size;
        }

        Ok(())
    }

    /// Finish the current segment: write its footer and record its metadata.
    fn finish_current_segment(&mut self) -> Result<()> {
        // Swap in a NullWriter placeholder while we finish the current segment.
        let seg_writer = std::mem::replace(
            &mut self.seg_writer,
            BufferedSegmentWriter::new(
                Box::new(NullWriter) as Box<dyn WritableFile>,
                &self.column_types,
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

        let seg_writer = create_buffered_segment_writer_vfs(
            &*self.vfs,
            &self.base_path,
            &self.data_prefix,
            self.current_segment_idx,
            &self.column_types,
        )?;
        self.seg_writer = seg_writer;
        Ok(())
    }

    /// Finalize the SFrame: flush remaining data, write segment footer
    /// and all metadata files.
    ///
    /// Returns the total number of rows written.
    pub fn finish(mut self) -> Result<u64> {
        // Finish the current (last) segment (flushes remaining buffered data)
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

/// Assemble SFrame metadata for pre-built segment files.
///
/// Writes dir_archive.ini, .sidx, .frame_idx, and objects.bin into
/// `base_path`, referencing the given segment files. The segment files
/// must already exist at `{base_path}/{segment_file}`.
///
/// This is the complement to building segments with `SegmentWriter`
/// directly — it creates the metadata that `SFrameReader` needs to
/// open the SFrame.
///
/// # Arguments
///
/// * `vfs` — filesystem backend to write metadata files
/// * `base_path` — directory containing the segment files
/// * `column_names` — names of the columns in the SFrame
/// * `column_types` — types of the columns in the SFrame
/// * `segment_files` — file names of the pre-built segment files (relative to `base_path`)
/// * `all_segment_sizes` — per-segment, per-column row counts from `SegmentWriter::finish()`
/// * `total_rows` — total number of rows across all segments
pub fn assemble_sframe_from_segments(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    column_names: &[&str],
    column_types: &[FlexTypeEnum],
    segment_files: &[String],
    all_segment_sizes: &[Vec<u64>],
    total_rows: u64,
    metadata: &std::collections::HashMap<String, String>,
) -> Result<()> {
    let hash = generate_hash(base_path);
    let data_prefix = format!("m_{hash}");

    vfs.mkdir_p(base_path)?;

    // Write .sidx — references segment_files as-is
    let sidx_file = format!("{data_prefix}.sidx");
    let sidx_path = format!("{base_path}/{sidx_file}");
    write_sidx(vfs, &sidx_path, segment_files, column_types, all_segment_sizes)?;

    // Write .frame_idx
    let frame_idx_file = format!("{data_prefix}.frame_idx");
    let frame_idx_path = format!("{base_path}/{frame_idx_file}");
    write_frame_idx(
        vfs,
        &frame_idx_path,
        column_names,
        &sidx_file,
        total_rows,
        segment_files.len(),
        metadata,
    )?;

    // Write dir_archive.ini
    write_dir_archive_ini(vfs, base_path, &data_prefix)?;

    // Write empty objects.bin
    vfs.write_string(&format!("{base_path}/objects.bin"), "")?;

    Ok(())
}

/// Create a segment file name like "m_xxx.0000".
pub fn segment_filename(data_prefix: &str, segment_idx: usize) -> String {
    format!("{data_prefix}.{segment_idx:04}")
}

/// Create a new BufferedSegmentWriter via VFS for the given segment index.
fn create_buffered_segment_writer_vfs(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    data_prefix: &str,
    segment_idx: usize,
    dtypes: &[FlexTypeEnum],
) -> Result<BufferedSegmentWriter<Box<dyn WritableFile>>> {
    let segment_file = segment_filename(data_prefix, segment_idx);
    let segment_path = format!("{base_path}/{segment_file}");
    let file = vfs.open_write(&segment_path)?;
    Ok(BufferedSegmentWriter::new(file, dtypes))
}

/// Generate a deterministic hash prefix from the path.
pub fn generate_hash(path: &str) -> String {
    // Simple hash - not cryptographic, just for unique file naming
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in path.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    format!("{hash:016x}")
}
