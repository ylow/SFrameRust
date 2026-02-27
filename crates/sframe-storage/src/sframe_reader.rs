//! Top-level SFrame reader.
//!
//! Reads a dir_archive (directory containing dir_archive.ini, frame_idx,
//! sidx, and segment files) and provides access to column data.

use sframe_io::local_fs::LocalFileSystem;
use sframe_io::vfs::VirtualFileSystem;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::FlexTypeEnum;

use crate::dir_archive::DirArchive;
use crate::index::{FrameIndex, GroupIndex};
use crate::segment_reader::SegmentReader;

/// A fully opened SFrame, ready to read column data.
pub struct SFrameReader {
    pub frame_index: FrameIndex,
    pub group_index: GroupIndex,
    pub segment_readers: Vec<SegmentReader>,
    base_path: String,
}

impl SFrameReader {
    /// Open an SFrame from a directory path.
    pub fn open(base_path: &str) -> Result<Self> {
        let fs = LocalFileSystem;
        Self::open_with_fs(&fs, base_path)
    }

    /// Open an SFrame using a specific VFS backend.
    pub fn open_with_fs(fs: &dyn VirtualFileSystem, base_path: &str) -> Result<Self> {
        // Read dir_archive.ini
        let archive_path = format!("{}/dir_archive.ini", base_path);
        let archive_content = fs.read_to_string(&archive_path)?;
        let archive = DirArchive::parse(&archive_content)?;

        if archive.contents != "sframe" {
            return Err(SFrameError::Format(format!(
                "Expected sframe contents, got '{}'",
                archive.contents
            )));
        }

        let data_prefix = archive.data_prefix()?;

        // Read frame_idx
        let frame_idx_path = format!("{}/{}.frame_idx", base_path, data_prefix);
        let frame_idx_content = fs.read_to_string(&frame_idx_path)?;
        let frame_index = FrameIndex::parse(&frame_idx_content)?;

        // Read sidx
        let sidx_path = format!("{}/{}.sidx", base_path, data_prefix);
        let sidx_content = fs.read_to_string(&sidx_path)?;
        let group_index = GroupIndex::parse(&sidx_content)?;

        // Collect column types
        let column_types: Vec<FlexTypeEnum> =
            group_index.columns.iter().map(|c| c.dtype).collect();

        // Open segment files
        let mut segment_readers = Vec::with_capacity(group_index.nsegments);
        for seg_file in &group_index.segment_files {
            let seg_path = format!("{}/{}", base_path, seg_file);
            let file = fs.open_read(&seg_path)?;
            let file_size = file.size()?;

            // Box the file as ReadSeek
            let reader = SegmentReader::open(
                Box::new(file),
                file_size,
                column_types.clone(),
            )?;
            segment_readers.push(reader);
        }

        Ok(SFrameReader {
            frame_index,
            group_index,
            segment_readers,
            base_path: base_path.to_string(),
        })
    }

    /// Read an entire column by name, concatenating across all segments.
    pub fn read_column_by_name(&mut self, name: &str) -> Result<Vec<sframe_types::flex_type::FlexType>> {
        let col_idx = self
            .frame_index
            .column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| {
                SFrameError::Format(format!("Column '{}' not found", name))
            })?;

        self.read_column(col_idx)
    }

    /// Read an entire column by index, concatenating across all segments.
    pub fn read_column(&mut self, column: usize) -> Result<Vec<sframe_types::flex_type::FlexType>> {
        let mut result = Vec::new();
        for seg in &mut self.segment_readers {
            let values = seg.read_column(column)?;
            result.extend(values);
        }
        Ok(result)
    }

    /// Number of segments.
    pub fn num_segments(&self) -> usize {
        self.segment_readers.len()
    }

    /// Read all columns from a single segment.
    ///
    /// Returns one `Vec<FlexType>` per column. This avoids loading all
    /// segments at once, limiting peak memory to one segment's worth of data.
    pub fn read_segment_columns(
        &mut self,
        segment: usize,
    ) -> Result<Vec<Vec<sframe_types::flex_type::FlexType>>> {
        if segment >= self.segment_readers.len() {
            return Err(SFrameError::Format(format!(
                "Segment index {} out of range ({})",
                segment,
                self.segment_readers.len()
            )));
        }
        let seg = &mut self.segment_readers[segment];
        let num_cols = seg.num_columns();
        let mut columns = Vec::with_capacity(num_cols);
        for col in 0..num_cols {
            columns.push(seg.read_column(col)?);
        }
        Ok(columns)
    }

    /// Number of rows.
    pub fn num_rows(&self) -> u64 {
        self.frame_index.nrows
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.frame_index.num_columns
    }

    /// Column names.
    pub fn column_names(&self) -> &[String] {
        &self.frame_index.column_names
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.frame_index.metadata.get(key).map(|s| s.as_str())
    }

    /// Get all metadata as a reference.
    pub fn metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.frame_index.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sframe_types::flex_type::FlexType;

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../samples", manifest)
    }

    #[test]
    fn test_open_sframe() {
        let sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
        assert_eq!(sf.num_rows(), 11536);
        assert_eq!(sf.num_columns(), 12);
        assert_eq!(sf.column_names()[0], "business_id");
        assert_eq!(sf.column_names()[11], "type");
    }

    #[test]
    fn test_read_integer_column() {
        let mut sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
        let open_col = sf.read_column_by_name("open").unwrap();
        assert_eq!(open_col.len(), 11536);

        // All values should be 0 or 1
        for (i, val) in open_col.iter().enumerate() {
            match val {
                FlexType::Integer(v) => {
                    assert!(*v == 0 || *v == 1, "Row {} open value: {}", i, v);
                }
                _ => panic!("Row {} expected Integer, got {:?}", i, val),
            }
        }
    }

    #[test]
    fn test_read_float_column() {
        let mut sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
        let lat_col = sf.read_column_by_name("latitude").unwrap();
        assert_eq!(lat_col.len(), 11536);

        // First value should be reasonable latitude
        match &lat_col[0] {
            FlexType::Float(v) => {
                assert!(*v > -90.0 && *v < 90.0, "Unreasonable latitude: {}", v);
            }
            _ => panic!("Expected Float for latitude"),
        }
    }

    #[test]
    fn test_read_vector_column() {
        let mut sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
        let cat_col = sf.read_column_by_name("categories").unwrap();
        assert_eq!(cat_col.len(), 11536);

        // Each value should be a Vector (some may be empty)
        let mut non_empty = 0;
        for (i, val) in cat_col.iter().enumerate() {
            match val {
                FlexType::Vector(v) => {
                    if !v.is_empty() {
                        non_empty += 1;
                    }
                }
                _ => panic!("Row {} expected Vector, got {:?}", i, val),
            }
        }
        assert!(non_empty > 0, "Expected at least some non-empty vectors");
    }

    #[test]
    fn test_read_string_column() {
        let mut sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
        let city_col = sf.read_column_by_name("city").unwrap();
        assert_eq!(city_col.len(), 11536);

        // Should be non-empty strings
        for (i, val) in city_col.iter().take(10).enumerate() {
            match val {
                FlexType::String(s) => {
                    assert!(!s.is_empty(), "Row {} city is empty", i);
                }
                _ => panic!("Row {} expected String, got {:?}", i, val),
            }
        }
    }
}
