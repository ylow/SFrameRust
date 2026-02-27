//! Parsers for .frame_idx (INI) and .sidx (JSON) index files.

use std::collections::HashMap;

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::FlexTypeEnum;

/// Parsed from .frame_idx (INI format).
/// Describes the overall SFrame structure: columns, names, row count.
#[derive(Debug)]
pub struct FrameIndex {
    pub version: u32,
    pub num_columns: usize,
    pub nrows: u64,
    pub column_names: Vec<String>,
    pub column_files: Vec<String>,
    pub metadata: HashMap<String, String>,
}

impl FrameIndex {
    /// Parse a .frame_idx file content (INI format).
    pub fn parse(content: &str) -> Result<Self> {
        let mut version = 0u32;
        let mut num_columns = 0usize;
        let mut nrows = 0u64;
        let mut column_names: Vec<(usize, String)> = Vec::new();
        let mut column_files: Vec<(usize, String)> = Vec::new();
        let mut metadata: HashMap<String, String> = HashMap::new();
        let mut section = String::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = line[1..line.len() - 1].to_string();
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                match section.as_str() {
                    "sframe" => match key {
                        "version" => {
                            version = value.parse().map_err(|_| {
                                SFrameError::Format(format!("Invalid version: {}", value))
                            })?
                        }
                        "num_columns" => {
                            num_columns = value.parse().map_err(|_| {
                                SFrameError::Format(format!("Invalid num_columns: {}", value))
                            })?
                        }
                        "nrows" => {
                            nrows = value.parse().map_err(|_| {
                                SFrameError::Format(format!("Invalid nrows: {}", value))
                            })?
                        }
                        _ => {}
                    },
                    "column_names" => {
                        let idx: usize = key.parse().map_err(|_| {
                            SFrameError::Format(format!("Invalid column index: {}", key))
                        })?;
                        column_names.push((idx, value.to_string()));
                    }
                    "column_files" => {
                        let idx: usize = key.parse().map_err(|_| {
                            SFrameError::Format(format!("Invalid column index: {}", key))
                        })?;
                        column_files.push((idx, value.to_string()));
                    }
                    "metadata" => {
                        metadata.insert(key.to_string(), value.to_string());
                    }
                    _ => {}
                }
            }
        }

        // Sort by index to ensure correct order
        column_names.sort_by_key(|(idx, _)| *idx);
        column_files.sort_by_key(|(idx, _)| *idx);

        let column_names: Vec<String> = column_names.into_iter().map(|(_, v)| v).collect();
        let column_files: Vec<String> = column_files.into_iter().map(|(_, v)| v).collect();

        if column_names.len() != num_columns {
            return Err(SFrameError::Format(format!(
                "Expected {} column names, got {}",
                num_columns,
                column_names.len()
            )));
        }

        Ok(FrameIndex {
            version,
            num_columns,
            nrows,
            column_names,
            column_files,
            metadata,
        })
    }
}

/// Per-column metadata from the .sidx file.
#[derive(Debug)]
pub struct ColumnIndex {
    pub dtype: FlexTypeEnum,
    pub content_type: String,
    pub segment_sizes: Vec<u64>,
}

/// Parsed from .sidx (JSON format).
/// Describes the array group: segments, columns, types.
#[derive(Debug)]
pub struct GroupIndex {
    pub version: u32,
    pub nsegments: usize,
    pub segment_files: Vec<String>,
    pub columns: Vec<ColumnIndex>,
}

impl GroupIndex {
    /// Parse a .sidx file content (JSON format).
    pub fn parse(content: &str) -> Result<Self> {
        let json: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| SFrameError::Format(format!("Invalid JSON in .sidx: {}", e)))?;

        // Parse sarray section
        let sarray = json
            .get("sarray")
            .ok_or_else(|| SFrameError::Format("Missing 'sarray' in .sidx".to_string()))?;
        let version = sarray
            .get("version")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| SFrameError::Format("Missing sarray.version".to_string()))?
            as u32;
        let nsegments = sarray
            .get("num_segments")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| SFrameError::Format("Missing sarray.num_segments".to_string()))?
            as usize;

        // Parse segment_files
        let seg_files_obj = json
            .get("segment_files")
            .and_then(|v| v.as_object())
            .ok_or_else(|| SFrameError::Format("Missing 'segment_files' in .sidx".to_string()))?;

        let mut segment_files: Vec<(usize, String)> = Vec::new();
        for (key, val) in seg_files_obj {
            let idx: usize = key.parse().map_err(|_| {
                SFrameError::Format(format!("Invalid segment file key: {}", key))
            })?;
            let filename = val.as_str().ok_or_else(|| {
                SFrameError::Format(format!("Invalid segment file value for key {}", key))
            })?;
            segment_files.push((idx, filename.to_string()));
        }
        segment_files.sort_by_key(|(idx, _)| *idx);
        let segment_files: Vec<String> = segment_files.into_iter().map(|(_, v)| v).collect();

        // Parse columns
        let columns_arr = json
            .get("columns")
            .and_then(|v| v.as_array())
            .ok_or_else(|| SFrameError::Format("Missing 'columns' in .sidx".to_string()))?;

        let mut columns = Vec::with_capacity(columns_arr.len());
        for col_val in columns_arr {
            let content_type = col_val
                .get("content_type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let metadata = col_val
                .get("metadata")
                .ok_or_else(|| SFrameError::Format("Missing column metadata".to_string()))?;

            let type_str = metadata
                .get("__type__")
                .and_then(|v| v.as_str())
                .ok_or_else(|| SFrameError::Format("Missing __type__ in metadata".to_string()))?;

            let type_id: u8 = type_str.parse().map_err(|_| {
                SFrameError::Format(format!("Invalid __type__ value: {}", type_str))
            })?;
            let dtype = FlexTypeEnum::try_from(type_id)?;

            let seg_sizes_obj = col_val
                .get("segment_sizes")
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    SFrameError::Format("Missing segment_sizes in column".to_string())
                })?;

            let mut seg_sizes: Vec<(usize, u64)> = Vec::new();
            for (key, val) in seg_sizes_obj {
                let idx: usize = key.parse().map_err(|_| {
                    SFrameError::Format(format!("Invalid segment size key: {}", key))
                })?;
                let size_str = val.as_str().ok_or_else(|| {
                    SFrameError::Format(format!("Invalid segment size value for key {}", key))
                })?;
                let size: u64 = size_str.parse().map_err(|_| {
                    SFrameError::Format(format!("Invalid segment size: {}", size_str))
                })?;
                seg_sizes.push((idx, size));
            }
            seg_sizes.sort_by_key(|(idx, _)| *idx);
            let segment_sizes: Vec<u64> = seg_sizes.into_iter().map(|(_, v)| v).collect();

            columns.push(ColumnIndex {
                dtype,
                content_type,
                segment_sizes,
            });
        }

        Ok(GroupIndex {
            version,
            nsegments,
            segment_files,
            columns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../samples", manifest)
    }

    #[test]
    fn test_parse_frame_idx() {
        let path = format!(
            "{}/business.sf/m_9688d6320ff94822.frame_idx",
            samples_dir()
        );
        let content = std::fs::read_to_string(&path).unwrap();
        let idx = FrameIndex::parse(&content).unwrap();
        assert_eq!(idx.nrows, 11536);
        assert_eq!(idx.num_columns, 12);
        assert_eq!(idx.column_names[0], "business_id");
        assert_eq!(idx.column_names[11], "type");
        assert_eq!(idx.column_files[0], "m_9688d6320ff94822.sidx:0");
        assert_eq!(idx.column_files[11], "m_9688d6320ff94822.sidx:11");
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

        // Column types from the sample:
        // 0: business_id = STRING (2)
        assert_eq!(idx.columns[0].dtype, FlexTypeEnum::String);
        // 1: categories = VECTOR (3)
        assert_eq!(idx.columns[1].dtype, FlexTypeEnum::Vector);
        // 2: city = STRING (2)
        assert_eq!(idx.columns[2].dtype, FlexTypeEnum::String);
        // 4: latitude = FLOAT (1)
        assert_eq!(idx.columns[4].dtype, FlexTypeEnum::Float);
        // 7: open = INTEGER (0)
        assert_eq!(idx.columns[7].dtype, FlexTypeEnum::Integer);
        // 9: stars = FLOAT (1)
        assert_eq!(idx.columns[9].dtype, FlexTypeEnum::Float);

        // All columns should have 11536 rows in segment 0
        for col in &idx.columns {
            assert_eq!(col.segment_sizes.len(), 1);
            assert_eq!(col.segment_sizes[0], 11536);
        }
    }
}
