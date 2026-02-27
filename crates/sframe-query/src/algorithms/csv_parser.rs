//! CSV parser with type inference.
//!
//! Reads a CSV file and produces SFrameRows with auto-detected column types.
//! Type inference examines initial rows and promotes types as needed.

use std::sync::Arc;

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::FlexTypeEnum;

use crate::batch::{ColumnData, SFrameRows};

/// Options for CSV parsing.
#[derive(Debug, Clone)]
pub struct CsvOptions {
    /// Whether the first row is a header.
    pub has_header: bool,
    /// Delimiter character.
    pub delimiter: u8,
    /// Column type hints (overrides inference).
    pub type_hints: Vec<(String, FlexTypeEnum)>,
}

impl Default for CsvOptions {
    fn default() -> Self {
        CsvOptions {
            has_header: true,
            delimiter: b',',
            type_hints: Vec::new(),
        }
    }
}

/// Parse a CSV file into SFrameRows with inferred types.
///
/// Returns (column_names, SFrameRows).
pub fn read_csv(path: &str, options: &CsvOptions) -> Result<(Vec<String>, SFrameRows)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(options.has_header)
        .delimiter(options.delimiter)
        .from_path(path)
        .map_err(|e| SFrameError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    // Get headers
    let column_names: Vec<String> = if options.has_header {
        reader
            .headers()
            .map_err(|e| SFrameError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        // Auto-generate names
        let first = reader
            .records()
            .next()
            .ok_or_else(|| SFrameError::Format("Empty CSV".to_string()))?
            .map_err(|e| SFrameError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        (0..first.len()).map(|i| format!("X{}", i + 1)).collect()
    };

    let num_cols = column_names.len();

    // Build type hint map
    let hint_map: std::collections::HashMap<&str, FlexTypeEnum> = options
        .type_hints
        .iter()
        .map(|(name, dtype)| (name.as_str(), *dtype))
        .collect();

    // Read all records as strings first
    let mut string_data: Vec<Vec<String>> = vec![Vec::new(); num_cols];

    for record in reader.records() {
        let record = record.map_err(|e| {
            SFrameError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
        })?;
        for (col, field) in record.iter().enumerate() {
            if col < num_cols {
                string_data[col].push(field.to_string());
            }
        }
    }

    // Infer types per column
    let mut columns: Vec<ColumnData> = Vec::with_capacity(num_cols);

    for (col_idx, col_name) in column_names.iter().enumerate() {
        let dtype = if let Some(&hint) = hint_map.get(col_name.as_str()) {
            hint
        } else {
            infer_type(&string_data[col_idx])
        };

        let col = parse_column(&string_data[col_idx], dtype)?;
        columns.push(col);
    }

    let batch = SFrameRows::new(columns)?;
    Ok((column_names, batch))
}

/// Infer the best type for a column of string values.
fn infer_type(values: &[String]) -> FlexTypeEnum {
    if values.is_empty() {
        return FlexTypeEnum::String;
    }

    let mut could_be_int = true;
    let mut could_be_float = true;

    for val in values {
        if val.is_empty() {
            continue; // Undefined/missing
        }
        if could_be_int && val.parse::<i64>().is_err() {
            could_be_int = false;
        }
        if could_be_float && val.parse::<f64>().is_err() {
            could_be_float = false;
        }
        if !could_be_int && !could_be_float {
            break;
        }
    }

    if could_be_int {
        FlexTypeEnum::Integer
    } else if could_be_float {
        FlexTypeEnum::Float
    } else {
        FlexTypeEnum::String
    }
}

/// Parse a column of strings into typed ColumnData.
fn parse_column(values: &[String], dtype: FlexTypeEnum) -> Result<ColumnData> {
    match dtype {
        FlexTypeEnum::Integer => {
            let mut col: Vec<Option<i64>> = Vec::with_capacity(values.len());
            for val in values {
                if val.is_empty() {
                    col.push(None);
                } else {
                    col.push(Some(val.parse::<i64>().map_err(|_| {
                        SFrameError::Format(format!("Cannot parse '{}' as integer", val))
                    })?));
                }
            }
            Ok(ColumnData::Integer(col))
        }
        FlexTypeEnum::Float => {
            let mut col: Vec<Option<f64>> = Vec::with_capacity(values.len());
            for val in values {
                if val.is_empty() {
                    col.push(None);
                } else {
                    col.push(Some(val.parse::<f64>().map_err(|_| {
                        SFrameError::Format(format!("Cannot parse '{}' as float", val))
                    })?));
                }
            }
            Ok(ColumnData::Float(col))
        }
        FlexTypeEnum::String => {
            let col: Vec<Option<Arc<str>>> = values
                .iter()
                .map(|val| {
                    if val.is_empty() {
                        None
                    } else {
                        Some(Arc::from(val.as_str()))
                    }
                })
                .collect();
            Ok(ColumnData::String(col))
        }
        _ => {
            // Fallback: store as strings
            let col: Vec<Option<Arc<str>>> = values
                .iter()
                .map(|val| {
                    if val.is_empty() {
                        None
                    } else {
                        Some(Arc::from(val.as_str()))
                    }
                })
                .collect();
            Ok(ColumnData::String(col))
        }
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
    fn test_read_business_csv() {
        let path = format!("{}/business.csv", samples_dir());
        let (col_names, batch) = read_csv(&path, &CsvOptions::default()).unwrap();

        assert_eq!(col_names.len(), 12);
        assert_eq!(col_names[0], "business_id");
        assert_eq!(batch.num_rows(), 11536);
        assert_eq!(batch.num_columns(), 12);
    }

    #[test]
    fn test_type_inference() {
        let path = format!("{}/business.csv", samples_dir());
        let (col_names, batch) = read_csv(&path, &CsvOptions::default()).unwrap();

        // Check inferred types
        let dtypes = batch.dtypes();

        // business_id should be String (alphanumeric)
        let biz_idx = col_names.iter().position(|n| n == "business_id").unwrap();
        assert_eq!(dtypes[biz_idx], FlexTypeEnum::String);

        // stars should be Float
        let stars_idx = col_names.iter().position(|n| n == "stars").unwrap();
        assert_eq!(dtypes[stars_idx], FlexTypeEnum::Float);

        // review_count should be Integer
        let rc_idx = col_names.iter().position(|n| n == "review_count").unwrap();
        assert_eq!(dtypes[rc_idx], FlexTypeEnum::Integer);

        // open should be Integer
        let open_idx = col_names.iter().position(|n| n == "open").unwrap();
        assert_eq!(dtypes[open_idx], FlexTypeEnum::Integer);
    }

    #[test]
    fn test_csv_with_type_hints() {
        let path = format!("{}/business.csv", samples_dir());
        let opts = CsvOptions {
            type_hints: vec![
                ("stars".to_string(), FlexTypeEnum::String),
            ],
            ..Default::default()
        };
        let (col_names, batch) = read_csv(&path, &opts).unwrap();

        let stars_idx = col_names.iter().position(|n| n == "stars").unwrap();
        assert_eq!(batch.dtypes()[stars_idx], FlexTypeEnum::String);
    }

    #[test]
    fn test_infer_type() {
        assert_eq!(
            infer_type(&["1".into(), "2".into(), "3".into()]),
            FlexTypeEnum::Integer
        );
        assert_eq!(
            infer_type(&["1.5".into(), "2.0".into(), "3.5".into()]),
            FlexTypeEnum::Float
        );
        assert_eq!(
            infer_type(&["hello".into(), "world".into()]),
            FlexTypeEnum::String
        );
        // Mixed int and float → float
        assert_eq!(
            infer_type(&["1".into(), "2.5".into()]),
            FlexTypeEnum::Float
        );
    }
}
