//! CSV parser with type inference supporting structured types.
//!
//! Reads a CSV file and produces SFrameRows with auto-detected column types.
//! Type inference uses the FlexType parser to detect vectors, lists, and dicts
//! in addition to integers, floats, and strings.
//!
//! Uses the custom CSV tokenizer which handles bracket/brace nesting inside
//! fields, unlike the standard `csv` crate.

use std::sync::Arc;

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};
use sframe_types::flex_type_parser::parse_flextype;

use crate::batch::{ColumnData, SFrameRows};

use super::csv_tokenizer::{self, CsvConfig};

/// Options for CSV parsing.
#[derive(Debug, Clone)]
pub struct CsvOptions {
    /// Whether the first row is a header.
    pub has_header: bool,
    /// Delimiter character.
    pub delimiter: u8,
    /// Column type hints (overrides inference).
    pub type_hints: Vec<(String, FlexTypeEnum)>,
    /// Strings that should be treated as NA/Undefined.
    pub na_values: Vec<String>,
    /// Comment character. Lines starting with this are skipped.
    pub comment_char: Option<char>,
    /// Number of rows to skip at the beginning.
    pub skip_rows: usize,
    /// Maximum rows to read.
    pub row_limit: Option<usize>,
    /// Only output these columns (by name).
    pub output_columns: Option<Vec<String>>,
    /// Whether to use the custom tokenizer (bracket-aware).
    /// When false, falls back to the csv crate for performance.
    pub use_custom_tokenizer: bool,
}

impl Default for CsvOptions {
    fn default() -> Self {
        CsvOptions {
            has_header: true,
            delimiter: b',',
            type_hints: Vec::new(),
            na_values: Vec::new(),
            comment_char: Some('#'),
            skip_rows: 0,
            row_limit: None,
            output_columns: None,
            use_custom_tokenizer: true,
        }
    }
}

/// Parse a CSV file into SFrameRows with inferred types.
///
/// Returns (column_names, SFrameRows).
pub fn read_csv(path: &str, options: &CsvOptions) -> Result<(Vec<String>, SFrameRows)> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| SFrameError::Io(e))?;

    read_csv_string(&content, options)
}

/// Parse a CSV string into SFrameRows with inferred types.
///
/// Returns (column_names, SFrameRows).
pub fn read_csv_string(content: &str, options: &CsvOptions) -> Result<(Vec<String>, SFrameRows)> {
    let config = CsvConfig {
        delimiter: String::from(options.delimiter as char),
        has_header: options.has_header,
        comment_char: options.comment_char,
        na_values: options.na_values.clone(),
        skip_rows: options.skip_rows,
        row_limit: options.row_limit,
        output_columns: options.output_columns.clone(),
        ..Default::default()
    };

    let (header, rows) = csv_tokenizer::tokenize(content, &config);

    // Determine column names
    let column_names: Vec<String> = if let Some(header) = header {
        header
    } else if let Some(first_row) = rows.first() {
        (0..first_row.len()).map(|i| format!("X{}", i + 1)).collect()
    } else {
        return Ok((Vec::new(), SFrameRows::empty(&[])));
    };

    let num_cols = column_names.len();

    // Build type hint map
    let hint_map: std::collections::HashMap<&str, FlexTypeEnum> = options
        .type_hints
        .iter()
        .map(|(name, dtype)| (name.as_str(), *dtype))
        .collect();

    // Build NA value set
    let na_set: std::collections::HashSet<&str> = options
        .na_values
        .iter()
        .map(|s| s.as_str())
        .collect();

    // Transpose rows → columns
    let mut string_data: Vec<Vec<String>> = vec![Vec::new(); num_cols];
    for row in &rows {
        for col in 0..num_cols {
            if col < row.len() {
                string_data[col].push(row[col].clone());
            } else {
                // Missing trailing fields → empty string
                string_data[col].push(String::new());
            }
        }
    }

    // Infer types per column and parse
    let mut columns: Vec<ColumnData> = Vec::with_capacity(num_cols);

    for (col_idx, col_name) in column_names.iter().enumerate() {
        let dtype = if let Some(&hint) = hint_map.get(col_name.as_str()) {
            hint
        } else {
            infer_type(&string_data[col_idx], &na_set)
        };

        let col = parse_column(&string_data[col_idx], dtype, &na_set)?;
        columns.push(col);
    }

    // Apply column subsetting if requested
    if let Some(ref output_cols) = options.output_columns {
        let mut final_names = Vec::new();
        let mut final_columns = Vec::new();

        for name in output_cols {
            if let Some(idx) = column_names.iter().position(|n| n == name) {
                final_names.push(column_names[idx].clone());
                let dtype = if let Some(&hint) = hint_map.get(name.as_str()) {
                    hint
                } else {
                    infer_type(&string_data[idx], &na_set)
                };
                let col = parse_column(&string_data[idx], dtype, &na_set)?;
                final_columns.push(col);
            }
        }

        let batch = SFrameRows::new(final_columns)?;
        return Ok((final_names, batch));
    }

    let batch = SFrameRows::new(columns)?;
    Ok((column_names, batch))
}

/// Infer the best type for a column of string values.
///
/// Priority: Integer > Float > Vector > List > Dict > String
fn infer_type(values: &[String], na_set: &std::collections::HashSet<&str>) -> FlexTypeEnum {
    if values.is_empty() {
        return FlexTypeEnum::String;
    }

    let mut could_be_int = true;
    let mut could_be_float = true;
    let mut could_be_vector = true;
    let mut could_be_list = true;
    let mut could_be_dict = true;

    for val in values {
        if val.is_empty() || na_set.contains(val.as_str()) {
            continue; // NA/missing — compatible with any type
        }

        // Try parsing with parse_flextype to see what type we get
        let parsed = parse_flextype(val);
        match parsed.type_enum() {
            FlexTypeEnum::Integer => {
                // Integer is compatible with int and float
                could_be_vector = false;
                could_be_list = false;
                could_be_dict = false;
            }
            FlexTypeEnum::Float => {
                could_be_int = false;
                could_be_vector = false;
                could_be_list = false;
                could_be_dict = false;
            }
            FlexTypeEnum::Vector => {
                could_be_int = false;
                could_be_float = false;
                could_be_list = false;
                could_be_dict = false;
            }
            FlexTypeEnum::List => {
                could_be_int = false;
                could_be_float = false;
                could_be_vector = false;
                could_be_dict = false;
            }
            FlexTypeEnum::Dict => {
                could_be_int = false;
                could_be_float = false;
                could_be_vector = false;
                could_be_list = false;
            }
            FlexTypeEnum::String | FlexTypeEnum::Undefined => {
                could_be_int = false;
                could_be_float = false;
                could_be_vector = false;
                could_be_list = false;
                could_be_dict = false;
            }
            _ => {
                could_be_int = false;
                could_be_float = false;
                could_be_vector = false;
                could_be_list = false;
                could_be_dict = false;
            }
        }

        // Early exit if only string is possible
        if !could_be_int && !could_be_float && !could_be_vector && !could_be_list && !could_be_dict
        {
            return FlexTypeEnum::String;
        }
    }

    if could_be_int {
        FlexTypeEnum::Integer
    } else if could_be_float {
        FlexTypeEnum::Float
    } else if could_be_vector {
        FlexTypeEnum::Vector
    } else if could_be_list {
        FlexTypeEnum::List
    } else if could_be_dict {
        FlexTypeEnum::Dict
    } else {
        FlexTypeEnum::String
    }
}

/// Parse a column of strings into typed ColumnData.
fn parse_column(
    values: &[String],
    dtype: FlexTypeEnum,
    na_set: &std::collections::HashSet<&str>,
) -> Result<ColumnData> {
    let mut col = ColumnData::empty(dtype);

    for val in values {
        if val.is_empty() || na_set.contains(val.as_str()) {
            col.push(&FlexType::Undefined)?;
            continue;
        }

        match dtype {
            FlexTypeEnum::Integer => {
                let v = val.parse::<i64>().map_err(|_| {
                    SFrameError::Format(format!("Cannot parse '{}' as integer", val))
                })?;
                col.push(&FlexType::Integer(v))?;
            }
            FlexTypeEnum::Float => {
                let v = val.parse::<f64>().map_err(|_| {
                    SFrameError::Format(format!("Cannot parse '{}' as float", val))
                })?;
                col.push(&FlexType::Float(v))?;
            }
            FlexTypeEnum::String => {
                col.push(&FlexType::String(Arc::from(val.as_str())))?;
            }
            FlexTypeEnum::Vector | FlexTypeEnum::List | FlexTypeEnum::Dict => {
                let parsed = parse_flextype(val);
                // If the parsed type doesn't match expected, try to coerce
                if parsed.type_enum() == dtype {
                    col.push(&parsed)?;
                } else if parsed == FlexType::Undefined {
                    col.push(&FlexType::Undefined)?;
                } else {
                    // Type mismatch — store as undefined
                    col.push(&FlexType::Undefined)?;
                }
            }
            _ => {
                col.push(&FlexType::String(Arc::from(val.as_str())))?;
            }
        }
    }

    Ok(col)
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
            type_hints: vec![("stars".to_string(), FlexTypeEnum::String)],
            ..Default::default()
        };
        let (col_names, batch) = read_csv(&path, &opts).unwrap();

        let stars_idx = col_names.iter().position(|n| n == "stars").unwrap();
        assert_eq!(batch.dtypes()[stars_idx], FlexTypeEnum::String);
    }

    #[test]
    fn test_infer_type() {
        let na = std::collections::HashSet::new();
        assert_eq!(
            infer_type(&["1".into(), "2".into(), "3".into()], &na),
            FlexTypeEnum::Integer
        );
        assert_eq!(
            infer_type(&["1.5".into(), "2.0".into(), "3.5".into()], &na),
            FlexTypeEnum::Float
        );
        assert_eq!(
            infer_type(&["hello".into(), "world".into()], &na),
            FlexTypeEnum::String
        );
        // Mixed int and float → float
        assert_eq!(
            infer_type(&["1".into(), "2.5".into()], &na),
            FlexTypeEnum::Float
        );
    }

    #[test]
    fn test_infer_type_vector() {
        let na = std::collections::HashSet::new();
        assert_eq!(
            infer_type(
                &["[1, 2, 3]".into(), "[4, 5, 6]".into()],
                &na
            ),
            FlexTypeEnum::Vector
        );
    }

    #[test]
    fn test_infer_type_dict() {
        let na = std::collections::HashSet::new();
        assert_eq!(
            infer_type(
                &["{a: 1, b: 2}".into(), "{c: 3}".into()],
                &na
            ),
            FlexTypeEnum::Dict
        );
    }

    #[test]
    fn test_parse_csv_with_vectors() {
        let csv = "name,data\nfoo,[1,2,3]\nbar,[4,5,6]\n";
        let opts = CsvOptions::default();
        let (col_names, batch) = read_csv_string(csv, &opts).unwrap();

        assert_eq!(col_names, vec!["name", "data"]);
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.dtypes()[1], FlexTypeEnum::Vector);

        match batch.column(1).get(0) {
            FlexType::Vector(v) => assert_eq!(v.as_ref(), &[1.0, 2.0, 3.0]),
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_csv_with_dicts() {
        let csv = "name,data\nfoo,{a: 1}\nbar,{b: 2}\n";
        let opts = CsvOptions::default();
        let (_, batch) = read_csv_string(csv, &opts).unwrap();

        assert_eq!(batch.dtypes()[1], FlexTypeEnum::Dict);
    }

    #[test]
    fn test_parse_csv_with_na_values() {
        let csv = "a,b\n1,hello\nNA,world\n3,NA\n";
        let opts = CsvOptions {
            na_values: vec!["NA".to_string()],
            ..Default::default()
        };
        let (_, batch) = read_csv_string(csv, &opts).unwrap();

        // Column "a" has "1", "NA", "3" — NA becomes undefined
        // With NA excluded, remaining values are 1, 3 → Integer
        assert_eq!(batch.dtypes()[0], FlexTypeEnum::Integer);
        assert_eq!(batch.column(0).get(1), FlexType::Undefined);
        assert_eq!(batch.column(0).get(0), FlexType::Integer(1));
    }

    #[test]
    fn test_parse_csv_with_comments() {
        let csv = "a,b\n# this is a comment\n1,2\n# another comment\n3,4\n";
        let opts = CsvOptions::default();
        let (_, batch) = read_csv_string(csv, &opts).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.column(0).get(0), FlexType::Integer(1));
        assert_eq!(batch.column(0).get(1), FlexType::Integer(3));
    }

    #[test]
    fn test_parse_csv_with_nested_list() {
        let csv = "name,data\nfoo,\"[1, \"\"hello\"\", 2.5]\"\n";
        let opts = CsvOptions::default();
        let (_, batch) = read_csv_string(csv, &opts).unwrap();

        assert_eq!(batch.dtypes()[1], FlexTypeEnum::List);
    }
}
