//! CSV parser with type inference supporting structured types.
//!
//! Reads a CSV file and produces SFrameRows with auto-detected column types.
//! Type inference uses the FlexType parser to detect vectors, lists, and dicts
//! in addition to integers, floats, and strings.
//!
//! Uses the custom CSV tokenizer which handles bracket/brace nesting inside
//! fields, unlike the standard `csv` crate.
//!
//! ## Chunked / parallel-ready design
//!
//! The parsing pipeline is split into independent stages following the C++
//! SFrame parallel CSV strategy:
//!
//! 1. **Tokenize** — split raw bytes into rows of string fields (sequential
//!    because quote/bracket state is inherently serial).
//! 2. **Infer types** — scan rows row-by-row to determine per-column types
//!    *without* transposing into column-major layout, avoiding the 2× memory
//!    copy that a full transpose would require.
//! 3. **Parse rows** — convert string fields → `FlexType` values in
//!    independent row-range chunks. Each chunk can run on a separate thread
//!    (rayon parallel iteration can be added later).
//!
//! The entry points for chunked consumers are [`tokenize_and_infer`] and
//! [`parse_rows_range`]. Legacy callers use [`read_csv`] / [`read_csv_string`]
//! which delegate to these internally.

use std::collections::HashSet;
use std::sync::Arc;

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};
use sframe_types::flex_type_parser::parse_flextype;

use crate::batch::SFrameRows;

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

// ---------------------------------------------------------------------------
// Chunked API
// ---------------------------------------------------------------------------

/// Schema + raw string data produced by [`tokenize_and_infer`].
///
/// Holds the tokenized row-major string data together with the inferred
/// column names, types, and NA set so that [`parse_rows_range`] can convert
/// arbitrary row ranges into column-oriented `Vec<Vec<FlexType>>` without
/// needing the original options.
pub struct CsvSchema {
    pub column_names: Vec<String>,
    pub column_types: Vec<FlexTypeEnum>,
    /// Row-major tokenized string data. `raw_rows[row][col]`.
    pub raw_rows: Vec<Vec<String>>,
    /// NA strings (owned for lifetime independence from CsvOptions).
    na_values: HashSet<String>,
    /// Indices of columns to output (when `output_columns` was specified).
    /// `None` means output all columns.
    output_indices: Option<Vec<usize>>,
}

impl CsvSchema {
    /// Return the (names, types) that will be emitted by `parse_rows_range`,
    /// respecting `output_columns` subsetting.
    pub fn output_names_types(&self) -> (Vec<String>, Vec<FlexTypeEnum>) {
        if let Some(ref indices) = self.output_indices {
            let names = indices
                .iter()
                .map(|&i| self.column_names[i].clone())
                .collect();
            let types = indices.iter().map(|&i| self.column_types[i]).collect();
            (names, types)
        } else {
            (self.column_names.clone(), self.column_types.clone())
        }
    }
}

/// Tokenize a CSV string and infer column types without building SFrameRows.
///
/// This is step 1+2 of the pipeline: tokenize → determine column names →
/// infer types row-by-row. The raw string rows are retained so that
/// [`parse_rows_range`] can convert them in independent chunks.
pub fn tokenize_and_infer(content: &str, options: &CsvOptions) -> Result<CsvSchema> {
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

    let (header, raw_rows) = csv_tokenizer::tokenize(content, &config);

    // Determine column names
    let column_names: Vec<String> = if let Some(header) = header {
        header
    } else if let Some(first_row) = raw_rows.first() {
        (0..first_row.len())
            .map(|i| format!("X{}", i + 1))
            .collect()
    } else {
        return Ok(CsvSchema {
            column_names: Vec::new(),
            column_types: Vec::new(),
            raw_rows: Vec::new(),
            na_values: HashSet::new(),
            output_indices: None,
        });
    };

    let num_cols = column_names.len();

    // Build type hint map
    let hint_map: std::collections::HashMap<&str, FlexTypeEnum> = options
        .type_hints
        .iter()
        .map(|(name, dtype)| (name.as_str(), *dtype))
        .collect();

    let na_values: HashSet<String> = options.na_values.iter().cloned().collect();

    // Row-by-row type inference (no column transpose).
    // For each column, track which types are still possible.
    let mut could_be_int = vec![true; num_cols];
    let mut could_be_float = vec![true; num_cols];
    let mut could_be_vector = vec![true; num_cols];
    let mut could_be_list = vec![true; num_cols];
    let mut could_be_dict = vec![true; num_cols];

    for row in &raw_rows {
        for col in 0..num_cols {
            // Skip columns that are already determined to be String
            if !could_be_int[col]
                && !could_be_float[col]
                && !could_be_vector[col]
                && !could_be_list[col]
                && !could_be_dict[col]
            {
                continue;
            }

            let val = if col < row.len() { &row[col] } else { "" };
            if val.is_empty() || na_values.contains(val) {
                continue; // NA — compatible with any type
            }

            let parsed = parse_flextype(val);
            match parsed.type_enum() {
                FlexTypeEnum::Integer => {
                    could_be_vector[col] = false;
                    could_be_list[col] = false;
                    could_be_dict[col] = false;
                }
                FlexTypeEnum::Float => {
                    could_be_int[col] = false;
                    could_be_vector[col] = false;
                    could_be_list[col] = false;
                    could_be_dict[col] = false;
                }
                FlexTypeEnum::Vector => {
                    could_be_int[col] = false;
                    could_be_float[col] = false;
                    could_be_list[col] = false;
                    could_be_dict[col] = false;
                }
                FlexTypeEnum::List => {
                    could_be_int[col] = false;
                    could_be_float[col] = false;
                    could_be_vector[col] = false;
                    could_be_dict[col] = false;
                }
                FlexTypeEnum::Dict => {
                    could_be_int[col] = false;
                    could_be_float[col] = false;
                    could_be_vector[col] = false;
                    could_be_list[col] = false;
                }
                _ => {
                    could_be_int[col] = false;
                    could_be_float[col] = false;
                    could_be_vector[col] = false;
                    could_be_list[col] = false;
                    could_be_dict[col] = false;
                }
            }
        }
    }

    // Resolve final types (hints override inference)
    let column_types: Vec<FlexTypeEnum> = (0..num_cols)
        .map(|col| {
            if let Some(&hint) = hint_map.get(column_names[col].as_str()) {
                return hint;
            }
            if could_be_int[col] {
                FlexTypeEnum::Integer
            } else if could_be_float[col] {
                FlexTypeEnum::Float
            } else if could_be_vector[col] {
                FlexTypeEnum::Vector
            } else if could_be_list[col] {
                FlexTypeEnum::List
            } else if could_be_dict[col] {
                FlexTypeEnum::Dict
            } else {
                FlexTypeEnum::String
            }
        })
        .collect();

    // Resolve output_columns → indices
    let output_indices = options.output_columns.as_ref().map(|out_cols| {
        out_cols
            .iter()
            .filter_map(|name| column_names.iter().position(|n| n == name))
            .collect::<Vec<usize>>()
    });

    Ok(CsvSchema {
        column_names,
        column_types,
        raw_rows,
        na_values,
        output_indices,
    })
}

/// Parse rows `[start..end)` from a [`CsvSchema`] into column-oriented
/// `Vec<Vec<FlexType>>`.
///
/// Each call processes an independent chunk — multiple calls can run in
/// parallel (the C++ "each chunk parsed independently" strategy).
///
/// The returned columns follow the schema's `output_indices` order if
/// `output_columns` was specified, otherwise all columns are returned.
pub fn parse_rows_range(
    schema: &CsvSchema,
    start: usize,
    end: usize,
) -> Result<Vec<Vec<FlexType>>> {
    let end = end.min(schema.raw_rows.len());
    let chunk_len = end.saturating_sub(start);

    // Determine which columns to emit
    let col_indices: &[usize];
    let all_indices: Vec<usize>;
    if let Some(ref out) = schema.output_indices {
        col_indices = out;
    } else {
        all_indices = (0..schema.column_types.len()).collect();
        col_indices = &all_indices;
    }

    let mut col_vecs: Vec<Vec<FlexType>> = col_indices
        .iter()
        .map(|_| Vec::with_capacity(chunk_len))
        .collect();

    for row_idx in start..end {
        let row = &schema.raw_rows[row_idx];
        for (out_col, &src_col) in col_indices.iter().enumerate() {
            let val_str = if src_col < row.len() {
                &row[src_col]
            } else {
                ""
            };
            let val = parse_cell(val_str, schema.column_types[src_col], &schema.na_values)?;
            col_vecs[out_col].push(val);
        }
    }

    Ok(col_vecs)
}

/// Parse a single cell string into a `FlexType` given the target column type.
fn parse_cell(val: &str, dtype: FlexTypeEnum, na_set: &HashSet<String>) -> Result<FlexType> {
    if val.is_empty() || na_set.contains(val) {
        return Ok(FlexType::Undefined);
    }

    match dtype {
        FlexTypeEnum::Integer => {
            let v = val.parse::<i64>().map_err(|_| {
                SFrameError::Format(format!("Cannot parse '{}' as integer", val))
            })?;
            Ok(FlexType::Integer(v))
        }
        FlexTypeEnum::Float => {
            let v = val.parse::<f64>().map_err(|_| {
                SFrameError::Format(format!("Cannot parse '{}' as float", val))
            })?;
            Ok(FlexType::Float(v))
        }
        FlexTypeEnum::String => Ok(FlexType::String(Arc::from(val))),
        FlexTypeEnum::Vector | FlexTypeEnum::List | FlexTypeEnum::Dict => {
            let parsed = parse_flextype(val);
            if parsed.type_enum() == dtype {
                Ok(parsed)
            } else {
                Ok(FlexType::Undefined)
            }
        }
        _ => Ok(FlexType::String(Arc::from(val))),
    }
}

// ---------------------------------------------------------------------------
// Legacy API (delegates to the chunked pipeline)
// ---------------------------------------------------------------------------

/// Parse a CSV file into SFrameRows with inferred types.
///
/// Returns (column_names, SFrameRows).
pub fn read_csv(path: &str, options: &CsvOptions) -> Result<(Vec<String>, SFrameRows)> {
    let content = std::fs::read_to_string(path).map_err(SFrameError::Io)?;
    read_csv_string(&content, options)
}

/// Parse a CSV string into SFrameRows with inferred types.
///
/// Returns (column_names, SFrameRows).
pub fn read_csv_string(content: &str, options: &CsvOptions) -> Result<(Vec<String>, SFrameRows)> {
    let schema = tokenize_and_infer(content, options)?;

    if schema.raw_rows.is_empty() {
        let dtypes = schema.column_types;
        return Ok((schema.column_names, SFrameRows::empty(&dtypes)));
    }

    let col_vecs = parse_rows_range(&schema, 0, schema.raw_rows.len())?;

    // Determine output names + types based on output_indices
    let (out_names, out_types) = if let Some(ref indices) = schema.output_indices {
        let names: Vec<String> = indices
            .iter()
            .map(|&i| schema.column_names[i].clone())
            .collect();
        let types: Vec<FlexTypeEnum> = indices
            .iter()
            .map(|&i| schema.column_types[i])
            .collect();
        (names, types)
    } else {
        (schema.column_names, schema.column_types)
    };

    let batch = SFrameRows::from_column_vecs(col_vecs, &out_types)?;
    Ok((out_names, batch))
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Infer the best type for a column of string values.
///
/// Priority: Integer > Float > Vector > List > Dict > String
#[cfg(test)]
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
            continue;
        }

        let parsed = parse_flextype(val);
        match parsed.type_enum() {
            FlexTypeEnum::Integer => {
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

    #[test]
    fn test_tokenize_and_infer_basic() {
        let csv = "a,b,c\n1,2.5,hello\n3,4.0,world\n";
        let opts = CsvOptions::default();
        let schema = tokenize_and_infer(csv, &opts).unwrap();

        assert_eq!(schema.column_names, vec!["a", "b", "c"]);
        assert_eq!(
            schema.column_types,
            vec![FlexTypeEnum::Integer, FlexTypeEnum::Float, FlexTypeEnum::String]
        );
        assert_eq!(schema.raw_rows.len(), 2);
    }

    #[test]
    fn test_parse_rows_range_chunks() {
        let csv = "x\n1\n2\n3\n4\n5\n";
        let opts = CsvOptions::default();
        let schema = tokenize_and_infer(csv, &opts).unwrap();

        // Parse first two rows
        let chunk1 = parse_rows_range(&schema, 0, 2).unwrap();
        assert_eq!(chunk1.len(), 1); // 1 column
        assert_eq!(chunk1[0].len(), 2);
        assert_eq!(chunk1[0][0], FlexType::Integer(1));
        assert_eq!(chunk1[0][1], FlexType::Integer(2));

        // Parse rows 2..5
        let chunk2 = parse_rows_range(&schema, 2, 5).unwrap();
        assert_eq!(chunk2[0].len(), 3);
        assert_eq!(chunk2[0][0], FlexType::Integer(3));
        assert_eq!(chunk2[0][2], FlexType::Integer(5));
    }

    #[test]
    fn test_chunked_matches_full_parse() {
        let path = format!("{}/business.csv", samples_dir());
        let opts = CsvOptions::default();

        // Full parse via legacy API
        let (names_full, batch_full) = read_csv(&path, &opts).unwrap();

        // Chunked parse
        let content = std::fs::read_to_string(&path).unwrap();
        let schema = tokenize_and_infer(&content, &opts).unwrap();

        assert_eq!(schema.column_names, names_full);
        assert_eq!(schema.column_types, batch_full.dtypes());

        // Parse in two chunks and compare
        let mid = schema.raw_rows.len() / 2;
        let chunk1 = parse_rows_range(&schema, 0, mid).unwrap();
        let chunk2 = parse_rows_range(&schema, mid, schema.raw_rows.len()).unwrap();

        // Verify first column of chunk1 matches batch_full
        for (i, val) in chunk1[0].iter().enumerate() {
            assert_eq!(*val, batch_full.column(0).get(i));
        }
        // Verify first column of chunk2
        for (i, val) in chunk2[0].iter().enumerate() {
            assert_eq!(*val, batch_full.column(0).get(mid + i));
        }
    }
}
