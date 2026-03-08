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
use sframe_types::flex_wrappers::{FlexList, FlexString, FlexVec};

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
    /// Field delimiter (supports multi-char, e.g. "::" or "\t\t").
    pub delimiter: String,
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
    /// Whether doubled quote chars represent an escaped quote. Default: true.
    pub double_quote: bool,
    /// Line terminator. Default: "\n" (handles \r\n and \r automatically).
    /// Set to a custom string (e.g. "zzz") for non-standard terminators.
    pub line_terminator: String,
    /// Escape character. Default: '\\'.
    pub escape_char: char,
    /// Skip leading whitespace before each field. Default: true.
    pub skip_initial_space: bool,
}

impl Default for CsvOptions {
    fn default() -> Self {
        CsvOptions {
            has_header: true,
            delimiter: ",".to_string(),
            type_hints: Vec::new(),
            na_values: Vec::new(),
            comment_char: Some('#'),
            skip_rows: 0,
            row_limit: None,
            output_columns: None,
            use_custom_tokenizer: true,
            double_quote: true,
            line_terminator: "\n".to_string(),
            escape_char: '\\',
            skip_initial_space: true,
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
        delimiter: options.delimiter.clone(),
        line_terminator: options.line_terminator.clone(),
        escape_char: options.escape_char,
        double_quote: options.double_quote,
        skip_initial_space: options.skip_initial_space,
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
pub(crate) fn parse_cell(val: &str, dtype: FlexTypeEnum, na_set: &HashSet<String>) -> Result<FlexType> {
    // Check NA values first
    if na_set.contains(val) {
        return Ok(FlexType::Undefined);
    }
    // Empty values: String columns produce String(""), all others produce Undefined.
    if val.is_empty() {
        return match dtype {
            FlexTypeEnum::String => Ok(FlexType::String(FlexString::from(""))),
            _ => Ok(FlexType::Undefined),
        };
    }

    match dtype {
        FlexTypeEnum::Integer => {
            let v = val.parse::<i64>().map_err(|_| {
                SFrameError::Format(format!("Cannot parse '{val}' as integer"))
            })?;
            Ok(FlexType::Integer(v))
        }
        FlexTypeEnum::Float => {
            let v = val.parse::<f64>().map_err(|_| {
                SFrameError::Format(format!("Cannot parse '{val}' as float"))
            })?;
            Ok(FlexType::Float(v))
        }
        FlexTypeEnum::String => Ok(FlexType::String(FlexString::from(val))),
        FlexTypeEnum::Vector => {
            let parsed = parse_flextype(val);
            match &parsed {
                FlexType::Vector(_) => Ok(parsed),
                // List of all-numeric can be coerced to Vector
                FlexType::List(items) => {
                    let mut floats = Vec::with_capacity(items.len());
                    for item in items.iter() {
                        match item {
                            FlexType::Integer(i) => floats.push(*i as f64),
                            FlexType::Float(f) => floats.push(*f),
                            _ => return Ok(FlexType::Undefined),
                        }
                    }
                    Ok(FlexType::Vector(FlexVec::from(floats)))
                }
                _ => Ok(FlexType::Undefined),
            }
        }
        FlexTypeEnum::List => {
            let parsed = parse_flextype(val);
            match &parsed {
                FlexType::List(_) => Ok(parsed),
                // Vector can be coerced to List. Prefer Integer for whole
                // numbers to match C++ recursive_parse behavior.
                FlexType::Vector(v) => {
                    let items: Vec<FlexType> = v
                        .iter()
                        .map(|&f| {
                            if f.fract() == 0.0
                                && f >= i64::MIN as f64
                                && f <= i64::MAX as f64
                            {
                                FlexType::Integer(f as i64)
                            } else {
                                FlexType::Float(f)
                            }
                        })
                        .collect();
                    Ok(FlexType::List(FlexList::from(items)))
                }
                _ => Ok(FlexType::Undefined),
            }
        }
        FlexTypeEnum::Dict => {
            let parsed = parse_flextype(val);
            if parsed.type_enum() == FlexTypeEnum::Dict {
                Ok(parsed)
            } else {
                Ok(FlexType::Undefined)
            }
        }
        FlexTypeEnum::Undefined => {
            // Per-value flexible parsing: auto-detect type for each cell.
            // This matches C++ UNDEFINED column type behavior where each
            // value is independently parsed via general_flexible_type_parse.
            let parsed = parse_flextype(val);
            // parse_flextype trims whitespace, so a whitespace-only string
            // (e.g. "\n" after escape processing) returns Undefined. In CSV
            // context, non-empty values should be preserved as String.
            if matches!(parsed, FlexType::Undefined) {
                Ok(FlexType::String(FlexString::from(val)))
            } else {
                Ok(parsed)
            }
        }
        _ => Ok(FlexType::String(FlexString::from(val))),
    }
}

// ---------------------------------------------------------------------------
// Legacy API (delegates to the chunked pipeline)
// ---------------------------------------------------------------------------

/// Parse a CSV file into SFrameRows with inferred types.
///
/// Returns (column_names, SFrameRows).
pub fn read_csv(path: &str, options: &CsvOptions) -> Result<(Vec<String>, SFrameRows)> {
    let streaming = CsvStreamingParse::open(path, options)?;
    let names = streaming.column_names.clone();
    let dtypes = streaming.column_types.clone();
    if names.is_empty() {
        return Ok((names, SFrameRows::empty(&dtypes)));
    }
    let mut result: Option<SFrameRows> = None;
    streaming.parse_chunks(|col_vecs| {
        let batch = SFrameRows::from_column_vecs(col_vecs, &dtypes)?;
        match &mut result {
            None => result = Some(batch),
            Some(existing) => existing.append(&batch)?,
        }
        Ok(())
    })?;
    Ok((names, result.unwrap_or_else(|| SFrameRows::empty(&dtypes))))
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
// Streaming / parallel API
// ---------------------------------------------------------------------------

/// Convert CsvOptions to the tokenizer's CsvConfig.
pub fn options_to_config(options: &CsvOptions) -> CsvConfig {
    CsvConfig {
        delimiter: options.delimiter.clone(),
        line_terminator: options.line_terminator.clone(),
        escape_char: options.escape_char,
        double_quote: options.double_quote,
        skip_initial_space: options.skip_initial_space,
        has_header: options.has_header,
        comment_char: options.comment_char,
        na_values: options.na_values.clone(),
        skip_rows: options.skip_rows,
        row_limit: options.row_limit,
        output_columns: options.output_columns.clone(),
        ..Default::default()
    }
}

/// Incremental type inference state.
///
/// Uses the same elimination logic as `tokenize_and_infer`, but can be
/// updated chunk-by-chunk so type inference doesn't require holding all
/// raw rows in memory simultaneously.
pub struct TypeInferenceState {
    could_be_int: Vec<bool>,
    could_be_float: Vec<bool>,
    could_be_vector: Vec<bool>,
    could_be_list: Vec<bool>,
    could_be_dict: Vec<bool>,
    num_cols: usize,
}

impl TypeInferenceState {
    pub fn new(num_cols: usize) -> Self {
        TypeInferenceState {
            could_be_int: vec![true; num_cols],
            could_be_float: vec![true; num_cols],
            could_be_vector: vec![true; num_cols],
            could_be_list: vec![true; num_cols],
            could_be_dict: vec![true; num_cols],
            num_cols,
        }
    }

    /// Observe a single row of raw string values.
    pub fn observe_row(&mut self, row: &[String], na_values: &HashSet<String>) {
        for col in 0..self.num_cols {
            if !self.could_be_int[col]
                && !self.could_be_float[col]
                && !self.could_be_vector[col]
                && !self.could_be_list[col]
                && !self.could_be_dict[col]
            {
                continue;
            }
            let val = if col < row.len() { &row[col] } else { "" };
            if val.is_empty() || na_values.contains(val) {
                continue;
            }

            let parsed = parse_flextype(val);
            match parsed.type_enum() {
                FlexTypeEnum::Integer => {
                    self.could_be_vector[col] = false;
                    self.could_be_list[col] = false;
                    self.could_be_dict[col] = false;
                }
                FlexTypeEnum::Float => {
                    self.could_be_int[col] = false;
                    self.could_be_vector[col] = false;
                    self.could_be_list[col] = false;
                    self.could_be_dict[col] = false;
                }
                FlexTypeEnum::Vector => {
                    self.could_be_int[col] = false;
                    self.could_be_float[col] = false;
                    self.could_be_list[col] = false;
                    self.could_be_dict[col] = false;
                }
                FlexTypeEnum::List => {
                    self.could_be_int[col] = false;
                    self.could_be_float[col] = false;
                    self.could_be_vector[col] = false;
                    self.could_be_dict[col] = false;
                }
                FlexTypeEnum::Dict => {
                    self.could_be_int[col] = false;
                    self.could_be_float[col] = false;
                    self.could_be_vector[col] = false;
                    self.could_be_list[col] = false;
                }
                _ => {
                    self.could_be_int[col] = false;
                    self.could_be_float[col] = false;
                    self.could_be_vector[col] = false;
                    self.could_be_list[col] = false;
                    self.could_be_dict[col] = false;
                }
            }
        }
    }

    /// Finalize: resolve column types, applying type hints.
    pub fn finalize(
        &self,
        hint_map: &std::collections::HashMap<&str, FlexTypeEnum>,
        column_names: &[String],
    ) -> Vec<FlexTypeEnum> {
        (0..self.num_cols)
            .map(|col| {
                if let Some(&hint) = hint_map.get(column_names[col].as_str()) {
                    return hint;
                }
                if self.could_be_int[col] {
                    FlexTypeEnum::Integer
                } else if self.could_be_float[col] {
                    FlexTypeEnum::Float
                } else if self.could_be_vector[col] {
                    FlexTypeEnum::Vector
                } else if self.could_be_list[col] {
                    FlexTypeEnum::List
                } else if self.could_be_dict[col] {
                    FlexTypeEnum::Dict
                } else {
                    FlexTypeEnum::String
                }
            })
            .collect()
    }
}

/// Parse a slice of raw rows into column-oriented FlexType vectors, in parallel.
///
/// Divides rows into sub-chunks, processes each via rayon, then concatenates.
pub fn parse_rows_slice_parallel(
    rows: &[Vec<String>],
    column_types: &[FlexTypeEnum],
    na_values: &HashSet<String>,
    output_indices: Option<&[usize]>,
) -> Result<Vec<Vec<FlexType>>> {
    use rayon::prelude::*;

    let all_indices: Vec<usize>;
    let col_indices: &[usize] = if let Some(out) = output_indices {
        out
    } else {
        all_indices = (0..column_types.len()).collect();
        &all_indices
    };
    let n_out = col_indices.len();

    if rows.is_empty() {
        return Ok(vec![Vec::new(); n_out]);
    }

    const SUB_CHUNK_SIZE: usize = 4096;

    if rows.len() <= SUB_CHUNK_SIZE {
        // Small: parse sequentially
        return parse_rows_slice(rows, column_types, na_values, col_indices);
    }

    let num_sub = rows.len().div_ceil(SUB_CHUNK_SIZE);
    let sub_results: Vec<Result<Vec<Vec<FlexType>>>> = (0..num_sub)
        .into_par_iter()
        .map(|idx| {
            let start = idx * SUB_CHUNK_SIZE;
            let end = (start + SUB_CHUNK_SIZE).min(rows.len());
            parse_rows_slice(&rows[start..end], column_types, na_values, col_indices)
        })
        .collect();

    let mut combined: Vec<Vec<FlexType>> = (0..n_out)
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for result in sub_results {
        let cols = result?;
        for (i, col) in cols.into_iter().enumerate() {
            combined[i].extend(col);
        }
    }

    Ok(combined)
}

/// Sequential parse of a row slice into column-oriented vectors.
fn parse_rows_slice(
    rows: &[Vec<String>],
    column_types: &[FlexTypeEnum],
    na_values: &HashSet<String>,
    col_indices: &[usize],
) -> Result<Vec<Vec<FlexType>>> {
    let mut col_vecs: Vec<Vec<FlexType>> = col_indices
        .iter()
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();

    for row in rows {
        for (out_col, &src_col) in col_indices.iter().enumerate() {
            let val_str = if src_col < row.len() { &row[src_col] } else { "" };
            let val = parse_cell(val_str, column_types[src_col], na_values)?;
            col_vecs[out_col].push(val);
        }
    }

    Ok(col_vecs)
}

/// Two-pass streaming CSV parser.
///
/// Pass 1 (`open`): Reads the file in chunks, tokenizes with resumable state,
/// and infers column types. O(chunk_size) memory.
///
/// Pass 2 (`parse_chunks`): Re-reads the file, tokenizes, and parses each
/// chunk in parallel via rayon, calling `consumer` with column vectors.
pub struct CsvStreamingParse {
    pub column_names: Vec<String>,
    pub column_types: Vec<FlexTypeEnum>,
    path: String,
    options: CsvOptions,
    col_types_full: Vec<FlexTypeEnum>,
    output_indices: Option<Vec<usize>>,
}

/// Default chunk size for streaming CSV (50 MB).
const CSV_CHUNK_SIZE: usize = 50 * 1024 * 1024;

/// Number of rows to sample for type inference.
const TYPE_INFERENCE_ROWS: usize = 100;

/// Read buffer for type inference pass (1 MB — only need a few hundred rows).
const INFERENCE_READ_SIZE: usize = 1024 * 1024;

impl CsvStreamingParse {
    /// Pass 1: Infer schema by reading the first few hundred rows.
    ///
    /// Reads small chunks (1 MB) from the start of the file and infers column
    /// types from the first `TYPE_INFERENCE_ROWS` data rows. Stops reading
    /// as soon as enough rows have been observed.
    pub fn open(path: &str, options: &CsvOptions) -> Result<Self> {
        use super::csv_parallel_tokenizer::{ByteConfig, parallel_tokenize, parallel_tokenize_eof};

        let config = options_to_config(options);
        let bcfg = ByteConfig::from_config(&config)?;
        let na_values: HashSet<String> = options.na_values.iter().cloned().collect();
        let hint_map: std::collections::HashMap<&str, FlexTypeEnum> = options
            .type_hints
            .iter()
            .map(|(n, t)| (n.as_str(), *t))
            .collect();

        let mut file = std::fs::File::open(path).map_err(SFrameError::Io)?;
        let mut buffer: Vec<u8> = Vec::new();
        let mut column_names: Option<Vec<String>> = None;
        let mut infer_state: Option<TypeInferenceState> = None;
        let mut rows_skipped = 0usize;
        let mut row_count = 0usize;
        let infer_limit = options.row_limit.unwrap_or(TYPE_INFERENCE_ROWS);
        let mut done = false;

        loop {
            // Read small chunks — we only need enough for inference.
            let old_len = buffer.len();
            buffer.resize(old_len + INFERENCE_READ_SIZE, 0);
            let mut filled = old_len;
            loop {
                match std::io::Read::read(&mut file, &mut buffer[filled..old_len + INFERENCE_READ_SIZE]) {
                    Ok(0) => break,
                    Ok(n) => filled += n,
                    Err(e) => return Err(SFrameError::Io(e)),
                }
                if filled >= old_len + INFERENCE_READ_SIZE {
                    break;
                }
            }
            buffer.truncate(filled);
            let is_eof = filled == old_len; // no new bytes read

            if buffer.is_empty() {
                break;
            }

            // Tokenize this chunk
            let (rows, last_parsed) = if is_eof {
                parallel_tokenize_eof(&buffer, &bcfg)
            } else {
                parallel_tokenize(&buffer, &bcfg)
            };

            // Shift buffer: keep unparsed remainder
            if last_parsed >= buffer.len() {
                buffer.clear();
            } else {
                buffer = buffer[last_parsed..].to_vec();
            }

            // Process rows for schema inference
            for row in rows {
                // Skip empty rows (including comment lines which become [""])
                if row.is_empty() || (row.len() == 1 && row[0].trim().is_empty()) {
                    continue;
                }

                // Check for comment lines: if first field starts with comment char
                if let Some(cc) = config.comment_char {
                    if let Some(first) = row.first() {
                        let trimmed = first.trim_start();
                        if trimmed.starts_with(cc) {
                            continue;
                        }
                    }
                }

                // Skip rows
                if rows_skipped < config.skip_rows {
                    rows_skipped += 1;
                    continue;
                }

                // Handle header (first data line)
                if column_names.is_none() {
                    if config.has_header {
                        column_names = Some(row);
                    } else {
                        let n = row.len();
                        column_names = Some((0..n).map(|i| format!("X{}", i + 1)).collect());
                        let mut state = TypeInferenceState::new(n);
                        state.observe_row(&row, &na_values);
                        infer_state = Some(state);
                        row_count += 1;
                    }
                    continue;
                }

                // Type inference
                let ncols = column_names.as_ref().unwrap().len();
                if infer_state.is_none() {
                    infer_state = Some(TypeInferenceState::new(ncols));
                }
                infer_state.as_mut().unwrap().observe_row(&row, &na_values);
                row_count += 1;

                // Stop once we have enough rows for inference
                if row_count >= infer_limit {
                    done = true;
                    break;
                }
            }

            if is_eof || done {
                break;
            }
        }

        let names = column_names.unwrap_or_default();
        let ncols = names.len();
        let infer = infer_state.unwrap_or_else(|| TypeInferenceState::new(ncols));
        let col_types = infer.finalize(&hint_map, &names);

        let output_indices = options.output_columns.as_ref().map(|out_cols| {
            out_cols
                .iter()
                .filter_map(|name| names.iter().position(|n| n == name))
                .collect::<Vec<usize>>()
        });

        let out_names: Vec<String> = output_indices
            .as_ref()
            .map(|idx| idx.iter().map(|&i| names[i].clone()).collect())
            .unwrap_or_else(|| names.clone());
        let out_types: Vec<FlexTypeEnum> = output_indices
            .as_ref()
            .map(|idx| idx.iter().map(|&i| col_types[i]).collect())
            .unwrap_or_else(|| col_types.clone());

        Ok(CsvStreamingParse {
            column_names: out_names,
            column_types: out_types,
            path: path.to_string(),
            options: options.clone(),
            col_types_full: col_types,
            output_indices,
        })
    }

    /// Pass 2: Parse chunks in parallel, calling `consumer` for each chunk.
    ///
    /// Architecture: a background I/O thread reads 50MB raw byte chunks and
    /// sends them through a bounded channel. The main thread dispatches
    /// parallel tokenize+parse work via rayon and calls `consumer` with
    /// column-major vectors. This overlaps disk I/O with CPU processing
    /// (double-buffering).
    ///
    /// Once the header and skip_rows are consumed (typically the first few
    /// lines), all subsequent chunks use the fused `parallel_tokenize_and_parse`
    /// path which tokenizes and parses in a single parallel pass, eliminating
    /// the intermediate `Vec<Vec<String>>` allocation and the serial
    /// row-major → column-major transpose.
    pub fn parse_chunks<F>(&self, mut consumer: F) -> Result<()>
    where
        F: FnMut(Vec<Vec<FlexType>>) -> Result<()>,
    {
        use super::csv_parallel_tokenizer::{
            ByteConfig, parallel_tokenize, parallel_tokenize_eof,
            parallel_tokenize_and_parse,
        };
        use rayon::prelude::*;

        let config = options_to_config(&self.options);
        let bcfg = ByteConfig::from_config(&config)?;
        let na_values: HashSet<String> = self.options.na_values.iter().cloned().collect();
        let path = self.path.clone();
        let has_header = self.options.has_header;
        let skip_rows = self.options.skip_rows;
        let row_limit = self.options.row_limit;

        let col_types = &self.col_types_full;
        let col_indices: Vec<usize> = if let Some(ref out) = self.output_indices {
            out.clone()
        } else {
            (0..col_types.len()).collect()
        };
        let n_out = col_indices.len();

        // Background I/O thread: reads raw byte chunks, sends through channel.
        // Capacity=2 so at most 2 chunks are buffered ahead (double-buffering).
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<(Vec<u8>, bool)>>(2);

        std::thread::spawn(move || {
            let mut file = match std::fs::File::open(&path) {
                Ok(f) => f,
                Err(e) => {
                    let _ = tx.send(Err(SFrameError::Io(e)));
                    return;
                }
            };
            loop {
                // Read a full chunk using read_exact-like loop to avoid
                // short reads being confused with EOF.
                let mut buf = vec![0u8; CSV_CHUNK_SIZE];
                let mut filled = 0;
                loop {
                    match std::io::Read::read(&mut file, &mut buf[filled..]) {
                        Ok(0) => break,
                        Ok(n) => filled += n,
                        Err(e) => {
                            let _ = tx.send(Err(SFrameError::Io(e)));
                            return;
                        }
                    }
                    if filled >= CSV_CHUNK_SIZE {
                        break;
                    }
                }
                if filled == 0 {
                    // True EOF: send empty marker so main thread can flush leftover
                    let _ = tx.send(Ok((Vec::new(), true)));
                    return;
                }
                buf.truncate(filled);
                let is_eof = filled < CSV_CHUNK_SIZE;
                if tx.send(Ok((buf, is_eof))).is_err() {
                    return; // consumer dropped
                }
                if is_eof {
                    return;
                }
            }
        });

        // Main thread: parallel tokenize + parse
        let mut header_consumed = !has_header;
        let mut rows_skipped = 0usize;
        let mut total_rows = 0usize;
        let mut leftover: Vec<u8> = Vec::new();

        for msg in rx {
            let (chunk, is_eof) = msg?;

            // Combine leftover + new chunk
            let buffer = if leftover.is_empty() {
                chunk
            } else {
                let mut combined = std::mem::take(&mut leftover);
                combined.extend_from_slice(&chunk);
                combined
            };

            if buffer.is_empty() {
                break;
            }

            // ---------------------------------------------------------------
            // Fast path: header consumed, skip_rows done → fused tokenize+parse
            // ---------------------------------------------------------------
            if header_consumed && rows_skipped >= skip_rows {
                let (col_vecs, last_parsed) = parallel_tokenize_and_parse(
                    &buffer,
                    &bcfg,
                    is_eof,
                    col_types,
                    &col_indices,
                    |val, dtype| {
                        parse_cell(val, dtype, &na_values).unwrap_or(FlexType::Undefined)
                    },
                );

                if last_parsed < buffer.len() {
                    leftover = buffer[last_parsed..].to_vec();
                }

                if !col_vecs.is_empty() && !col_vecs[0].is_empty() {
                    let chunk_rows = col_vecs[0].len();

                    // Apply row limit
                    if let Some(limit) = row_limit {
                        let remaining = limit.saturating_sub(total_rows);
                        if remaining == 0 {
                            break;
                        }
                        if chunk_rows > remaining {
                            let truncated: Vec<Vec<FlexType>> = col_vecs
                                .into_iter()
                                .map(|mut col| {
                                    col.truncate(remaining);
                                    col
                                })
                                .collect();
                            consumer(truncated)?;
                            break;
                        }
                        total_rows += chunk_rows;
                    } else {
                        total_rows += chunk_rows;
                    }
                    consumer(col_vecs)?;
                }

                if is_eof {
                    break;
                }
                continue;
            }

            // ---------------------------------------------------------------
            // Slow path: still processing header / skip_rows.
            // Uses parallel_tokenize → filter → par_iter parse.
            // Only runs for the first chunk (usually a handful of lines).
            // ---------------------------------------------------------------
            let (rows, last_parsed) = if is_eof {
                parallel_tokenize_eof(&buffer, &bcfg)
            } else {
                parallel_tokenize(&buffer, &bcfg)
            };

            // Save unparsed remainder
            if last_parsed < buffer.len() {
                leftover = buffer[last_parsed..].to_vec();
            }

            // Filter rows: skip empty/comment rows, header, skip_rows, row_limit
            let mut data_rows: Vec<Vec<String>> = Vec::with_capacity(rows.len());
            let mut hit_limit = false;
            for row in rows {
                // Skip empty rows (including comment lines which become [""])
                if row.is_empty() || (row.len() == 1 && row[0].trim().is_empty()) {
                    continue;
                }
                // Check for comment lines
                if let Some(cc) = config.comment_char {
                    if let Some(first) = row.first() {
                        let trimmed = first.trim_start();
                        if trimmed.starts_with(cc) {
                            continue;
                        }
                    }
                }
                // Skip rows
                if rows_skipped < skip_rows {
                    rows_skipped += 1;
                    continue;
                }
                // Skip header row
                if !header_consumed {
                    header_consumed = true;
                    continue;
                }
                // Row limit
                if let Some(limit) = row_limit {
                    if total_rows >= limit {
                        hit_limit = true;
                        break;
                    }
                }
                data_rows.push(row);
                total_rows += 1;
            }

            if !data_rows.is_empty() {
                // Parallel parse to FlexType
                let row_results: Vec<Result<Vec<FlexType>>> = data_rows
                    .par_iter()
                    .map(|fields| {
                        let mut row = Vec::with_capacity(n_out);
                        for &src_col in &col_indices {
                            let val_str = if src_col < fields.len() {
                                &fields[src_col]
                            } else {
                                ""
                            };
                            row.push(parse_cell(val_str, col_types[src_col], &na_values)?);
                        }
                        Ok(row)
                    })
                    .collect();

                // Transpose row-major → column-major
                let mut col_vecs: Vec<Vec<FlexType>> = (0..n_out)
                    .map(|_| Vec::with_capacity(row_results.len()))
                    .collect();
                for row_result in row_results {
                    let row = row_result?;
                    for (i, val) in row.into_iter().enumerate() {
                        col_vecs[i].push(val);
                    }
                }

                consumer(col_vecs)?;
            }

            if hit_limit || is_eof {
                break;
            }
        }

        Ok(())
    }
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
        format!("{manifest}/../../samples")
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
            other => panic!("Expected Vector, got {other:?}"),
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
