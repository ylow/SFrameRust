//! JSON reader/writer for SFrame data.
//!
//! Supports JSON Lines format (one JSON object per line) and
//! FlexType ↔ JSON value conversion.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use serde_json::Value as JsonValue;

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::SFrameRows;

#[cfg(test)]
use crate::batch::ColumnData;

/// Convert a FlexType to a JSON value.
pub fn flex_to_json(val: &FlexType) -> JsonValue {
    match val {
        FlexType::Undefined => JsonValue::Null,
        FlexType::Integer(i) => JsonValue::Number(serde_json::Number::from(*i)),
        FlexType::Float(f) => {
            if let Some(n) = serde_json::Number::from_f64(*f) {
                JsonValue::Number(n)
            } else {
                // NaN/Infinity → null
                JsonValue::Null
            }
        }
        FlexType::String(s) => JsonValue::String(s.to_string()),
        FlexType::Vector(v) => {
            let arr: Vec<JsonValue> = v.iter().map(|&f| {
                serde_json::Number::from_f64(f)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null)
            }).collect();
            JsonValue::Array(arr)
        }
        FlexType::List(l) => {
            let arr: Vec<JsonValue> = l.iter().map(|v| flex_to_json(v)).collect();
            JsonValue::Array(arr)
        }
        FlexType::Dict(d) => {
            let mut map = serde_json::Map::new();
            for (k, v) in d.iter() {
                let key = format!("{}", k);
                map.insert(key, flex_to_json(v));
            }
            JsonValue::Object(map)
        }
        FlexType::DateTime(dt) => JsonValue::Number(serde_json::Number::from(dt.posix_timestamp)),
    }
}

/// Convert a JSON value to a FlexType.
pub fn json_to_flex(val: &JsonValue) -> FlexType {
    match val {
        JsonValue::Null => FlexType::Undefined,
        JsonValue::Bool(b) => FlexType::Integer(if *b { 1 } else { 0 }),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                FlexType::Integer(i)
            } else if let Some(f) = n.as_f64() {
                FlexType::Float(f)
            } else {
                FlexType::Undefined
            }
        }
        JsonValue::String(s) => FlexType::String(Arc::from(s.as_str())),
        JsonValue::Array(arr) => {
            // Check if all elements are numbers → Vector, else List
            let all_numeric = arr.iter().all(|v| v.is_number());
            if all_numeric && !arr.is_empty() {
                let floats: Vec<f64> = arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
                FlexType::Vector(Arc::from(floats))
            } else {
                let items: Vec<FlexType> = arr.iter().map(json_to_flex).collect();
                FlexType::List(Arc::from(items))
            }
        }
        JsonValue::Object(map) => {
            let entries: Vec<(FlexType, FlexType)> = map
                .iter()
                .map(|(k, v)| (FlexType::String(Arc::from(k.as_str())), json_to_flex(v)))
                .collect();
            FlexType::Dict(Arc::from(entries))
        }
    }
}

/// Write SFrameRows to a JSON Lines file.
pub fn write_json_file(
    path: &str,
    batch: &SFrameRows,
    column_names: &[String],
) -> Result<()> {
    let content = write_json_string(batch, column_names)?;
    std::fs::write(path, content).map_err(SFrameError::Io)
}

/// Write SFrameRows to a JSON Lines string.
pub fn write_json_string(batch: &SFrameRows, column_names: &[String]) -> Result<String> {
    let nrows = batch.num_rows();
    let ncols = batch.num_columns();
    let mut output = String::new();

    for row in 0..nrows {
        let mut map = serde_json::Map::new();
        for col in 0..ncols {
            let val = batch.column(col).get(row);
            map.insert(column_names[col].clone(), flex_to_json(&val));
        }
        let json = serde_json::to_string(&JsonValue::Object(map))
            .map_err(|e| SFrameError::Format(format!("JSON serialization error: {}", e)))?;
        output.push_str(&json);
        output.push('\n');
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Chunked API
// ---------------------------------------------------------------------------

/// Parsed JSON data with schema, ready for chunked column conversion.
pub struct JsonSchema {
    pub column_names: Vec<String>,
    pub column_types: Vec<FlexTypeEnum>,
    /// Row-major parsed data. Each row maps column name → FlexType.
    pub rows: Vec<HashMap<String, FlexType>>,
}

/// Parse a JSON Lines file into a [`JsonSchema`] without building SFrameRows.
pub fn parse_json_file_schema(path: &str) -> Result<JsonSchema> {
    let file = std::fs::File::open(path).map_err(SFrameError::Io)?;
    let reader = BufReader::new(file);
    build_json_schema(reader)
}

/// Parse a JSON Lines string into a [`JsonSchema`].
pub fn parse_json_string_schema(content: &str) -> Result<JsonSchema> {
    let reader = BufReader::new(content.as_bytes());
    build_json_schema(reader)
}

/// Convert rows `[start..end)` from a [`JsonSchema`] into column-oriented
/// `Vec<Vec<FlexType>>`.
pub fn rows_to_columns_range(
    schema: &JsonSchema,
    start: usize,
    end: usize,
) -> Vec<Vec<FlexType>> {
    let end = end.min(schema.rows.len());
    let chunk_len = end.saturating_sub(start);
    let ncols = schema.column_names.len();

    let mut col_vecs: Vec<Vec<FlexType>> = (0..ncols)
        .map(|_| Vec::with_capacity(chunk_len))
        .collect();

    for row_idx in start..end {
        let row = &schema.rows[row_idx];
        for (i, name) in schema.column_names.iter().enumerate() {
            let val = row.get(name).cloned().unwrap_or(FlexType::Undefined);
            col_vecs[i].push(val);
        }
    }

    col_vecs
}

fn build_json_schema<R: BufRead>(reader: R) -> Result<JsonSchema> {
    let mut rows: Vec<HashMap<String, FlexType>> = Vec::new();
    let mut col_order: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(SFrameError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let json: JsonValue = serde_json::from_str(trimmed)
            .map_err(|e| SFrameError::Format(format!("JSON parse error: {}", e)))?;

        if let JsonValue::Object(map) = json {
            let mut row = HashMap::new();
            for (key, val) in &map {
                if !col_order.contains(key) {
                    col_order.push(key.clone());
                }
                row.insert(key.clone(), json_to_flex(val));
            }
            rows.push(row);
        } else {
            return Err(SFrameError::Format(
                "Each line must be a JSON object".to_string(),
            ));
        }
    }

    // Infer types from first non-null value in each column
    let col_types: Vec<FlexTypeEnum> = col_order
        .iter()
        .map(|name| {
            for row in &rows {
                if let Some(val) = row.get(name) {
                    match val {
                        FlexType::Undefined => continue,
                        FlexType::Integer(_) => return FlexTypeEnum::Integer,
                        FlexType::Float(_) => return FlexTypeEnum::Float,
                        FlexType::String(_) => return FlexTypeEnum::String,
                        FlexType::Vector(_) => return FlexTypeEnum::Vector,
                        FlexType::List(_) => return FlexTypeEnum::List,
                        FlexType::Dict(_) => return FlexTypeEnum::Dict,
                        FlexType::DateTime(_) => return FlexTypeEnum::DateTime,
                    }
                }
            }
            FlexTypeEnum::String
        })
        .collect();

    Ok(JsonSchema {
        column_names: col_order,
        column_types: col_types,
        rows,
    })
}

// ---------------------------------------------------------------------------
// Legacy API (delegates to chunked pipeline)
// ---------------------------------------------------------------------------

/// Read a JSON Lines file into column data.
///
/// Returns `(column_names, SFrameRows)`.
pub fn read_json_file(path: &str) -> Result<(Vec<String>, SFrameRows)> {
    let schema = parse_json_file_schema(path)?;
    json_schema_to_batch(schema)
}

/// Read JSON Lines from a string.
pub fn read_json_string(content: &str) -> Result<(Vec<String>, SFrameRows)> {
    let schema = parse_json_string_schema(content)?;
    json_schema_to_batch(schema)
}

fn json_schema_to_batch(schema: JsonSchema) -> Result<(Vec<String>, SFrameRows)> {
    if schema.rows.is_empty() {
        return Ok((vec![], SFrameRows::empty(&[])));
    }
    let col_vecs = rows_to_columns_range(&schema, 0, schema.rows.len());
    let batch = SFrameRows::from_column_vecs(col_vecs, &schema.column_types)?;
    Ok((schema.column_names, batch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flex_to_json_roundtrip() {
        let vals = vec![
            FlexType::Integer(42),
            FlexType::Float(3.14),
            FlexType::String("hello".into()),
            FlexType::Undefined,
        ];

        for val in &vals {
            let json = flex_to_json(val);
            let back = json_to_flex(&json);
            match (val, &back) {
                (FlexType::Undefined, FlexType::Undefined) => {}
                (FlexType::Integer(a), FlexType::Integer(b)) => assert_eq!(a, b),
                (FlexType::Float(a), FlexType::Float(b)) => assert!((a - b).abs() < 1e-10),
                (FlexType::String(a), FlexType::String(b)) => assert_eq!(&**a, &**b),
                _ => panic!("Roundtrip failed for {:?} → {:?}", val, back),
            }
        }
    }

    #[test]
    fn test_json_vector() {
        let v = FlexType::Vector(Arc::from(vec![1.0, 2.0, 3.0]));
        let json = flex_to_json(&v);
        assert!(json.is_array());
        let back = json_to_flex(&json);
        assert!(matches!(back, FlexType::Vector(_)));
    }

    #[test]
    fn test_write_json_string() {
        let batch = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1), Some(2)]),
            ColumnData::String(vec![Some("alice".into()), Some("bob".into())]),
        ]).unwrap();

        let names = vec!["id".to_string(), "name".to_string()];
        let json = write_json_string(&batch, &names).unwrap();

        let lines: Vec<&str> = json.trim().split('\n').collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"id\":1"));
        assert!(lines[0].contains("\"name\":\"alice\""));
    }

    #[test]
    fn test_read_json_string() {
        let input = r#"{"id": 1, "name": "alice"}
{"id": 2, "name": "bob"}
{"id": 3, "name": "charlie"}
"#;

        let (names, batch) = read_json_string(input).unwrap();
        assert_eq!(names, vec!["id", "name"]);
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn test_json_roundtrip() {
        let batch = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1), Some(2)]),
            ColumnData::Float(vec![Some(1.5), Some(2.5)]),
            ColumnData::String(vec![Some("a".into()), Some("b".into())]),
        ]).unwrap();

        let names = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let json = write_json_string(&batch, &names).unwrap();
        let (read_names, read_batch) = read_json_string(&json).unwrap();

        assert_eq!(read_names, names);
        assert_eq!(read_batch.num_rows(), 2);
        assert_eq!(read_batch.num_columns(), 3);
    }
}
