//! Integration test: read business.sf and verify against business.csv.
//!
//! The SFrame may not preserve CSV row order, so we match rows by
//! business_id and compare all columns.

use std::collections::HashMap;

use sframe_storage::sframe_reader::SFrameReader;
use sframe_types::flex_type::FlexType;

fn samples_dir() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{manifest}/../../samples")
}

/// Parse the categories column "[4 5 6]" → Vec<f64>.
fn parse_categories(s: &str) -> Vec<f64> {
    let trimmed = s.trim_start_matches('[').trim_end_matches(']').trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split_whitespace()
        .map(|tok| tok.parse::<f64>().unwrap())
        .collect()
}

/// Load CSV into a map from business_id → HashMap<column_name, value_string>.
fn load_csv() -> HashMap<String, HashMap<String, String>> {
    let path = format!("{}/business.csv", samples_dir());
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)
        .unwrap();

    let headers: Vec<String> = rdr
        .headers()
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut rows = HashMap::new();

    for record in rdr.records() {
        let record = record.unwrap();
        let business_id = record[0].to_string();
        let mut row = HashMap::new();
        for (i, h) in headers.iter().enumerate() {
            row.insert(h.clone(), record[i].to_string());
        }
        rows.insert(business_id, row);
    }

    rows
}

fn assert_flex_matches_csv(
    col_name: &str,
    row_id: &str,
    sf_val: &FlexType,
    csv_val: &str,
) {
    match sf_val {
        FlexType::Integer(v) => {
            let expected: i64 = csv_val.parse().unwrap_or_else(|_| {
                panic!(
                    "Column '{col_name}' id '{row_id}': CSV value '{csv_val}' not an integer"
                )
            });
            assert_eq!(
                *v, expected,
                "Column '{col_name}' id '{row_id}': SFrame={v}, CSV={expected}"
            );
        }
        FlexType::Float(v) => {
            let expected: f64 = csv_val.parse().unwrap_or_else(|_| {
                panic!(
                    "Column '{col_name}' id '{row_id}': CSV value '{csv_val}' not a float"
                )
            });
            assert!(
                (v - expected).abs() < 1e-6
                    || (expected != 0.0 && ((v - expected) / expected).abs() < 1e-10),
                "Column '{col_name}' id '{row_id}': SFrame={v}, CSV={expected}"
            );
        }
        FlexType::String(s) => {
            // Python's CSV writer doubles backslashes (e.g. \n → \\n in CSV).
            // Undo this to get the original SFrame string value.
            let csv_unescaped = csv_val.replace("\\\\", "\\");
            assert_eq!(
                s.as_ref(),
                &csv_unescaped,
                "Column '{col_name}' id '{row_id}': SFrame='{s}', CSV='{csv_unescaped}'"
            );
        }
        FlexType::Vector(v) => {
            let expected = parse_categories(csv_val);
            let actual: &[f64] = v.as_ref();
            assert_eq!(
                actual.len(),
                expected.len(),
                "Column '{}' id '{}': vector lengths differ ({} vs {})",
                col_name,
                row_id,
                actual.len(),
                expected.len()
            );
            for (j, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() < 1e-10,
                    "Column '{col_name}' id '{row_id}' element {j}: SFrame={a}, CSV={e}"
                );
            }
        }
        FlexType::Undefined => {
            assert!(
                csv_val.is_empty() || csv_val == "NA",
                "Column '{col_name}' id '{row_id}': SFrame=Undefined, CSV='{csv_val}'"
            );
        }
        other => {
            panic!(
                "Column '{col_name}' id '{row_id}': unexpected type {other:?}"
            );
        }
    }
}

#[test]
fn test_all_columns_match_csv() {
    let csv_rows = load_csv();
    let mut sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();

    let nrows = sf.num_rows() as usize;
    assert_eq!(nrows, 11536);
    assert_eq!(csv_rows.len(), 11536);

    // Read all columns from SFrame
    let col_names = sf.column_names().to_vec();
    let mut sf_columns: HashMap<String, Vec<FlexType>> = HashMap::new();
    for col_name in &col_names {
        let col = sf.read_column_by_name(col_name)
            .unwrap_or_else(|e| panic!("Failed to read column '{col_name}': {e}"));
        sf_columns.insert(col_name.clone(), col);
    }

    // Read business_id column for row matching
    let id_col = &sf_columns["business_id"];

    // For each SFrame row, find the matching CSV row and compare all columns
    for i in 0..nrows {
        let business_id = match &id_col[i] {
            FlexType::String(s) => s.as_ref().to_string(),
            other => panic!("Row {i} business_id: expected String, got {other:?}"),
        };

        let csv_row = csv_rows.get(&business_id).unwrap_or_else(|| {
            panic!(
                "SFrame row {i} business_id '{business_id}' not found in CSV"
            )
        });

        for col_name in &col_names {
            let sf_val = &sf_columns[col_name][i];
            let csv_val = &csv_row[col_name];
            assert_flex_matches_csv(col_name, &business_id, sf_val, csv_val);
        }
    }
}

#[test]
fn test_column_names_match_csv_header() {
    let sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
    let expected = vec![
        "business_id",
        "categories",
        "city",
        "full_address",
        "latitude",
        "longitude",
        "name",
        "open",
        "review_count",
        "stars",
        "state",
        "type",
    ];
    assert_eq!(sf.column_names(), &expected);
}

#[test]
fn test_row_count() {
    let sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
    assert_eq!(sf.num_rows(), 11536);
    assert_eq!(sf.num_columns(), 12);
}
