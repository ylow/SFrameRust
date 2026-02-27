//! Round-trip integration test: read business.sf → write → read back → compare.
//!
//! Reads the sample business.sf, extracts all data, writes it out as a new
//! SFrame using the writer, then reads it back and verifies every value matches.

use sframe_storage::sframe_reader::SFrameReader;
use sframe_storage::sframe_writer::write_sframe;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

fn samples_dir() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/../../samples", manifest)
}

/// Compare two FlexType values for equality (floats within tolerance).
fn flex_eq(a: &FlexType, b: &FlexType) -> bool {
    match (a, b) {
        (FlexType::Integer(x), FlexType::Integer(y)) => x == y,
        (FlexType::Float(x), FlexType::Float(y)) => {
            if x.is_nan() && y.is_nan() {
                true
            } else {
                (x - y).abs() < 1e-10
                    || (*y != 0.0 && ((x - y) / y).abs() < 1e-10)
            }
        }
        (FlexType::String(x), FlexType::String(y)) => x == y,
        (FlexType::Vector(x), FlexType::Vector(y)) => {
            if x.len() != y.len() {
                return false;
            }
            x.iter()
                .zip(y.iter())
                .all(|(a, b)| (a - b).abs() < 1e-10)
        }
        (FlexType::Undefined, FlexType::Undefined) => true,
        (FlexType::List(x), FlexType::List(y)) => {
            if x.len() != y.len() {
                return false;
            }
            x.iter().zip(y.iter()).all(|(a, b)| flex_eq(a, b))
        }
        (FlexType::Dict(x), FlexType::Dict(y)) => {
            if x.len() != y.len() {
                return false;
            }
            x.iter()
                .zip(y.iter())
                .all(|((k1, v1), (k2, v2))| flex_eq(k1, k2) && flex_eq(v1, v2))
        }
        (FlexType::DateTime(x), FlexType::DateTime(y)) => x == y,
        _ => false,
    }
}

#[test]
fn test_roundtrip_business_sf() {
    // Step 1: Read the original SFrame
    let mut sf = SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap();
    let nrows = sf.num_rows() as usize;
    let col_names: Vec<String> = sf.column_names().to_vec();
    let num_cols = col_names.len();

    // Get column types from the group_index
    let column_types: Vec<FlexTypeEnum> = sf
        .group_index
        .columns
        .iter()
        .map(|c| c.dtype)
        .collect();

    // Read all columns
    let mut all_columns: Vec<Vec<FlexType>> = Vec::new();
    for name in &col_names {
        all_columns.push(sf.read_column_by_name(name).unwrap());
    }

    // Transpose: columns → rows for the writer
    let mut rows: Vec<Vec<FlexType>> = Vec::with_capacity(nrows);
    for row_idx in 0..nrows {
        let mut row = Vec::with_capacity(num_cols);
        for col in &all_columns {
            row.push(col[row_idx].clone());
        }
        rows.push(row);
    }

    // Step 2: Write to a temp directory
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("roundtrip.sf");
    let out_path_str = out_path.to_str().unwrap();

    let col_name_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();

    write_sframe(out_path_str, &col_name_refs, &column_types, &rows).unwrap();

    // Step 3: Read back and verify
    let mut sf2 = SFrameReader::open(out_path_str).unwrap();

    assert_eq!(sf2.num_rows(), nrows as u64, "Row count mismatch");
    assert_eq!(sf2.num_columns(), num_cols, "Column count mismatch");
    assert_eq!(sf2.column_names(), &col_names[..], "Column names mismatch");

    // Verify column types match
    let column_types2: Vec<FlexTypeEnum> = sf2
        .group_index
        .columns
        .iter()
        .map(|c| c.dtype)
        .collect();
    assert_eq!(column_types2, column_types, "Column types mismatch");

    // Verify all data matches
    for (col_idx, name) in col_names.iter().enumerate() {
        let col2 = sf2.read_column_by_name(name).unwrap();
        assert_eq!(
            col2.len(),
            all_columns[col_idx].len(),
            "Column '{}' length mismatch: original={}, roundtrip={}",
            name,
            all_columns[col_idx].len(),
            col2.len()
        );

        for (row_idx, (orig, round)) in
            all_columns[col_idx].iter().zip(col2.iter()).enumerate()
        {
            assert!(
                flex_eq(orig, round),
                "Mismatch at column '{}' row {}: original={:?}, roundtrip={:?}",
                name,
                row_idx,
                orig,
                round
            );
        }
    }
}

/// Smaller test with hand-crafted data to catch edge cases.
#[test]
fn test_roundtrip_small() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("small.sf");
    let out_path_str = out_path.to_str().unwrap();

    let col_names = &["id", "name", "score", "tags"];
    let col_types = &[
        FlexTypeEnum::Integer,
        FlexTypeEnum::String,
        FlexTypeEnum::Float,
        FlexTypeEnum::Vector,
    ];

    let rows = vec![
        vec![
            FlexType::Integer(1),
            FlexType::String("alice".into()),
            FlexType::Float(95.5),
            FlexType::Vector(vec![1.0, 2.0, 3.0].into()),
        ],
        vec![
            FlexType::Integer(2),
            FlexType::String("bob".into()),
            FlexType::Float(87.3),
            FlexType::Vector(vec![4.0, 5.0].into()),
        ],
        vec![
            FlexType::Integer(3),
            FlexType::String("charlie".into()),
            FlexType::Float(92.1),
            FlexType::Vector(vec![].into()),
        ],
    ];

    write_sframe(out_path_str, col_names, col_types, &rows).unwrap();

    let mut sf = SFrameReader::open(out_path_str).unwrap();
    assert_eq!(sf.num_rows(), 3);
    assert_eq!(sf.num_columns(), 4);

    let ids = sf.read_column_by_name("id").unwrap();
    assert_eq!(ids[0], FlexType::Integer(1));
    assert_eq!(ids[1], FlexType::Integer(2));
    assert_eq!(ids[2], FlexType::Integer(3));

    let names = sf.read_column_by_name("name").unwrap();
    assert_eq!(names[0], FlexType::String("alice".into()));
    assert_eq!(names[1], FlexType::String("bob".into()));
    assert_eq!(names[2], FlexType::String("charlie".into()));

    let scores = sf.read_column_by_name("score").unwrap();
    match &scores[0] {
        FlexType::Float(v) => assert!((v - 95.5).abs() < 1e-10),
        other => panic!("Expected Float, got {:?}", other),
    }

    let tags = sf.read_column_by_name("tags").unwrap();
    match &tags[0] {
        FlexType::Vector(v) => assert_eq!(v.as_ref(), &[1.0, 2.0, 3.0]),
        other => panic!("Expected Vector, got {:?}", other),
    }
    match &tags[2] {
        FlexType::Vector(v) => assert!(v.is_empty()),
        other => panic!("Expected empty Vector, got {:?}", other),
    }
}

/// Test empty SFrame roundtrip.
#[test]
fn test_roundtrip_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("empty.sf");
    let out_path_str = out_path.to_str().unwrap();

    let col_names = &["x", "y"];
    let col_types = &[FlexTypeEnum::Integer, FlexTypeEnum::Float];
    let rows: Vec<Vec<FlexType>> = vec![];

    write_sframe(out_path_str, col_names, col_types, &rows).unwrap();

    let sf = SFrameReader::open(out_path_str).unwrap();
    assert_eq!(sf.num_rows(), 0);
    assert_eq!(sf.num_columns(), 2);
    assert_eq!(sf.column_names()[0], "x");
    assert_eq!(sf.column_names()[1], "y");
}
