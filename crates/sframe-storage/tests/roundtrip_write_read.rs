//! Round-trip integration test: read business.sf → write → read back → compare.
//!
//! Reads the sample business.sf, extracts all data, writes it out as a new
//! SFrame using the writer, then reads it back and verifies every value matches.

use sframe_storage::sframe_reader::SFrameReader;
use sframe_storage::sframe_writer::{write_sframe, SFrameWriter};
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

/// Test SFrameWriter with many small batches (cross-batch buffering).
///
/// Sends 100 batches of 10 rows each. The writer should coalesce them
/// into proper-sized blocks instead of writing 100 tiny blocks.
#[test]
fn test_streaming_writer_small_batches() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("small_batches.sf");
    let out_path_str = out_path.to_str().unwrap();

    let col_names = &["id", "value"];
    let col_types = &[FlexTypeEnum::Integer, FlexTypeEnum::Float];

    let mut writer = SFrameWriter::new(out_path_str, col_names, col_types).unwrap();

    let total_rows = 1000;
    let batch_size = 10;
    for batch_start in (0..total_rows).step_by(batch_size) {
        let ids: Vec<FlexType> = (batch_start..batch_start + batch_size as i64)
            .map(FlexType::Integer)
            .collect();
        let values: Vec<FlexType> = (batch_start..batch_start + batch_size as i64)
            .map(|i| FlexType::Float(i as f64 * 0.5))
            .collect();
        writer.write_columns(&[ids, values]).unwrap();
    }

    writer.finish().unwrap();

    // Read back and verify
    let mut sf = SFrameReader::open(out_path_str).unwrap();
    assert_eq!(sf.num_rows(), total_rows as u64);
    assert_eq!(sf.num_columns(), 2);

    let ids = sf.read_column_by_name("id").unwrap();
    assert_eq!(ids.len(), total_rows as usize);
    assert_eq!(ids[0], FlexType::Integer(0));
    assert_eq!(ids[999], FlexType::Integer(999));

    let values = sf.read_column_by_name("value").unwrap();
    assert!(flex_eq(&values[0], &FlexType::Float(0.0)));
    assert!(flex_eq(&values[999], &FlexType::Float(499.5)));
}

/// Test SFrameWriter with a single large batch.
#[test]
fn test_streaming_writer_large_batch() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("large_batch.sf");
    let out_path_str = out_path.to_str().unwrap();

    let col_names = &["x"];
    let col_types = &[FlexTypeEnum::Integer];

    let mut writer = SFrameWriter::new(out_path_str, col_names, col_types).unwrap();

    let n = 50_000;
    let data: Vec<FlexType> = (0..n).map(|i| FlexType::Integer(i)).collect();
    writer.write_columns(&[data]).unwrap();
    writer.finish().unwrap();

    let mut sf = SFrameReader::open(out_path_str).unwrap();
    assert_eq!(sf.num_rows(), n as u64);
    let col = sf.read_column_by_name("x").unwrap();
    assert_eq!(col[0], FlexType::Integer(0));
    assert_eq!(col[(n - 1) as usize], FlexType::Integer(n - 1));
}

/// Test SFrameWriter with empty finish (no data written).
#[test]
fn test_streaming_writer_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("empty_stream.sf");
    let out_path_str = out_path.to_str().unwrap();

    let col_names = &["a", "b"];
    let col_types = &[FlexTypeEnum::String, FlexTypeEnum::Integer];

    let writer = SFrameWriter::new(out_path_str, col_names, col_types).unwrap();
    writer.finish().unwrap();

    let sf = SFrameReader::open(out_path_str).unwrap();
    assert_eq!(sf.num_rows(), 0);
    assert_eq!(sf.num_columns(), 2);
}

/// Test SFrameWriter with batches of varying sizes.
#[test]
fn test_streaming_writer_variable_batch_sizes() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("variable_batches.sf");
    let out_path_str = out_path.to_str().unwrap();

    let col_names = &["val"];
    let col_types = &[FlexTypeEnum::Integer];

    let mut writer = SFrameWriter::new(out_path_str, col_names, col_types).unwrap();

    // Write batches of varying sizes: 1, 5, 100, 3, 50, 1000, 7
    let batch_sizes = [1, 5, 100, 3, 50, 1000, 7];
    let mut next_val: i64 = 0;
    let total: usize = batch_sizes.iter().sum();

    for &size in &batch_sizes {
        let data: Vec<FlexType> = (next_val..next_val + size as i64)
            .map(FlexType::Integer)
            .collect();
        writer.write_columns(&[data]).unwrap();
        next_val += size as i64;
    }

    writer.finish().unwrap();

    let mut sf = SFrameReader::open(out_path_str).unwrap();
    assert_eq!(sf.num_rows(), total as u64);
    let col = sf.read_column_by_name("val").unwrap();
    for i in 0..total {
        assert_eq!(col[i], FlexType::Integer(i as i64), "Mismatch at row {}", i);
    }
}

/// Test multi-segment writing with a small rows_per_segment threshold.
///
/// Writes 2500 rows with rows_per_segment=1000. Should produce 3 segments:
/// segment 0: 1000 rows, segment 1: 1000 rows, segment 2: 500 rows.
#[test]
fn test_multi_segment_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("multi_seg.sf");
    let out_path_str = out_path.to_str().unwrap();

    let col_names = &["id", "value"];
    let col_types = &[FlexTypeEnum::Integer, FlexTypeEnum::Float];

    let mut writer =
        SFrameWriter::with_segment_size(out_path_str, col_names, col_types, 1000).unwrap();

    let total_rows = 2500usize;
    for batch_start in (0..total_rows).step_by(100) {
        let batch_end = (batch_start + 100).min(total_rows);
        let ids: Vec<FlexType> = (batch_start..batch_end)
            .map(|i| FlexType::Integer(i as i64))
            .collect();
        let values: Vec<FlexType> = (batch_start..batch_end)
            .map(|i| FlexType::Float(i as f64 * 0.1))
            .collect();
        writer.write_columns(&[ids, values]).unwrap();
    }

    writer.finish().unwrap();

    // Verify that multiple segment files were created
    let sf = SFrameReader::open(out_path_str).unwrap();
    assert!(
        sf.segment_readers.len() > 1,
        "Expected multiple segments, got {}",
        sf.segment_readers.len()
    );

    // Read back and verify all data
    let mut sf = SFrameReader::open(out_path_str).unwrap();
    assert_eq!(sf.num_rows(), total_rows as u64);
    assert_eq!(sf.num_columns(), 2);

    let ids = sf.read_column_by_name("id").unwrap();
    assert_eq!(ids.len(), total_rows);
    for i in 0..total_rows {
        assert_eq!(ids[i], FlexType::Integer(i as i64), "id mismatch at row {}", i);
    }

    let values = sf.read_column_by_name("value").unwrap();
    assert_eq!(values.len(), total_rows);
    for i in 0..total_rows {
        assert!(
            flex_eq(&values[i], &FlexType::Float(i as f64 * 0.1)),
            "value mismatch at row {}: {:?}",
            i,
            values[i]
        );
    }
}

/// Test multi-segment with exact segment boundary (no remainder).
#[test]
fn test_multi_segment_exact_boundary() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("exact_boundary.sf");
    let out_path_str = out_path.to_str().unwrap();

    let col_names = &["x"];
    let col_types = &[FlexTypeEnum::Integer];

    // 2000 rows, 1000 per segment → exactly 2 segments
    let mut writer =
        SFrameWriter::with_segment_size(out_path_str, col_names, col_types, 1000).unwrap();

    let data: Vec<FlexType> = (0..2000).map(|i| FlexType::Integer(i)).collect();
    writer.write_columns(&[data]).unwrap();
    writer.finish().unwrap();

    let sf = SFrameReader::open(out_path_str).unwrap();
    assert_eq!(
        sf.segment_readers.len(),
        2,
        "Expected 2 segments for 2000 rows / 1000 per segment"
    );

    let mut sf = SFrameReader::open(out_path_str).unwrap();
    assert_eq!(sf.num_rows(), 2000);
    let col = sf.read_column_by_name("x").unwrap();
    for i in 0..2000 {
        assert_eq!(col[i as usize], FlexType::Integer(i));
    }
}
