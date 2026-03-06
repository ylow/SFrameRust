//! Test: manually write segment files, assemble SFrame metadata, read back.

use std::io::BufWriter;
use sframe_io::local_fs::LocalFileSystem;
use sframe_storage::segment_writer::SegmentWriter;
use sframe_storage::sframe_reader::SFrameReader;
use sframe_storage::sframe_writer::assemble_sframe_from_segments;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

#[test]
fn test_assemble_from_two_segments() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().to_str().unwrap();

    // Write segment 0: rows [1, 2, 3]
    let seg0_name = "seg.0000".to_string();
    let seg0_path = format!("{base_path}/{seg0_name}");
    let seg0_sizes = {
        let file = std::fs::File::create(&seg0_path).unwrap();
        let mut sw = SegmentWriter::new(BufWriter::new(file), 2);
        sw.write_column_block(
            0,
            &[
                FlexType::Integer(1),
                FlexType::Integer(2),
                FlexType::Integer(3),
            ],
            FlexTypeEnum::Integer,
        )
        .unwrap();
        sw.write_column_block(
            1,
            &[
                FlexType::String("a".into()),
                FlexType::String("b".into()),
                FlexType::String("c".into()),
            ],
            FlexTypeEnum::String,
        )
        .unwrap();
        sw.finish().unwrap()
    };

    // Write segment 1: rows [4, 5]
    let seg1_name = "seg.0001".to_string();
    let seg1_path = format!("{base_path}/{seg1_name}");
    let seg1_sizes = {
        let file = std::fs::File::create(&seg1_path).unwrap();
        let mut sw = SegmentWriter::new(BufWriter::new(file), 2);
        sw.write_column_block(
            0,
            &[FlexType::Integer(4), FlexType::Integer(5)],
            FlexTypeEnum::Integer,
        )
        .unwrap();
        sw.write_column_block(
            1,
            &[
                FlexType::String("d".into()),
                FlexType::String("e".into()),
            ],
            FlexTypeEnum::String,
        )
        .unwrap();
        sw.finish().unwrap()
    };

    // Assemble metadata
    let vfs = LocalFileSystem;
    assemble_sframe_from_segments(
        &vfs,
        base_path,
        &["id", "name"],
        &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        &[seg0_name, seg1_name],
        &[seg0_sizes, seg1_sizes],
        5,
    )
    .unwrap();

    // Read back and verify
    let mut reader = SFrameReader::open(base_path).unwrap();
    assert_eq!(reader.num_rows(), 5);
    assert_eq!(reader.column_names(), &["id", "name"]);

    // Verify segment 0 data
    let col0_seg0 = reader.segment_readers[0].read_column(0).unwrap();
    assert_eq!(
        col0_seg0,
        vec![
            FlexType::Integer(1),
            FlexType::Integer(2),
            FlexType::Integer(3),
        ]
    );
    let col1_seg0 = reader.segment_readers[0].read_column(1).unwrap();
    assert_eq!(
        col1_seg0,
        vec![
            FlexType::String("a".into()),
            FlexType::String("b".into()),
            FlexType::String("c".into()),
        ]
    );

    // Verify segment 1 data
    let col0_seg1 = reader.segment_readers[1].read_column(0).unwrap();
    assert_eq!(
        col0_seg1,
        vec![FlexType::Integer(4), FlexType::Integer(5)]
    );
    let col1_seg1 = reader.segment_readers[1].read_column(1).unwrap();
    assert_eq!(
        col1_seg1,
        vec![
            FlexType::String("d".into()),
            FlexType::String("e".into()),
        ]
    );
}
