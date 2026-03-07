//! EC Sort (External Column Sort) — permutes an SFrame by a forward map.
//!
//! Given an SFrame and a forward map (an SArray of integers where
//! `forward_map[i]` is the output row index for input row `i`), produces
//! a new SFrame with rows rearranged according to the map.
//!
//! The algorithm has two phases:
//! 1. **Scatter**: Stream each column alongside the forward map, writing
//!    values into buckets (segment files) based on `forward_map[i] / rows_per_bucket`.
//! 2. **Permute**: For each bucket, read the scattered values and the
//!    forward map, then permute values into their final positions within the
//!    bucket. Assemble the buckets into an output SFrame.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_query::planner::PlannerNode;
use sframe_storage::scatter_writer::ScatterWriter;
use sframe_storage::segment_reader::SegmentReader;
use sframe_storage::segment_writer::BufferedSegmentWriter;
use sframe_storage::sframe_writer::{assemble_sframe_from_segments, generate_hash, segment_filename};
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::sarray::SArray;
use crate::sframe::{AnonymousStore, SFrame};

const CHUNK_SIZE: usize = 8192;

// ---------------------------------------------------------------------------
// Column size estimation
// ---------------------------------------------------------------------------

/// Rough estimate of bytes per value for a given column type.
///
/// Mirrors `sframe_storage::segment_writer::estimate_bytes_per_value`
/// (which is `pub(crate)` in that crate and not accessible here).
fn estimate_bytes_per_value(dtype: FlexTypeEnum) -> usize {
    match dtype {
        FlexTypeEnum::Integer => 8,
        FlexTypeEnum::Float => 8,
        FlexTypeEnum::String => 32,
        FlexTypeEnum::Vector => 64,
        FlexTypeEnum::List => 64,
        FlexTypeEnum::Dict => 64,
        FlexTypeEnum::DateTime => 12,
        FlexTypeEnum::Undefined => 1,
    }
}

/// Estimate bytes per value for each column type.
fn estimate_column_bytes(column_types: &[FlexTypeEnum]) -> Vec<usize> {
    column_types
        .iter()
        .map(|&dt| estimate_bytes_per_value(dt))
        .collect()
}

/// Compute the number of scatter buckets.
///
/// Formula: find the column with the largest total estimated size
/// (`bytes_per_value * num_rows`), divide by half the per-thread budget,
/// and multiply by the number of CPUs. Clamp to 1 if there would be
/// more buckets than rows.
fn compute_num_buckets(num_rows: u64, column_bytes: &[usize], budget_per_thread: usize) -> usize {
    if num_rows == 0 || column_bytes.is_empty() {
        return 1;
    }

    let max_col_bytes = column_bytes
        .iter()
        .map(|&b| b as u64 * num_rows)
        .max()
        .unwrap_or(0);

    let half_budget = (budget_per_thread / 2).max(1) as u64;
    let buckets_per_cpu = ((max_col_bytes + half_budget - 1) / half_budget).max(1) as usize;
    let num_cpus = rayon::current_num_threads().max(1);
    let num_buckets = buckets_per_cpu * num_cpus;

    // Don't create more buckets than rows
    if num_buckets as u64 > num_rows {
        (num_rows as usize).max(1)
    } else {
        num_buckets
    }
}

// ---------------------------------------------------------------------------
// Scatter phase
// ---------------------------------------------------------------------------

/// Scatter all input columns (plus the forward map) into segment files.
///
/// Creates `num_buckets` segment files. Each segment stores all input columns
/// plus one extra column (the forward map). Values are distributed to segments
/// based on `bucket_id = (forward_map_value / rows_per_bucket).min(num_buckets - 1)`.
fn scatter_columns(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    data_prefix: &str,
    input: &SFrame,
    forward_map: &SArray,
    num_buckets: usize,
    rows_per_bucket: u64,
) -> Result<sframe_storage::scatter_writer::ScatterResult> {
    let input_dtypes = input.column_types();
    let num_input_cols = input_dtypes.len();

    // Scatter columns: input columns + forward_map as the last column
    let mut scatter_dtypes = input_dtypes.clone();
    scatter_dtypes.push(FlexTypeEnum::Integer);

    let mut scatter_writer = ScatterWriter::new(
        vfs,
        base_path,
        data_prefix,
        &scatter_dtypes,
        num_buckets,
    )?;

    // Process each input column: stream [forward_map, column] together
    for col_idx in 0..num_input_cols {
        let col_sarray = input.columns()[col_idx].clone();
        let fmap_sarray = forward_map.clone();

        // Build a 2-column SFrame [forward_map, value_column]
        let pair_sf = SFrame::new_with_columns(
            vec![fmap_sarray, col_sarray],
            vec!["__fmap__".to_string(), "__val__".to_string()],
        );

        let mut stream = pair_sf.compile_stream()?;
        while let Some(batch_result) = stream.next_batch() {
            let batch = batch_result?;
            let fmap_col = batch.column(0);
            let val_col = batch.column(1);

            for r in 0..batch.num_rows() {
                let fmap_val = match fmap_col.get(r) {
                    FlexType::Integer(v) => v as u64,
                    _ => {
                        return Err(SFrameError::Type(
                            "forward_map must contain Integer values".into(),
                        ))
                    }
                };

                let bucket = (fmap_val / rows_per_bucket).min(num_buckets as u64 - 1) as usize;
                scatter_writer.write_to_segment(col_idx, bucket, val_col.get(r))?;
            }
        }
        scatter_writer.flush_column(col_idx)?;
    }

    // Now scatter the forward_map itself as the last column
    {
        let fmap_col_idx = num_input_cols;
        let fmap_sarray = forward_map.clone();
        let fmap_sf = SFrame::new_with_columns(
            vec![fmap_sarray],
            vec!["__fmap__".to_string()],
        );

        let mut stream = fmap_sf.compile_stream()?;
        while let Some(batch_result) = stream.next_batch() {
            let batch = batch_result?;
            let col = batch.column(0);

            for r in 0..batch.num_rows() {
                let fmap_val = match col.get(r) {
                    FlexType::Integer(v) => v as u64,
                    _ => {
                        return Err(SFrameError::Type(
                            "forward_map must contain Integer values".into(),
                        ))
                    }
                };

                let bucket = (fmap_val / rows_per_bucket).min(num_buckets as u64 - 1) as usize;
                scatter_writer.write_to_segment(
                    fmap_col_idx,
                    bucket,
                    FlexType::Integer(fmap_val as i64),
                )?;
            }
        }
        scatter_writer.flush_column(fmap_col_idx)?;
    }

    scatter_writer.finish()
}

// ---------------------------------------------------------------------------
// Permute phase
// ---------------------------------------------------------------------------

/// Permute all buckets in parallel and assemble the output SFrame.
///
/// For each bucket (scatter segment), reads the forward map column to build
/// the local permutation, then reads value columns in groups that fit in
/// the memory budget, permutes them, and writes to output segments.
fn permute_buckets(
    scatter_vfs: &Arc<dyn VirtualFileSystem>,
    scatter_base_path: &str,
    scatter_result: &sframe_storage::scatter_writer::ScatterResult,
    output_vfs: &Arc<dyn VirtualFileSystem>,
    output_base_path: &str,
    column_names: &[String],
    column_types: &[FlexTypeEnum],
    _num_rows: u64,
    rows_per_bucket: u64,
    column_bytes: &[usize],
    budget_per_thread: usize,
) -> Result<SFrame> {
    let num_buckets = scatter_result.segment_files.len();
    let num_input_cols = column_types.len();
    // The scatter segments have num_input_cols + 1 columns (last is forward_map)
    let mut scatter_col_types = column_types.to_vec();
    scatter_col_types.push(FlexTypeEnum::Integer);

    let data_prefix = format!("m_{}", generate_hash(output_base_path));

    // Process each bucket in parallel
    let results: Vec<Option<Result<(String, Vec<u64>, u64)>>> = (0..num_buckets)
        .into_par_iter()
        .map(|bucket_idx| {
            // Check if this bucket has any rows
            let bucket_rows = scatter_result.all_segment_sizes[bucket_idx]
                .first()
                .copied()
                .unwrap_or(0);
            if bucket_rows == 0 {
                return None;
            }

            Some((|| -> Result<(String, Vec<u64>, u64)> {
                permute_one_bucket(
                    scatter_vfs,
                    scatter_base_path,
                    &scatter_result.segment_files[bucket_idx],
                    &scatter_col_types,
                    output_vfs,
                    output_base_path,
                    &data_prefix,
                    bucket_idx,
                    num_input_cols,
                    column_types,
                    rows_per_bucket,
                    bucket_rows,
                    column_bytes,
                    budget_per_thread,
                )
            })())
        })
        .collect();

    // Collect results, skipping empty buckets
    let mut segment_files = Vec::new();
    let mut all_segment_sizes = Vec::new();
    let mut total_rows = 0u64;

    for result in results.into_iter().flatten() {
        let (seg_file, sizes, rows) = result?;
        if rows > 0 {
            segment_files.push(seg_file);
            all_segment_sizes.push(sizes);
            total_rows += rows;
        }
    }

    if total_rows == 0 {
        // Return empty SFrame
        let empty_columns: Vec<SArray> = column_types
            .iter()
            .enumerate()
            .map(|(_, &dt)| {
                SArray::from_vec(Vec::new(), dt).unwrap()
            })
            .collect();
        return Ok(SFrame::new_with_columns(
            empty_columns,
            column_names.to_vec(),
        ));
    }

    // Assemble SFrame metadata from segments
    let col_name_refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
    assemble_sframe_from_segments(
        &**output_vfs,
        output_base_path,
        &col_name_refs,
        column_types,
        &segment_files,
        &all_segment_sizes,
        total_rows,
        &HashMap::new(),
    )?;

    // Construct SFrame backed by the output cache directory
    let cache_fs = global_cache_fs();
    let store: Arc<dyn Send + Sync> = Arc::new(AnonymousStore {
        path: output_base_path.to_string(),
        cache_fs: cache_fs.clone(),
    });
    let plan = PlannerNode::sframe_source_cached(
        output_base_path,
        column_names.to_vec(),
        column_types.to_vec(),
        total_rows,
        store,
    );

    let columns: Vec<SArray> = column_types
        .iter()
        .enumerate()
        .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(total_rows), i))
        .collect();

    Ok(SFrame::new_with_columns(columns, column_names.to_vec()))
}

/// Permute a single bucket: read scatter segment, apply permutation, write output segment.
fn permute_one_bucket(
    scatter_vfs: &Arc<dyn VirtualFileSystem>,
    scatter_base_path: &str,
    scatter_seg_file: &str,
    scatter_col_types: &[FlexTypeEnum],
    output_vfs: &Arc<dyn VirtualFileSystem>,
    output_base_path: &str,
    output_data_prefix: &str,
    bucket_idx: usize,
    num_input_cols: usize,
    output_col_types: &[FlexTypeEnum],
    rows_per_bucket: u64,
    _bucket_rows: u64,
    column_bytes: &[usize],
    budget_per_thread: usize,
) -> Result<(String, Vec<u64>, u64)> {
    let seg_path = format!("{scatter_base_path}/{scatter_seg_file}");
    let bucket_start = bucket_idx as u64 * rows_per_bucket;

    // Open scatter segment and read the forward map column (last column)
    let file = scatter_vfs.open_read(&seg_path)?;
    let file_size = file.size()?;
    let mut seg_reader =
        SegmentReader::open(Box::new(file), file_size, scatter_col_types.to_vec())?;

    let fmap_values = seg_reader.read_column(num_input_cols)?;
    let num_rows_in_bucket = fmap_values.len();

    if num_rows_in_bucket == 0 {
        let seg_file = segment_filename(output_data_prefix, bucket_idx);
        let seg_out_path = format!("{output_base_path}/{seg_file}");
        let out_file = output_vfs.open_write(&seg_out_path)?;
        let seg_writer = BufferedSegmentWriter::new(out_file, output_col_types);
        let sizes = seg_writer.finish()?;
        return Ok((seg_file, sizes, 0));
    }

    // Build permutation: permutation[i] = local_target for scatter row i
    // fmap_values[i] is the global target row index.
    // local_target = fmap_values[i] - bucket_start
    let mut permutation = vec![0usize; num_rows_in_bucket];
    for (i, fmap_val) in fmap_values.iter().enumerate() {
        let global_target = match fmap_val {
            FlexType::Integer(v) => *v as u64,
            _ => {
                return Err(SFrameError::Type(
                    "forward_map must contain Integer values".into(),
                ))
            }
        };
        let local_target = (global_target - bucket_start) as usize;
        permutation[i] = local_target;
    }

    // Create output segment writer
    let seg_file = segment_filename(output_data_prefix, bucket_idx);
    let seg_out_path = format!("{output_base_path}/{seg_file}");
    let out_file = output_vfs.open_write(&seg_out_path)?;
    let mut seg_writer = BufferedSegmentWriter::new(out_file, output_col_types);

    // Process columns in groups that fit in memory budget
    let mut col_start = 0;
    while col_start < num_input_cols {
        // Determine how many columns fit in budget
        let mut col_end = col_start + 1; // At least one column per group
        let mut memory_estimate =
            column_bytes[col_start] * num_rows_in_bucket;

        while col_end < num_input_cols {
            let next_mem = column_bytes[col_end] * num_rows_in_bucket;
            if memory_estimate + next_mem > budget_per_thread {
                break;
            }
            memory_estimate += next_mem;
            col_end += 1;
        }

        // Re-open the scatter segment for reading value columns
        let file = scatter_vfs.open_read(&seg_path)?;
        let file_size = file.size()?;
        let mut col_reader =
            SegmentReader::open(Box::new(file), file_size, scatter_col_types.to_vec())?;

        // Read and permute each column in this group
        for col_idx in col_start..col_end {
            let values = col_reader.read_column(col_idx)?;

            // Apply permutation
            let mut permuted = vec![FlexType::Undefined; num_rows_in_bucket];
            for (src_idx, &dst_idx) in permutation.iter().enumerate() {
                permuted[dst_idx] = values[src_idx].clone();
            }

            // Write in chunks
            for chunk in permuted.chunks(CHUNK_SIZE) {
                seg_writer.write_column_block(col_idx, chunk, output_col_types[col_idx])?;
            }
        }

        col_start = col_end;
    }

    let sizes = seg_writer.finish()?;
    Ok((seg_file, sizes, num_rows_in_bucket as u64))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Permute an SFrame according to a forward map.
///
/// `forward_map` must be an SArray of `Integer` type with length equal to
/// `input.num_rows()`. Each value `forward_map[i]` specifies the output row
/// index for input row `i`. The forward map must be a permutation of `0..N`.
///
/// Returns a new SFrame with the same columns but rows rearranged according
/// to the forward map.
pub(crate) fn permute_sframe(input: &SFrame, forward_map: &SArray) -> Result<SFrame> {
    let num_rows = input.num_rows()?;
    if num_rows == 0 {
        return input.head(0);
    }

    let column_types = input.column_types();
    let column_names = input.column_names().to_vec();
    let column_bytes = estimate_column_bytes(&column_types);

    let budget = sframe_config::global().sort_max_memory / rayon::current_num_threads().max(1);
    let num_buckets = compute_num_buckets(num_rows, &column_bytes, budget);
    let rows_per_bucket = ((num_rows + num_buckets as u64 - 1) / num_buckets as u64).max(1);

    eprintln!(
        "[sframe] ec_sort permute: {num_rows} rows, {num_buckets} buckets, \
         {rows_per_bucket} rows/bucket, budget {budget}"
    );

    let cache_fs = global_cache_fs();

    // Allocate scatter scratch directory
    let scatter_path = cache_fs.alloc_dir();
    let scatter_vfs: Arc<dyn VirtualFileSystem> = Arc::new(ArcCacheFsVfs(cache_fs.clone()));
    scatter_vfs.mkdir_p(&scatter_path)?;
    let scatter_prefix = format!("s_{}", generate_hash(&scatter_path));

    // Phase 1: Scatter
    eprintln!("[sframe] ec_sort phase 1/2: scattering...");
    let scatter_result = scatter_columns(
        &*scatter_vfs,
        &scatter_path,
        &scatter_prefix,
        input,
        forward_map,
        num_buckets,
        rows_per_bucket,
    )?;

    // Allocate output directory
    let output_path = cache_fs.alloc_dir();
    let output_vfs: Arc<dyn VirtualFileSystem> = Arc::new(ArcCacheFsVfs(cache_fs.clone()));
    output_vfs.mkdir_p(&output_path)?;

    // Phase 2: Permute
    eprintln!("[sframe] ec_sort phase 2/2: permuting...");
    let result = permute_buckets(
        &scatter_vfs,
        &scatter_path,
        &scatter_result,
        &output_vfs,
        &output_path,
        &column_names,
        &column_types,
        num_rows,
        rows_per_bucket,
        &column_bytes,
        budget,
    )?;

    // Clean up scatter scratch
    cache_fs.remove_dir(&scatter_path).ok();

    Ok(result)
}

/// Sort an SFrame using EC Sort.
///
/// This is a placeholder for the full EC Sort algorithm that first generates
/// a forward map from the sort keys, then calls `permute_sframe`.
#[allow(dead_code)]
pub(crate) fn ec_sort(_sf: &SFrame, _sort_keys: &[sframe_query::algorithms::sort::SortKey]) -> Result<SFrame> {
    todo!("ec_sort: generate forward map from sort keys, then call permute_sframe")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sframe_types::flex_type::FlexTypeEnum;

    #[test]
    fn test_estimate_column_bytes() {
        let types = vec![FlexTypeEnum::Integer, FlexTypeEnum::String, FlexTypeEnum::Float];
        let bytes = estimate_column_bytes(&types);
        assert_eq!(bytes, vec![8, 32, 8]);
    }

    #[test]
    fn test_compute_num_buckets_zero_rows() {
        assert_eq!(compute_num_buckets(0, &[8, 32], 1024), 1);
    }

    #[test]
    fn test_compute_num_buckets_empty_cols() {
        assert_eq!(compute_num_buckets(100, &[], 1024), 1);
    }

    #[test]
    fn test_compute_num_buckets_clamp_to_rows() {
        // With very small budget and few rows, buckets should not exceed rows
        let result = compute_num_buckets(3, &[8], 1);
        assert!(result <= 3, "buckets ({result}) should not exceed rows (3)");
        assert!(result >= 1);
    }

    #[test]
    fn test_permute_sframe_reverse() {
        // Create a 6-row, 2-column SFrame and reverse it
        let col_a = SArray::from_vec(
            vec![
                FlexType::Integer(0),
                FlexType::Integer(1),
                FlexType::Integer(2),
                FlexType::Integer(3),
                FlexType::Integer(4),
                FlexType::Integer(5),
            ],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let col_b = SArray::from_vec(
            vec![
                FlexType::String(Arc::from("a")),
                FlexType::String(Arc::from("b")),
                FlexType::String(Arc::from("c")),
                FlexType::String(Arc::from("d")),
                FlexType::String(Arc::from("e")),
                FlexType::String(Arc::from("f")),
            ],
            FlexTypeEnum::String,
        )
        .unwrap();

        let sf = SFrame::from_columns(vec![("x", col_a), ("y", col_b)]).unwrap();

        // Reverse permutation: [5, 4, 3, 2, 1, 0]
        let fmap = SArray::from_vec(
            vec![
                FlexType::Integer(5),
                FlexType::Integer(4),
                FlexType::Integer(3),
                FlexType::Integer(2),
                FlexType::Integer(1),
                FlexType::Integer(0),
            ],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let result = permute_sframe(&sf, &fmap).unwrap();

        assert_eq!(result.num_rows().unwrap(), 6);
        let rows = result.iter_rows().unwrap();

        // Row 0 of output should be original row 5 (since fmap[5]=0)
        // Row 5 of output should be original row 0 (since fmap[0]=5)
        assert_eq!(rows[0], vec![FlexType::Integer(5), FlexType::String(Arc::from("f"))]);
        assert_eq!(rows[1], vec![FlexType::Integer(4), FlexType::String(Arc::from("e"))]);
        assert_eq!(rows[2], vec![FlexType::Integer(3), FlexType::String(Arc::from("d"))]);
        assert_eq!(rows[3], vec![FlexType::Integer(2), FlexType::String(Arc::from("c"))]);
        assert_eq!(rows[4], vec![FlexType::Integer(1), FlexType::String(Arc::from("b"))]);
        assert_eq!(rows[5], vec![FlexType::Integer(0), FlexType::String(Arc::from("a"))]);
    }

    #[test]
    fn test_permute_sframe_identity() {
        // Create a 3-row SFrame and apply identity permutation
        let col_a = SArray::from_vec(
            vec![
                FlexType::Integer(10),
                FlexType::Integer(20),
                FlexType::Integer(30),
            ],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let sf = SFrame::from_columns(vec![("v", col_a)]).unwrap();

        // Identity permutation: [0, 1, 2]
        let fmap = SArray::from_vec(
            vec![
                FlexType::Integer(0),
                FlexType::Integer(1),
                FlexType::Integer(2),
            ],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let result = permute_sframe(&sf, &fmap).unwrap();

        assert_eq!(result.num_rows().unwrap(), 3);
        let rows = result.iter_rows().unwrap();

        assert_eq!(rows[0], vec![FlexType::Integer(10)]);
        assert_eq!(rows[1], vec![FlexType::Integer(20)]);
        assert_eq!(rows[2], vec![FlexType::Integer(30)]);
    }
}
