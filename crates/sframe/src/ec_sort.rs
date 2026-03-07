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
/// Reads the forward map in chunks, then processes columns in parallel
/// within each chunk using plan slicing (`try_slice`). No materialization
/// needed — row-range reads are projected through the lazy plan.
fn scatter_columns(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    data_prefix: &str,
    input: &SFrame,
    forward_map: &SArray,
    num_buckets: usize,
    rows_per_bucket: u64,
    num_rows: u64,
    budget: usize,
) -> Result<sframe_storage::scatter_writer::ScatterResult> {
    use sframe_query::execute::materialize_sync;

    let input_dtypes = input.column_types();
    let num_input_cols = input_dtypes.len();
    let fmap_col_idx = num_input_cols;

    // Output columns: input columns + forward_map as the last column
    let mut scatter_dtypes = input_dtypes.clone();
    scatter_dtypes.push(FlexTypeEnum::Integer);

    let mut scatter_writer = ScatterWriter::new(
        vfs,
        base_path,
        data_prefix,
        &scatter_dtypes,
        num_buckets,
    )?;

    // Forward map chunk size: how many forward_map entries fit in memory.
    // Each entry is an i64 (8 bytes) but stored as FlexType (~24 bytes).
    let fmap_chunk_size = (budget / 24).max(1024) as u64;

    // Build a single-column SFrame for the forward_map (for slicing)
    let fmap_sf = SFrame::new_with_columns(
        vec![forward_map.clone()],
        vec!["__fmap__".to_string()],
    );

    let mut chunk_start = 0u64;
    while chunk_start < num_rows {
        let chunk_end = (chunk_start + fmap_chunk_size).min(num_rows);

        // Read forward_map values for this chunk
        let fmap_slice = fmap_sf.try_slice(chunk_start, chunk_end)?;
        let fmap_batch = materialize_sync(fmap_slice.compile_stream()?)?;
        let fmap_col = fmap_batch.column(0);
        let chunk_len = fmap_batch.num_rows();

        // Decode forward_map values into a shared Vec<u64>
        let fmap_values: Vec<u64> = (0..chunk_len)
            .map(|i| match fmap_col.get(i) {
                FlexType::Integer(v) => v as u64,
                _ => 0, // validated below
            })
            .collect();

        // Process columns in parallel: each thread reads its column's
        // data for this chunk via try_slice and classifies into buckets.
        let input_columns = input.columns();
        let per_col_results: Vec<Result<Vec<Vec<FlexType>>>> = (0..num_input_cols)
            .into_par_iter()
            .map(|col_idx| {
                // Slice this column to the chunk range
                let col_sa = input_columns[col_idx].try_slice(chunk_start, chunk_end)?;
                let col_sf = SFrame::new_with_columns(
                    vec![col_sa],
                    vec!["__c__".to_string()],
                );
                let col_batch = materialize_sync(col_sf.compile_stream()?)?;
                let col_data = col_batch.column(0);

                // Classify each value into its bucket
                let mut seg_bufs: Vec<Vec<FlexType>> = vec![Vec::new(); num_buckets];
                for r in 0..col_batch.num_rows() {
                    let bucket =
                        (fmap_values[r] / rows_per_bucket).min(num_buckets as u64 - 1) as usize;
                    seg_bufs[bucket].push(col_data.get(r));
                }
                Ok(seg_bufs)
            })
            .collect();

        // Write per-column per-segment buffers to ScatterWriter (sequential)
        for (col_idx, result) in per_col_results.into_iter().enumerate() {
            let seg_bufs = result?;
            for (seg_idx, values) in seg_bufs.into_iter().enumerate() {
                for value in values {
                    scatter_writer.write_to_segment(col_idx, seg_idx, value)?;
                }
            }
        }

        // Also scatter the forward_map values as the last column
        for r in 0..chunk_len {
            let fmap_val = fmap_values[r];
            let bucket = (fmap_val / rows_per_bucket).min(num_buckets as u64 - 1) as usize;
            scatter_writer.write_to_segment(
                fmap_col_idx,
                bucket,
                FlexType::Integer(fmap_val as i64),
            )?;
        }

        chunk_start = chunk_end;
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

    // Phase 1: Scatter (column-parallel via plan slicing)
    eprintln!("[sframe] ec_sort phase 1/2: scattering...");
    let scatter_result = scatter_columns(
        &*scatter_vfs,
        &scatter_path,
        &scatter_prefix,
        input,
        forward_map,
        num_buckets,
        rows_per_bucket,
        num_rows,
        budget,
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

/// Sort an SFrame using External Columnar Sort.
///
/// This algorithm is more memory-efficient than a full sort when the SFrame
/// has many or large value (non-key) columns. It works by:
///
/// 1. Sorting only the key columns (plus a row-number column) to produce the
///    inverse map and sorted key columns.
/// 2. Computing the forward map by permuting `[0..N)` with the inverse map.
/// 3. Permuting only the value columns using the forward map.
/// 4. Reassembling the final SFrame in original column order.
///
/// # Arguments
/// * `input` - The SFrame to sort.
/// * `key_column_indices` - Indices of columns to sort by.
/// * `sort_orders` - For each key column, `true` = ascending, `false` = descending.
pub(crate) fn ec_sort(
    input: &SFrame,
    key_column_indices: &[usize],
    sort_orders: &[bool],
) -> Result<SFrame> {
    use sframe_query::algorithms::sort::SortOrder;

    let num_rows = input.num_rows()?;
    if num_rows == 0 {
        return input.head(0);
    }

    let column_names = input.column_names();
    let num_columns = column_names.len();

    // Validate indices
    for &idx in key_column_indices {
        if idx >= num_columns {
            return Err(SFrameError::Format(format!(
                "Key column index {idx} out of range (SFrame has {num_columns} columns)"
            )));
        }
    }

    // -----------------------------------------------------------------------
    // Step 1: Sort key columns + row-number column to get inverse_map
    // -----------------------------------------------------------------------

    // Build a sub-SFrame with only the key columns + a row number column
    let row_number_name = "__ec_row_number__";
    let range_plan = PlannerNode::range(0, 1, num_rows);
    let row_number_sarray =
        SArray::from_plan(range_plan, FlexTypeEnum::Integer, Some(num_rows), 0);

    let mut key_columns: Vec<SArray> = Vec::with_capacity(key_column_indices.len() + 1);
    let mut key_names: Vec<String> = Vec::with_capacity(key_column_indices.len() + 1);
    for &idx in key_column_indices {
        key_columns.push(input.columns()[idx].clone());
        key_names.push(column_names[idx].clone());
    }
    key_columns.push(row_number_sarray);
    key_names.push(row_number_name.to_string());

    let keys_sf = SFrame::new_with_columns(key_columns, key_names.clone());

    // Build sort specification: sort by each key column in its specified order
    let sort_spec: Vec<(&str, SortOrder)> = key_column_indices
        .iter()
        .zip(sort_orders.iter())
        .map(|(&idx, &asc)| {
            let name = column_names[idx].as_str();
            let order = if asc {
                SortOrder::Ascending
            } else {
                SortOrder::Descending
            };
            (name, order)
        })
        .collect();

    eprintln!(
        "[sframe] ec_sort: {num_rows} rows, {} key cols, {} value cols",
        key_column_indices.len(),
        num_columns - key_column_indices.len(),
    );

    // Use standard_sort to bypass the decision layer (which would route
    // back to ec_sort, causing infinite recursion).
    let sorted_keys_sf = keys_sf.standard_sort(&sort_spec)?;

    // Extract the inverse_map (the row number column from the sorted result)
    let inverse_map = sorted_keys_sf.column(row_number_name)?.clone();

    // Extract sorted key columns (without the row number column)
    let sorted_key_names: Vec<&str> = key_names[..key_column_indices.len()]
        .iter()
        .map(|s| s.as_str())
        .collect();
    let sorted_keys = sorted_keys_sf.select(&sorted_key_names)?;

    // -----------------------------------------------------------------------
    // Step 2: Compute forward_map by permuting [0..N) with the inverse_map
    // -----------------------------------------------------------------------
    // forward_map[inverse_map[i]] = i, so permuting [0..N) by inverse_map
    // gives the forward_map.

    let range_plan2 = PlannerNode::range(0, 1, num_rows);
    let range_sarray =
        SArray::from_plan(range_plan2, FlexTypeEnum::Integer, Some(num_rows), 0);
    let range_sf = SFrame::new_with_columns(
        vec![range_sarray],
        vec!["__idx__".to_string()],
    );

    let permuted_range_sf = permute_sframe(&range_sf, &inverse_map)?;
    let forward_map = permuted_range_sf.columns()[0].clone();

    // -----------------------------------------------------------------------
    // Step 3: Permute value columns using the forward_map
    // -----------------------------------------------------------------------

    // Identify value column indices (everything not in key_column_indices)
    let key_set: std::collections::HashSet<usize> =
        key_column_indices.iter().copied().collect();
    let value_column_indices: Vec<usize> = (0..num_columns)
        .filter(|i| !key_set.contains(i))
        .collect();

    let sorted_value_columns: Option<SFrame> = if !value_column_indices.is_empty() {
        // Build a sub-SFrame of just the value columns
        let value_cols: Vec<SArray> = value_column_indices
            .iter()
            .map(|&idx| input.columns()[idx].clone())
            .collect();
        let value_names: Vec<String> = value_column_indices
            .iter()
            .map(|&idx| column_names[idx].clone())
            .collect();
        let value_sf = SFrame::new_with_columns(value_cols, value_names);

        Some(permute_sframe(&value_sf, &forward_map)?)
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 4: Reassemble in original column order
    // -----------------------------------------------------------------------

    let mut final_columns: Vec<SArray> = Vec::with_capacity(num_columns);
    let mut final_names: Vec<String> = Vec::with_capacity(num_columns);

    for col_idx in 0..num_columns {
        final_names.push(column_names[col_idx].clone());
        if key_set.contains(&col_idx) {
            // Find which position in key_column_indices this column is
            let key_pos = key_column_indices
                .iter()
                .position(|&k| k == col_idx)
                .unwrap();
            final_columns.push(sorted_keys.columns()[key_pos].clone());
        } else {
            // Find which position in value_column_indices this column is
            let val_pos = value_column_indices
                .iter()
                .position(|&v| v == col_idx)
                .unwrap();
            final_columns.push(
                sorted_value_columns
                    .as_ref()
                    .unwrap()
                    .columns()[val_pos]
                    .clone(),
            );
        }
    }

    Ok(SFrame::new_with_columns(final_columns, final_names))
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

    // -----------------------------------------------------------------------
    // ec_sort integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ec_sort_basic() {
        let ids = SArray::from_vec(
            vec![FlexType::Integer(3), FlexType::Integer(1), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let names = SArray::from_vec(
            vec![
                FlexType::String(Arc::from("three")),
                FlexType::String(Arc::from("one")),
                FlexType::String(Arc::from("two")),
            ],
            FlexTypeEnum::String,
        ).unwrap();

        let sf = SFrame::from_columns(vec![("id", ids), ("name", names)]).unwrap();
        let result = ec_sort(&sf, &[0], &[true]).unwrap();

        assert_eq!(result.num_rows().unwrap(), 3);
        let rows = result.iter_rows().unwrap();
        assert_eq!(rows[0], vec![FlexType::Integer(1), FlexType::String(Arc::from("one"))]);
        assert_eq!(rows[1], vec![FlexType::Integer(2), FlexType::String(Arc::from("two"))]);
        assert_eq!(rows[2], vec![FlexType::Integer(3), FlexType::String(Arc::from("three"))]);

        // Column names should be preserved in order
        assert_eq!(result.column_names(), &["id", "name"]);
    }

    #[test]
    fn test_ec_sort_descending() {
        let ids = SArray::from_vec(
            vec![FlexType::Integer(3), FlexType::Integer(1), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let names = SArray::from_vec(
            vec![
                FlexType::String(Arc::from("three")),
                FlexType::String(Arc::from("one")),
                FlexType::String(Arc::from("two")),
            ],
            FlexTypeEnum::String,
        ).unwrap();

        let sf = SFrame::from_columns(vec![("id", ids), ("name", names)]).unwrap();
        let result = ec_sort(&sf, &[0], &[false]).unwrap();

        assert_eq!(result.num_rows().unwrap(), 3);
        let rows = result.iter_rows().unwrap();
        assert_eq!(rows[0], vec![FlexType::Integer(3), FlexType::String(Arc::from("three"))]);
        assert_eq!(rows[1], vec![FlexType::Integer(2), FlexType::String(Arc::from("two"))]);
        assert_eq!(rows[2], vec![FlexType::Integer(1), FlexType::String(Arc::from("one"))]);
    }

    #[test]
    fn test_ec_sort_matches_standard_sort() {
        use sframe_query::algorithms::sort::SortOrder;

        let n = 1000;
        let int_vals: Vec<FlexType> = (0..n)
            .rev()
            .map(|i| FlexType::Integer(i as i64))
            .collect();
        let str_vals: Vec<FlexType> = (0..n)
            .rev()
            .map(|i| FlexType::String(Arc::from(format!("str_{:04}", i))))
            .collect();
        let float_vals: Vec<FlexType> = (0..n)
            .rev()
            .map(|i| FlexType::Float(i as f64 * 1.5))
            .collect();

        let col_i = SArray::from_vec(int_vals, FlexTypeEnum::Integer).unwrap();
        let col_s = SArray::from_vec(str_vals, FlexTypeEnum::String).unwrap();
        let col_f = SArray::from_vec(float_vals, FlexTypeEnum::Float).unwrap();

        let sf = SFrame::from_columns(vec![
            ("ints", col_i),
            ("strs", col_s),
            ("floats", col_f),
        ]).unwrap();

        let ec_result = ec_sort(&sf, &[0], &[true]).unwrap();
        let std_result = sf.sort(&[("ints", SortOrder::Ascending)]).unwrap();

        assert_eq!(ec_result.num_rows().unwrap(), std_result.num_rows().unwrap());

        let ec_rows = ec_result.iter_rows().unwrap();
        let std_rows = std_result.iter_rows().unwrap();

        for i in 0..n {
            assert_eq!(
                ec_rows[i], std_rows[i],
                "Row {i} mismatch: ec_sort={:?}, sort={:?}",
                ec_rows[i], std_rows[i]
            );
        }
    }

    #[test]
    fn test_ec_sort_single_row() {
        let ids = SArray::from_vec(
            vec![FlexType::Integer(42)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let vals = SArray::from_vec(
            vec![FlexType::String(Arc::from("hello"))],
            FlexTypeEnum::String,
        ).unwrap();

        let sf = SFrame::from_columns(vec![("id", ids), ("val", vals)]).unwrap();
        let result = ec_sort(&sf, &[0], &[true]).unwrap();

        assert_eq!(result.num_rows().unwrap(), 1);
        let rows = result.iter_rows().unwrap();
        assert_eq!(rows[0], vec![FlexType::Integer(42), FlexType::String(Arc::from("hello"))]);
    }

    #[test]
    fn test_ec_sort_already_sorted() {
        let n = 100;
        let int_vals: Vec<FlexType> = (0..n).map(|i| FlexType::Integer(i as i64)).collect();
        let str_vals: Vec<FlexType> = (0..n)
            .map(|i| FlexType::String(Arc::from(format!("row_{:03}", i))))
            .collect();

        let col_i = SArray::from_vec(int_vals.clone(), FlexTypeEnum::Integer).unwrap();
        let col_s = SArray::from_vec(str_vals.clone(), FlexTypeEnum::String).unwrap();

        let sf = SFrame::from_columns(vec![("id", col_i), ("label", col_s)]).unwrap();
        let result = ec_sort(&sf, &[0], &[true]).unwrap();

        assert_eq!(result.num_rows().unwrap(), n as u64);
        let rows = result.iter_rows().unwrap();
        for i in 0..n {
            assert_eq!(rows[i][0], int_vals[i]);
            assert_eq!(rows[i][1], str_vals[i]);
        }
    }

    #[test]
    fn test_ec_sort_all_identical_keys() {
        let n = 10;
        let int_vals: Vec<FlexType> = (0..n).map(|_| FlexType::Integer(7)).collect();
        let str_vals: Vec<FlexType> = (0..n)
            .map(|i| FlexType::String(Arc::from(format!("item_{}", i))))
            .collect();

        let col_i = SArray::from_vec(int_vals, FlexTypeEnum::Integer).unwrap();
        let col_s = SArray::from_vec(str_vals, FlexTypeEnum::String).unwrap();

        let sf = SFrame::from_columns(vec![("key", col_i), ("val", col_s)]).unwrap();
        let result = ec_sort(&sf, &[0], &[true]).unwrap();

        // Should not crash; all 10 rows present
        assert_eq!(result.num_rows().unwrap(), 10);

        // All keys should be 7
        let rows = result.iter_rows().unwrap();
        for row in &rows {
            assert_eq!(row[0], FlexType::Integer(7));
        }

        // All value strings should be present (order among ties is unspecified)
        let mut vals: Vec<String> = rows.iter().map(|r| {
            match &r[1] {
                FlexType::String(s) => s.to_string(),
                _ => panic!("expected string"),
            }
        }).collect();
        vals.sort();
        let mut expected: Vec<String> = (0..n).map(|i| format!("item_{}", i)).collect();
        expected.sort();
        assert_eq!(vals, expected);
    }

    #[test]
    fn test_ec_sort_multi_key() {
        // 4 rows with two key columns: sort by (group ASC, priority DESC)
        let groups = SArray::from_vec(
            vec![
                FlexType::Integer(2),
                FlexType::Integer(1),
                FlexType::Integer(1),
                FlexType::Integer(2),
            ],
            FlexTypeEnum::Integer,
        ).unwrap();
        let priorities = SArray::from_vec(
            vec![
                FlexType::Integer(10),
                FlexType::Integer(20),
                FlexType::Integer(30),
                FlexType::Integer(40),
            ],
            FlexTypeEnum::Integer,
        ).unwrap();
        let labels = SArray::from_vec(
            vec![
                FlexType::String(Arc::from("g2p10")),
                FlexType::String(Arc::from("g1p20")),
                FlexType::String(Arc::from("g1p30")),
                FlexType::String(Arc::from("g2p40")),
            ],
            FlexTypeEnum::String,
        ).unwrap();

        let sf = SFrame::from_columns(vec![
            ("group", groups),
            ("priority", priorities),
            ("label", labels),
        ]).unwrap();

        // Sort by group ASC, priority DESC
        let result = ec_sort(&sf, &[0, 1], &[true, false]).unwrap();

        assert_eq!(result.num_rows().unwrap(), 4);
        let rows = result.iter_rows().unwrap();

        // Expected order: group=1,pri=30 -> group=1,pri=20 -> group=2,pri=40 -> group=2,pri=10
        assert_eq!(rows[0][0], FlexType::Integer(1));
        assert_eq!(rows[0][1], FlexType::Integer(30));
        assert_eq!(rows[0][2], FlexType::String(Arc::from("g1p30")));

        assert_eq!(rows[1][0], FlexType::Integer(1));
        assert_eq!(rows[1][1], FlexType::Integer(20));
        assert_eq!(rows[1][2], FlexType::String(Arc::from("g1p20")));

        assert_eq!(rows[2][0], FlexType::Integer(2));
        assert_eq!(rows[2][1], FlexType::Integer(40));
        assert_eq!(rows[2][2], FlexType::String(Arc::from("g2p40")));

        assert_eq!(rows[3][0], FlexType::Integer(2));
        assert_eq!(rows[3][1], FlexType::Integer(10));
        assert_eq!(rows[3][2], FlexType::String(Arc::from("g2p10")));
    }

    #[test]
    fn test_ec_sort_many_columns() {
        let n = 200;
        // 1 key column + 20 value string columns
        let key_vals: Vec<FlexType> = (0..n)
            .rev()
            .map(|i| FlexType::Integer(i as i64))
            .collect();
        let key_col = SArray::from_vec(key_vals, FlexTypeEnum::Integer).unwrap();

        let col_names: Vec<String> = std::iter::once("key".to_string())
            .chain((0..20).map(|i| format!("val_{}", i)))
            .collect();

        let mut all_columns: Vec<SArray> = vec![key_col];
        for i in 0..20 {
            let str_vals: Vec<FlexType> = (0..n)
                .rev()
                .map(|j| FlexType::String(Arc::from(format!("c{}_r{}", i, j))))
                .collect();
            all_columns.push(SArray::from_vec(str_vals, FlexTypeEnum::String).unwrap());
        }

        let sf = SFrame::new_with_columns(all_columns, col_names);
        let result = ec_sort(&sf, &[0], &[true]).unwrap();

        assert_eq!(result.num_rows().unwrap(), n as u64);
        let rows = result.iter_rows().unwrap();

        // First row should have key=0 (originally last row)
        assert_eq!(rows[0][0], FlexType::Integer(0));
        // Its value columns should reference row 0
        for i in 0..20 {
            assert_eq!(rows[0][1 + i], FlexType::String(Arc::from(format!("c{}_r0", i))));
        }

        // Last row should have key=199 (originally first row)
        assert_eq!(rows[n - 1][0], FlexType::Integer((n - 1) as i64));
        for i in 0..20 {
            assert_eq!(
                rows[n - 1][1 + i],
                FlexType::String(Arc::from(format!("c{}_r{}", i, n - 1)))
            );
        }
    }

    #[test]
    fn test_ec_sort_via_sframe_api() {
        // Test the public SFrame::ec_sort method
        let ids = SArray::from_vec(
            vec![FlexType::Integer(3), FlexType::Integer(1), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let names = SArray::from_vec(
            vec![
                FlexType::String(Arc::from("three")),
                FlexType::String(Arc::from("one")),
                FlexType::String(Arc::from("two")),
            ],
            FlexTypeEnum::String,
        ).unwrap();

        let sf = SFrame::from_columns(vec![("id", ids), ("name", names)]).unwrap();
        let result = sf.ec_sort(&[("id", true)]).unwrap();

        assert_eq!(result.num_rows().unwrap(), 3);
        let rows = result.iter_rows().unwrap();
        assert_eq!(rows[0], vec![FlexType::Integer(1), FlexType::String(Arc::from("one"))]);
        assert_eq!(rows[1], vec![FlexType::Integer(2), FlexType::String(Arc::from("two"))]);
        assert_eq!(rows[2], vec![FlexType::Integer(3), FlexType::String(Arc::from("three"))]);
    }

    #[test]
    fn test_ec_sort_all_key_columns() {
        // Edge case: all columns are key columns (no value columns to permute)
        let ids = SArray::from_vec(
            vec![FlexType::Integer(3), FlexType::Integer(1), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        ).unwrap();

        let sf = SFrame::from_columns(vec![("id", ids)]).unwrap();
        let result = ec_sort(&sf, &[0], &[true]).unwrap();

        assert_eq!(result.num_rows().unwrap(), 3);
        let rows = result.iter_rows().unwrap();
        assert_eq!(rows[0], vec![FlexType::Integer(1)]);
        assert_eq!(rows[1], vec![FlexType::Integer(2)]);
        assert_eq!(rows[2], vec![FlexType::Integer(3)]);
    }
}
