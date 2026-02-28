//! SFrame — user-facing dataframe API with lazy evaluation.
//!
//! Operations build PlannerNode DAGs. Materialization happens on
//! `.head()`, `.iter_rows()`, `.save()`, `.materialize()`, or `Display`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use sframe_io::cache_fs::{global_cache_fs, CacheFs};
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_query::algorithms::aggregators::AggSpec;
use sframe_query::algorithms::csv_parser::{self, CsvOptions};
use sframe_query::algorithms::csv_writer::{self, CsvWriterOptions};
use sframe_query::algorithms::json as json_io;
use sframe_query::algorithms::groupby;
use sframe_query::algorithms::join::{self, JoinOn, JoinType};
use sframe_query::algorithms::sort::{self, SortKey, SortOrder};
use sframe_query::batch::SFrameRows;
use sframe_query::execute::{
    compile, for_each_batch_sync, materialize_head_sync, materialize_sync,
};
use sframe_query::planner::PlannerNode;
use sframe_storage::sframe_reader::SFrameReader;
use sframe_storage::sframe_writer::SFrameWriter;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::sarray::SArray;

const DEFAULT_CHUNK_SIZE: usize = 8192;

/// RAII guard for anonymous cache-backed SFrames.
/// When dropped, removes the cache directory and all its files.
struct AnonymousStore {
    path: String,
    cache_fs: Arc<CacheFs>,
}

impl Drop for AnonymousStore {
    fn drop(&mut self) {
        self.cache_fs.remove_dir(&self.path).ok();
    }
}

/// Incremental SFrame builder backed by an arbitrary VFS.
///
/// Writes data in chunks to avoid holding the full column-vector copy in
/// memory at once. Call `finish()` to finalize the SFrame; if the builder
/// is dropped without `finish()`, the directory is cleaned up (for
/// anonymous builders only).
pub(crate) struct SFrameBuilder {
    writer: Option<SFrameWriter>,
    column_names: Vec<String>,
    dtypes: Vec<FlexTypeEnum>,
    dir_path: String,
    /// When `Some`, this is an anonymous builder — the cache dir is removed
    /// on Drop if `finish()` was never called.
    cache_fs: Option<Arc<CacheFs>>,
}

impl SFrameBuilder {
    /// Open a builder targeting the given VFS and directory path.
    fn new(
        vfs: Arc<dyn VirtualFileSystem>,
        dir_path: String,
        column_names: Vec<String>,
        dtypes: Vec<FlexTypeEnum>,
    ) -> Result<Self> {
        let col_name_refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
        let writer = SFrameWriter::with_vfs(vfs, &dir_path, &col_name_refs, &dtypes)?;

        Ok(SFrameBuilder {
            writer: Some(writer),
            column_names,
            dtypes,
            dir_path,
            cache_fs: None,
        })
    }

    /// Create an anonymous builder backed by the global `CacheFs`.
    ///
    /// Allocates a fresh cache directory and arranges for it to be cleaned
    /// up if the builder is dropped without calling `finish()`.
    pub(crate) fn anonymous(column_names: Vec<String>, dtypes: Vec<FlexTypeEnum>) -> Result<Self> {
        let cache_fs = global_cache_fs();
        let dir_path = cache_fs.alloc_dir();
        let vfs: Arc<dyn VirtualFileSystem> = Arc::new(ArcCacheFsVfs(cache_fs.clone()));
        let mut builder = Self::new(vfs, dir_path, column_names, dtypes)?;
        builder.cache_fs = Some(cache_fs.clone());
        Ok(builder)
    }

    /// Write pre-built column vectors directly.
    pub(crate) fn write_columns(&mut self, cols: &[Vec<FlexType>]) -> Result<()> {
        self.writer
            .as_mut()
            .expect("write_columns called after finish")
            .write_columns(cols)
    }

    /// Write an `SFrameRows` batch in chunks to limit peak memory.
    pub(crate) fn write_batch_chunked(&mut self, batch: &SFrameRows, chunk_size: usize) -> Result<()> {
        let nrows = batch.num_rows();
        let ncols = batch.num_columns();
        let mut offset = 0;

        while offset < nrows {
            let end = (offset + chunk_size).min(nrows);
            let chunk_len = end - offset;

            let mut col_vecs: Vec<Vec<FlexType>> = Vec::with_capacity(ncols);
            for c in 0..ncols {
                let col = batch.column(c);
                let mut v = Vec::with_capacity(chunk_len);
                for r in offset..end {
                    v.push(col.get(r));
                }
                col_vecs.push(v);
            }

            self.write_columns(&col_vecs)?;
            offset = end;
        }
        Ok(())
    }

    /// Write rows from `batch` in the order given by `indices`, in chunks.
    /// This avoids building a full sorted copy of the batch.
    pub(crate) fn write_indexed_chunked(
        &mut self,
        batch: &SFrameRows,
        indices: &[usize],
        chunk_size: usize,
    ) -> Result<()> {
        let ncols = batch.num_columns();
        let total = indices.len();
        let mut offset = 0;

        while offset < total {
            let end = (offset + chunk_size).min(total);
            let chunk_len = end - offset;

            let mut col_vecs: Vec<Vec<FlexType>> = Vec::with_capacity(ncols);
            for c in 0..ncols {
                let col = batch.column(c);
                let mut v = Vec::with_capacity(chunk_len);
                for &idx in &indices[offset..end] {
                    v.push(col.get(idx));
                }
                col_vecs.push(v);
            }

            self.write_columns(&col_vecs)?;
            offset = end;
        }
        Ok(())
    }

    /// Finalize the writer and return an SFrame backed by the written data.
    ///
    /// For anonymous builders (created via `anonymous()`), the returned
    /// SFrame holds an `AnonymousStore` keep-alive that cleans up the
    /// cache directory when the SFrame is dropped.
    pub(crate) fn finish(mut self) -> Result<SFrame> {
        let writer = self
            .writer
            .take()
            .expect("finish called twice on SFrameBuilder");
        let total_rows = writer.finish()?;

        // Build the plan node — anonymous builders get a keep-alive guard.
        let plan = if let Some(ref cache_fs) = self.cache_fs {
            let store: Arc<dyn Send + Sync> = Arc::new(AnonymousStore {
                path: self.dir_path.clone(),
                cache_fs: cache_fs.clone(),
            });
            PlannerNode::sframe_source_cached(
                &self.dir_path,
                self.column_names.clone(),
                self.dtypes.clone(),
                total_rows,
                store,
            )
        } else {
            PlannerNode::sframe_source(
                &self.dir_path,
                self.column_names.clone(),
                self.dtypes.clone(),
                total_rows,
            )
        };

        let columns: Vec<SArray> = self
            .dtypes
            .iter()
            .enumerate()
            .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(total_rows), i))
            .collect();

        Ok(SFrame::new_with_columns(columns, self.column_names.clone()))
    }
}

impl Drop for SFrameBuilder {
    fn drop(&mut self) {
        if self.writer.is_some() {
            // finish() was never called — clean up the directory for
            // anonymous builders (cache-backed). Non-anonymous builders
            // leave the directory in place since the caller manages it.
            if let Some(ref cache_fs) = self.cache_fs {
                cache_fs.remove_dir(&self.dir_path).ok();
            }
        }
    }
}

/// Write a materialized batch to the global CacheFs and return an SFrame
/// backed by an SFrameSource plan node pointing at the cache path.
///
/// For empty batches, returns a MaterializedSource-backed SFrame to avoid
/// creating empty SFrame files on disk.
fn write_to_cache(batch: SFrameRows, column_names: Vec<String>) -> Result<SFrame> {
    let num_rows = batch.num_rows();
    let dtypes = batch.dtypes();

    if num_rows == 0 {
        let plan = PlannerNode::materialized(batch);
        let columns: Vec<SArray> = dtypes
            .iter()
            .enumerate()
            .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(0), i))
            .collect();
        return Ok(SFrame::new_with_columns(columns, column_names));
    }

    let mut builder = SFrameBuilder::anonymous(column_names, dtypes)?;
    builder.write_batch_chunked(&batch, DEFAULT_CHUNK_SIZE)?;
    builder.finish()
}

/// A columnar dataframe with lazy evaluation.
pub struct SFrame {
    columns: Vec<SArray>,
    column_names: Vec<String>,
    metadata: HashMap<String, String>,
}

impl SFrame {
    /// Internal constructor with default empty metadata.
    pub(crate) fn new_with_columns(columns: Vec<SArray>, column_names: Vec<String>) -> Self {
        SFrame {
            columns,
            column_names,
            metadata: HashMap::new(),
        }
    }

    /// Read an SFrame from disk.
    pub fn read(path: &str) -> Result<Self> {
        let reader = SFrameReader::open(path)?;
        let col_names: Vec<String> = reader.column_names().to_vec();
        let col_types: Vec<FlexTypeEnum> = reader
            .group_index
            .columns
            .iter()
            .map(|c| c.dtype)
            .collect();
        let num_rows = reader.num_rows();

        let plan = PlannerNode::sframe_source(path, col_names.clone(), col_types.clone(), num_rows);

        let columns: Vec<SArray> = col_types
            .iter()
            .enumerate()
            .map(|(i, &dtype)| {
                SArray::from_plan(plan.clone(), dtype, Some(num_rows), i)
            })
            .collect();

        let mut sf = SFrame::new_with_columns(columns, col_names);
        // Load metadata from disk
        let reader = SFrameReader::open(path)?;
        sf.metadata = reader.metadata().clone();
        Ok(sf)
    }

    /// Read a CSV file into an SFrame.
    ///
    /// Parses the CSV in chunks to limit peak memory: the file is tokenized
    /// and types are inferred up-front, then rows are converted and written
    /// to the cache in `DEFAULT_CHUNK_SIZE` increments.  Each chunk is an
    /// independent parse unit (the C++ "split at line boundaries, parse
    /// independently" strategy — tokenization is serial, but row→FlexType
    /// conversion is per-chunk and parallelizable).
    pub fn from_csv(path: &str, options: Option<CsvOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();
        let content = std::fs::read_to_string(path).map_err(SFrameError::Io)?;
        let schema = csv_parser::tokenize_and_infer(&content, &opts)?;

        // Resolve output names/types (respects output_columns subsetting)
        let (out_names, out_types) = schema.output_names_types();

        if schema.raw_rows.is_empty() {
            let batch = SFrameRows::empty(&out_types);
            let plan = PlannerNode::materialized(batch);
            let columns: Vec<SArray> = out_types
                .iter()
                .enumerate()
                .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(0), i))
                .collect();
            return Ok(SFrame::new_with_columns(columns, out_names));
        }

        let mut builder = SFrameBuilder::anonymous(out_names, out_types)?;

        let total = schema.raw_rows.len();
        let mut offset = 0;
        while offset < total {
            let end = (offset + DEFAULT_CHUNK_SIZE).min(total);
            let col_vecs = csv_parser::parse_rows_range(&schema, offset, end)?;
            builder.write_columns(&col_vecs)?;
            offset = end;
        }

        builder.finish()
    }

    /// Read a JSON Lines file into an SFrame.
    ///
    /// Parses the JSON file up-front to discover column names and types,
    /// then writes rows to the cache in `DEFAULT_CHUNK_SIZE` chunks.
    pub fn from_json(path: &str) -> Result<Self> {
        let parsed = json_io::parse_json_file_schema(path)?;

        if parsed.rows.is_empty() {
            return Ok(SFrame::new_with_columns(Vec::new(), Vec::new()));
        }

        let mut builder = SFrameBuilder::anonymous(
            parsed.column_names.clone(),
            parsed.column_types.clone(),
        )?;

        let total = parsed.rows.len();
        let mut offset = 0;
        while offset < total {
            let end = (offset + DEFAULT_CHUNK_SIZE).min(total);
            let col_vecs = json_io::rows_to_columns_range(&parsed, offset, end);
            builder.write_columns(&col_vecs)?;
            offset = end;
        }

        builder.finish()
    }

    /// Build an SFrame from named columns.
    pub fn from_columns(cols: Vec<(&str, SArray)>) -> Result<Self> {
        let column_names: Vec<String> = cols.iter().map(|(name, _)| name.to_string()).collect();
        let columns: Vec<SArray> = cols.into_iter().map(|(_, col)| col).collect();
        Ok(SFrame::new_with_columns(columns, column_names))
    }

    /// Number of rows.
    pub fn num_rows(&self) -> Result<u64> {
        if self.columns.is_empty() {
            return Ok(0);
        }
        self.columns[0].len()
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Column names.
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Column types.
    pub fn column_types(&self) -> Vec<FlexTypeEnum> {
        self.columns.iter().map(|c| c.dtype()).collect()
    }

    /// Schema as (name, type) pairs.
    pub fn schema(&self) -> Vec<(String, FlexTypeEnum)> {
        self.column_names
            .iter()
            .zip(self.columns.iter())
            .map(|(name, col)| (name.clone(), col.dtype()))
            .collect()
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Remove a metadata key.
    pub fn remove_metadata(&mut self, key: &str) {
        self.metadata.remove(key);
    }

    /// Get all metadata.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Access a column by name.
    pub fn column(&self, name: &str) -> Result<&SArray> {
        let idx = self.column_index(name)?;
        Ok(&self.columns[idx])
    }

    /// Select specific columns by name.
    pub fn select(&self, names: &[&str]) -> Result<SFrame> {
        let mut new_columns = Vec::with_capacity(names.len());
        let mut new_names = Vec::with_capacity(names.len());
        for &name in names {
            let idx = self.column_index(name)?;
            new_columns.push(self.columns[idx].clone());
            new_names.push(name.to_string());
        }
        Ok(SFrame::new_with_columns(new_columns, new_names))
    }

    /// Add a column (returns a new SFrame).
    pub fn add_column(&self, name: &str, col: SArray) -> Result<SFrame> {
        if self.column_names.contains(&name.to_string()) {
            return Err(SFrameError::Format(format!(
                "Column '{}' already exists",
                name
            )));
        }
        let mut columns = self.columns.clone();
        let mut names = self.column_names.clone();
        columns.push(col);
        names.push(name.to_string());
        Ok(SFrame::new_with_columns(columns, names))
    }

    /// Remove a column (returns a new SFrame).
    pub fn remove_column(&self, name: &str) -> Result<SFrame> {
        let idx = self.column_index(name)?;
        let mut columns = self.columns.clone();
        let mut names = self.column_names.clone();
        columns.remove(idx);
        names.remove(idx);
        Ok(SFrame::new_with_columns(columns, names))
    }

    /// Filter rows by a predicate on a named column.
    ///
    /// Returns a lazy SFrame whose plan wraps a filter operator. No
    /// materialization occurs until the result is consumed (e.g., by
    /// `head()`, `save()`, or `num_rows()`).
    pub fn filter(
        &self,
        column_name: &str,
        pred: Arc<dyn Fn(&FlexType) -> bool + Send + Sync>,
    ) -> Result<SFrame> {
        let filter_col_idx = self.column_index(column_name)?;

        // Lazy path: if all columns share the same plan, build a filter node
        if let Some(plan) = self.shared_plan() {
            let plan_col_idx = self.columns[filter_col_idx].column_index();
            let filtered_plan = PlannerNode::filter(plan.clone(), plan_col_idx, pred);

            let columns: Vec<SArray> = self
                .columns
                .iter()
                .map(|c| {
                    SArray::from_plan(filtered_plan.clone(), c.dtype(), None, c.column_index())
                })
                .collect();

            return Ok(SFrame::new_with_columns(columns, self.column_names.clone()));
        }

        // Fallback: materialize, filter, rebuild
        let batch = self.materialize_batch()?;
        let filtered = batch.filter_by_column(filter_col_idx, &*pred)?;
        self.from_batch(filtered)
    }

    /// Append another SFrame vertically.
    ///
    /// Returns a lazy SFrame backed by an Append plan node when possible,
    /// avoiding immediate materialization of both sides.
    pub fn append(&self, other: &SFrame) -> Result<SFrame> {
        if self.column_names != other.column_names {
            return Err(SFrameError::Format(
                "Column names must match for append".to_string(),
            ));
        }

        // Lazy path: build an Append plan node
        if self.shared_plan().is_some() && other.shared_plan().is_some() {
            let self_indices: Vec<usize> =
                self.columns.iter().map(|c| c.column_index()).collect();
            let other_indices: Vec<usize> =
                other.columns.iter().map(|c| c.column_index()).collect();

            let left =
                PlannerNode::project(self.shared_plan().unwrap().clone(), self_indices);
            let right =
                PlannerNode::project(other.shared_plan().unwrap().clone(), other_indices);
            let appended = PlannerNode::append(left, right);

            let columns: Vec<SArray> = self
                .columns
                .iter()
                .enumerate()
                .map(|(i, c)| SArray::from_plan(appended.clone(), c.dtype(), None, i))
                .collect();

            return Ok(SFrame::new_with_columns(columns, self.column_names.clone()));
        }

        // Fallback: materialize both sides
        let mut batch1 = self.materialize_batch()?;
        let batch2 = other.materialize_batch()?;
        batch1.append(&batch2)?;
        self.from_batch(batch1)
    }

    /// Return the first n rows as a new SFrame.
    ///
    /// Only pulls enough batches from the stream to fill n rows, then stops.
    /// This avoids materializing the entire dataset for small head requests.
    pub fn head(&self, n: usize) -> Result<SFrame> {
        if n == 0 {
            let dtypes = self.column_types();
            let batch = SFrameRows::empty(&dtypes);
            return self.from_batch(batch);
        }
        let stream = self.compile_stream()?;
        let batch = materialize_head_sync(stream, n)?;
        self.from_batch(batch)
    }

    /// Sort by one or more columns.
    ///
    /// Uses `sort_indices` to compute the sort permutation without copying
    /// the full dataset, then writes the reordered rows in chunks via
    /// `CacheSFrameBuilder::write_indexed_chunked`.
    pub fn sort(&self, keys: &[(&str, SortOrder)]) -> Result<SFrame> {
        let sort_keys: Vec<SortKey> = keys
            .iter()
            .map(|(name, order)| {
                let idx = self.column_index(name)?;
                Ok(match order {
                    SortOrder::Ascending => SortKey::asc(idx),
                    SortOrder::Descending => SortKey::desc(idx),
                })
            })
            .collect::<Result<_>>()?;

        // Estimate data size and decide between in-memory vs external sort
        let estimated_size = self.estimate_size();
        let budget = sframe_query::config::SFrameConfig::global().sort_memory_budget;

        if estimated_size <= budget {
            self.sort_in_memory(&sort_keys)
        } else {
            crate::external_sort::external_sort(self, &sort_keys)
        }
    }

    /// Sort entirely in memory. Used when data fits within the sort memory budget.
    pub(crate) fn sort_in_memory(&self, sort_keys: &[SortKey]) -> Result<SFrame> {
        let stream = self.compile_stream()?;
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SFrameError::Format(format!("Runtime error: {}", e)))?;
        let (batch, indices) = rt.block_on(sort::sort_indices(stream, sort_keys))?;

        if batch.num_rows() == 0 {
            return self.from_batch(batch);
        }

        let mut builder =
            SFrameBuilder::anonymous(self.column_names.clone(), self.column_types())?;
        builder.write_indexed_chunked(&batch, &indices, DEFAULT_CHUNK_SIZE)?;
        builder.finish()
    }

    /// Estimate the in-memory size of this SFrame's data in bytes.
    pub(crate) fn estimate_size(&self) -> usize {
        let num_rows = self.num_rows().unwrap_or(0) as usize;
        let per_row: usize = self
            .column_types()
            .iter()
            .map(|dt| match dt {
                FlexTypeEnum::Integer | FlexTypeEnum::Float => 9,
                FlexTypeEnum::String => 32,
                _ => 64,
            })
            .sum();
        num_rows * per_row
    }

    /// Join with another SFrame on a single column.
    pub fn join(
        &self,
        other: &SFrame,
        left_col: &str,
        right_col: &str,
        how: JoinType,
    ) -> Result<SFrame> {
        self.join_on(other, &[(left_col, right_col)], how)
    }

    /// Join with another SFrame on one or more column pairs.
    pub fn join_on(
        &self,
        other: &SFrame,
        on: &[(&str, &str)],
        how: JoinType,
    ) -> Result<SFrame> {
        if on.is_empty() {
            return Err(SFrameError::Format("Join requires at least one key pair".to_string()));
        }

        let pairs: Vec<(usize, usize)> = on
            .iter()
            .map(|&(l, r)| Ok((self.column_index(l)?, other.column_index(r)?)))
            .collect::<Result<_>>()?;

        let right_key_indices: HashSet<usize> =
            pairs.iter().map(|&(_, r)| r).collect();

        let left_stream = self.compile_stream()?;
        let right_stream = other.compile_stream()?;

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SFrameError::Format(format!("Runtime error: {}", e)))?;

        let joined = rt.block_on(join::join(
            left_stream,
            right_stream,
            &JoinOn::multi(pairs),
            how,
        ))?;

        // Build output column names: all left cols + right cols (minus join keys)
        let mut names: Vec<String> = self.column_names.clone();
        for (i, name) in other.column_names.iter().enumerate() {
            if !right_key_indices.contains(&i) {
                let out_name = if names.contains(name) {
                    format!("{}.1", name)
                } else {
                    name.clone()
                };
                names.push(out_name);
            }
        }

        write_to_cache(joined, names)
    }

    /// Group by columns and aggregate.
    pub fn groupby(&self, key_names: &[&str], agg_specs: Vec<AggSpec>) -> Result<SFrame> {
        let key_indices: Vec<usize> = key_names
            .iter()
            .map(|name| self.column_index(name))
            .collect::<Result<_>>()?;

        let stream = self.compile_stream()?;
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SFrameError::Format(format!("Runtime error: {}", e)))?;

        let grouped = rt.block_on(groupby::groupby(stream, &key_indices, &agg_specs))?;

        // Build output names: key columns + agg output names
        let mut names: Vec<String> = key_names.iter().map(|s| s.to_string()).collect();
        for spec in &agg_specs {
            names.push(spec.output_name.clone());
        }

        write_to_cache(grouped, names)
    }

    // === Phase 11.1: Column Mutation ===

    /// Replace a column with a new SArray (returns a new SFrame).
    pub fn replace_column(&self, name: &str, col: SArray) -> Result<SFrame> {
        let idx = self.column_index(name)?;
        let mut columns = self.columns.clone();
        columns[idx] = col;
        Ok(SFrame::new_with_columns(columns, self.column_names.clone()))
    }

    /// Rename columns according to a mapping (returns a new SFrame).
    pub fn rename(&self, mapping: &HashMap<&str, &str>) -> Result<SFrame> {
        let mut new_names = self.column_names.clone();
        for (&old_name, &new_name) in mapping {
            let idx = self.column_index(old_name)?;
            // Check that new name doesn't conflict with an existing column
            // (unless it's the column being renamed itself)
            if let Some(existing_idx) = new_names.iter().position(|n| n == new_name) {
                if existing_idx != idx {
                    return Err(SFrameError::Format(format!(
                        "Column '{}' already exists",
                        new_name
                    )));
                }
            }
            new_names[idx] = new_name.to_string();
        }
        Ok(SFrame::new_with_columns(self.columns.clone(), new_names))
    }

    /// Swap two columns by name (returns a new SFrame).
    pub fn swap_columns(&self, name1: &str, name2: &str) -> Result<SFrame> {
        let idx1 = self.column_index(name1)?;
        let idx2 = self.column_index(name2)?;
        let mut columns = self.columns.clone();
        let mut names = self.column_names.clone();
        columns.swap(idx1, idx2);
        names.swap(idx1, idx2);
        Ok(SFrame::new_with_columns(columns, names))
    }

    // === Phase 11.2: Missing Value Handling ===

    /// Drop rows with missing values.
    ///
    /// - `column_name`: if `Some`, only check that column; if `None`, check all columns.
    /// - `how`: `"any"` drops rows where any specified column is Undefined,
    ///          `"all"` drops rows where all specified columns are Undefined.
    pub fn dropna(&self, column_name: Option<&str>, how: &str) -> Result<SFrame> {
        let batch = self.materialize_batch()?;
        let nrows = batch.num_rows();
        let ncols = batch.num_columns();

        let check_cols: Vec<usize> = if let Some(name) = column_name {
            vec![self.column_index(name)?]
        } else {
            (0..ncols).collect()
        };

        let keep: Vec<bool> = (0..nrows)
            .map(|row| {
                let undefs = check_cols
                    .iter()
                    .filter(|&&c| matches!(batch.column(c).get(row), FlexType::Undefined))
                    .count();
                match how {
                    "all" => undefs < check_cols.len(), // keep unless ALL are undef
                    _ => undefs == 0,                    // "any": keep only if none are undef
                }
            })
            .collect();

        let mut col_vecs: Vec<Vec<FlexType>> = vec![Vec::new(); ncols];
        for (row, &kept) in keep.iter().enumerate() {
            if kept {
                for col in 0..ncols {
                    col_vecs[col].push(batch.column(col).get(row).clone());
                }
            }
        }

        let dtypes = self.column_types();
        let result = SFrameRows::from_column_vecs(col_vecs, &dtypes)?;
        self.from_batch(result)
    }

    /// Fill missing values in a column.
    pub fn fillna(&self, column_name: &str, value: FlexType) -> Result<SFrame> {
        let idx = self.column_index(column_name)?;
        let filled = self.columns[idx].fillna(value);
        self.replace_column(column_name, filled)
    }

    // === Phase 11.3: Sampling & Splitting ===

    /// Random sample of rows.
    pub fn sample(&self, fraction: f64, seed: Option<u64>) -> Result<SFrame> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let batch = self.materialize_batch()?;
        let nrows = batch.num_rows();
        let ncols = batch.num_columns();
        let seed = seed.unwrap_or(42);
        let threshold = (fraction * u64::MAX as f64) as u64;

        let mut col_vecs: Vec<Vec<FlexType>> = vec![Vec::new(); ncols];
        for row in 0..nrows {
            let mut hasher = DefaultHasher::new();
            (seed, row as u64).hash(&mut hasher);
            if hasher.finish() < threshold {
                for col in 0..ncols {
                    col_vecs[col].push(batch.column(col).get(row).clone());
                }
            }
        }

        let dtypes = self.column_types();
        let result = SFrameRows::from_column_vecs(col_vecs, &dtypes)?;
        self.from_batch(result)
    }

    /// Random split into two SFrames.
    ///
    /// Returns `(train, test)` where `train` contains approximately `fraction`
    /// of the rows and `test` contains the rest.
    pub fn random_split(&self, fraction: f64, seed: Option<u64>) -> Result<(SFrame, SFrame)> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let batch = self.materialize_batch()?;
        let nrows = batch.num_rows();
        let ncols = batch.num_columns();
        let seed = seed.unwrap_or(42);
        let threshold = (fraction * u64::MAX as f64) as u64;

        let mut left_vecs: Vec<Vec<FlexType>> = vec![Vec::new(); ncols];
        let mut right_vecs: Vec<Vec<FlexType>> = vec![Vec::new(); ncols];

        for row in 0..nrows {
            let mut hasher = DefaultHasher::new();
            (seed, row as u64).hash(&mut hasher);
            let target = if hasher.finish() < threshold {
                &mut left_vecs
            } else {
                &mut right_vecs
            };
            for col in 0..ncols {
                target[col].push(batch.column(col).get(row).clone());
            }
        }

        let dtypes = self.column_types();
        let left = SFrameRows::from_column_vecs(left_vecs, &dtypes)?;
        let right = SFrameRows::from_column_vecs(right_vecs, &dtypes)?;
        Ok((self.from_batch(left)?, self.from_batch(right)?))
    }

    /// Top-k rows by a column.
    pub fn topk(&self, column_name: &str, k: usize, reverse: bool) -> Result<SFrame> {
        let order = if reverse {
            SortOrder::Ascending
        } else {
            SortOrder::Descending
        };
        let sorted = self.sort(&[(column_name, order)])?;
        sorted.head(k)
    }

    // === Phase 11.4: Reshaping ===

    /// Pack multiple columns into a single Dict column.
    ///
    /// Each row becomes a dict mapping column_name → value for the packed columns.
    /// The packed columns are removed and replaced by a new column.
    pub fn pack_columns(
        &self,
        columns: &[&str],
        new_column_name: &str,
    ) -> Result<SFrame> {
        let pack_indices: Vec<usize> = columns
            .iter()
            .map(|&name| self.column_index(name))
            .collect::<Result<_>>()?;
        let pack_set: HashSet<usize> = pack_indices.iter().copied().collect();
        let pack_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();

        let batch = self.materialize_batch()?;
        let nrows = batch.num_rows();

        // Build non-packed columns
        let mut new_names: Vec<String> = Vec::new();
        let mut new_col_vecs: Vec<Vec<FlexType>> = Vec::new();
        let mut new_dtypes: Vec<FlexTypeEnum> = Vec::new();
        for (i, name) in self.column_names.iter().enumerate() {
            if !pack_set.contains(&i) {
                new_names.push(name.clone());
                let mut col = Vec::with_capacity(nrows);
                for row in 0..nrows {
                    col.push(batch.column(i).get(row).clone());
                }
                new_col_vecs.push(col);
                new_dtypes.push(self.columns[i].dtype());
            }
        }

        // Build the packed dict column
        let mut dict_col = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let entries: Vec<(FlexType, FlexType)> = pack_indices
                .iter()
                .enumerate()
                .map(|(pi, &ci)| {
                    (
                        FlexType::String(pack_names[pi].clone().into()),
                        batch.column(ci).get(row).clone(),
                    )
                })
                .collect();
            dict_col.push(FlexType::Dict(Arc::from(entries)));
        }
        new_names.push(new_column_name.to_string());
        new_col_vecs.push(dict_col);
        new_dtypes.push(FlexTypeEnum::Dict);

        let result = SFrameRows::from_column_vecs(new_col_vecs, &new_dtypes)?;
        write_to_cache(result, new_names)
    }

    /// Unpack a Dict/List column into separate columns.
    ///
    /// For Dict columns: creates one column per unique key.
    /// For List columns: creates numbered columns (prefix.0, prefix.1, ...).
    pub fn unpack_column(
        &self,
        column_name: &str,
        prefix: Option<&str>,
    ) -> Result<SFrame> {
        let col_idx = self.column_index(column_name)?;
        let batch = self.materialize_batch()?;
        let nrows = batch.num_rows();
        let pfx = prefix.unwrap_or(column_name);

        // Determine keys/positions by scanning the column
        let col_dtype = self.columns[col_idx].dtype();

        match col_dtype {
            FlexTypeEnum::Dict => {
                // Collect all unique keys
                let mut all_keys: Vec<FlexType> = Vec::new();
                for row in 0..nrows {
                    if let FlexType::Dict(d) = batch.column(col_idx).get(row) {
                        for (k, _) in d.iter() {
                            if !all_keys.contains(k) {
                                all_keys.push(k.clone());
                            }
                        }
                    }
                }

                // Build unpacked columns
                let mut new_names: Vec<String> = Vec::new();
                let mut new_col_vecs: Vec<Vec<FlexType>> = Vec::new();
                let mut new_dtypes: Vec<FlexTypeEnum> = Vec::new();

                // Keep other columns
                for (i, name) in self.column_names.iter().enumerate() {
                    if i != col_idx {
                        new_names.push(name.clone());
                        let mut col = Vec::with_capacity(nrows);
                        for row in 0..nrows {
                            col.push(batch.column(i).get(row).clone());
                        }
                        new_col_vecs.push(col);
                        new_dtypes.push(self.columns[i].dtype());
                    }
                }

                // Add unpacked columns
                for key in &all_keys {
                    let key_str = format!("{}", key);
                    new_names.push(format!("{}.{}", pfx, key_str));

                    let mut col = Vec::with_capacity(nrows);
                    let mut inferred_type = FlexTypeEnum::String;
                    for row in 0..nrows {
                        if let FlexType::Dict(d) = batch.column(col_idx).get(row) {
                            let val = d.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
                            let v = val.unwrap_or(FlexType::Undefined);
                            if !matches!(v, FlexType::Undefined) && inferred_type == FlexTypeEnum::String {
                                inferred_type = match &v {
                                    FlexType::Integer(_) => FlexTypeEnum::Integer,
                                    FlexType::Float(_) => FlexTypeEnum::Float,
                                    FlexType::Vector(_) => FlexTypeEnum::Vector,
                                    FlexType::List(_) => FlexTypeEnum::List,
                                    FlexType::Dict(_) => FlexTypeEnum::Dict,
                                    _ => FlexTypeEnum::String,
                                };
                            }
                            col.push(v);
                        } else {
                            col.push(FlexType::Undefined);
                        }
                    }
                    new_col_vecs.push(col);
                    new_dtypes.push(inferred_type);
                }

                let result = SFrameRows::from_column_vecs(new_col_vecs, &new_dtypes)?;
                write_to_cache(result, new_names)
            }
            FlexTypeEnum::List => {
                // Determine max list length
                let mut max_len = 0usize;
                for row in 0..nrows {
                    if let FlexType::List(l) = batch.column(col_idx).get(row) {
                        max_len = max_len.max(l.len());
                    }
                }

                let mut new_names: Vec<String> = Vec::new();
                let mut new_col_vecs: Vec<Vec<FlexType>> = Vec::new();
                let mut new_dtypes: Vec<FlexTypeEnum> = Vec::new();

                // Keep other columns
                for (i, name) in self.column_names.iter().enumerate() {
                    if i != col_idx {
                        new_names.push(name.clone());
                        let mut col = Vec::with_capacity(nrows);
                        for row in 0..nrows {
                            col.push(batch.column(i).get(row).clone());
                        }
                        new_col_vecs.push(col);
                        new_dtypes.push(self.columns[i].dtype());
                    }
                }

                // Add unpacked columns
                for pos in 0..max_len {
                    new_names.push(format!("{}.{}", pfx, pos));
                    let mut col = Vec::with_capacity(nrows);
                    let mut inferred_type = FlexTypeEnum::String;
                    for row in 0..nrows {
                        if let FlexType::List(l) = batch.column(col_idx).get(row) {
                            let v = l.get(pos).cloned().unwrap_or(FlexType::Undefined);
                            if !matches!(v, FlexType::Undefined) && inferred_type == FlexTypeEnum::String {
                                inferred_type = match &v {
                                    FlexType::Integer(_) => FlexTypeEnum::Integer,
                                    FlexType::Float(_) => FlexTypeEnum::Float,
                                    _ => FlexTypeEnum::String,
                                };
                            }
                            col.push(v);
                        } else {
                            col.push(FlexType::Undefined);
                        }
                    }
                    new_col_vecs.push(col);
                    new_dtypes.push(inferred_type);
                }

                let result = SFrameRows::from_column_vecs(new_col_vecs, &new_dtypes)?;
                write_to_cache(result, new_names)
            }
            _ => Err(SFrameError::Format(format!(
                "Cannot unpack column '{}' of type {:?}; expected Dict or List",
                column_name, col_dtype
            ))),
        }
    }

    /// Unnest a List or Dict column (one row per element).
    ///
    /// For List: each list element becomes its own row, with other columns duplicated.
    /// For Dict: expands to (key, value) columns.
    pub fn stack(&self, column_name: &str, new_column_name: &str) -> Result<SFrame> {
        let col_idx = self.column_index(column_name)?;
        let batch = self.materialize_batch()?;
        let nrows = batch.num_rows();
        let ncols = batch.num_columns();
        let col_dtype = self.columns[col_idx].dtype();

        let mut new_names: Vec<String> = self.column_names.clone();
        new_names[col_idx] = new_column_name.to_string();

        let mut new_dtypes: Vec<FlexTypeEnum> = self.column_types();

        // For dict stacking, replace the column with key-value pair columns
        match col_dtype {
            FlexTypeEnum::List | FlexTypeEnum::Vector => {
                // Infer element type from first non-empty element
                let elem_type = if col_dtype == FlexTypeEnum::Vector {
                    FlexTypeEnum::Float
                } else {
                    // Scan for first non-empty list and use its first element's type
                    let mut found = FlexTypeEnum::String;
                    for row in 0..nrows {
                        if let FlexType::List(l) = batch.column(col_idx).get(row) {
                            if let Some(first) = l.first() {
                                found = match first {
                                    FlexType::Integer(_) => FlexTypeEnum::Integer,
                                    FlexType::Float(_) => FlexTypeEnum::Float,
                                    FlexType::String(_) => FlexTypeEnum::String,
                                    FlexType::Vector(_) => FlexTypeEnum::Vector,
                                    FlexType::List(_) => FlexTypeEnum::List,
                                    FlexType::Dict(_) => FlexTypeEnum::Dict,
                                    FlexType::DateTime(_) => FlexTypeEnum::DateTime,
                                    FlexType::Undefined => FlexTypeEnum::String,
                                };
                                break;
                            }
                        }
                    }
                    found
                };
                new_dtypes[col_idx] = elem_type;
                let mut col_vecs: Vec<Vec<FlexType>> = vec![Vec::new(); ncols];

                for row in 0..nrows {
                    let elements: Vec<FlexType> = match batch.column(col_idx).get(row) {
                        FlexType::List(l) => l.to_vec(),
                        FlexType::Vector(v) => v.iter().map(|&f| FlexType::Float(f)).collect(),
                        _ => vec![FlexType::Undefined],
                    };

                    if elements.is_empty() {
                        // Keep row with Undefined for the stacked column
                        for c in 0..ncols {
                            if c == col_idx {
                                col_vecs[c].push(FlexType::Undefined);
                            } else {
                                col_vecs[c].push(batch.column(c).get(row).clone());
                            }
                        }
                    } else {
                        for elem in elements {
                            for c in 0..ncols {
                                if c == col_idx {
                                    col_vecs[c].push(elem.clone());
                                } else {
                                    col_vecs[c].push(batch.column(c).get(row).clone());
                                }
                            }
                        }
                    }
                }

                let result = SFrameRows::from_column_vecs(col_vecs, &new_dtypes)?;
                write_to_cache(result, new_names)
            }
            _ => Err(SFrameError::Format(format!(
                "Cannot stack column '{}' of type {:?}; expected List or Vector",
                column_name, col_dtype
            ))),
        }
    }

    // === Phase 11.5: Deduplication ===

    /// Remove duplicate rows.
    pub fn unique(&self) -> Result<SFrame> {
        let batch = self.materialize_batch()?;
        let nrows = batch.num_rows();
        let ncols = batch.num_columns();

        let mut seen = std::collections::HashSet::new();
        let mut col_vecs: Vec<Vec<FlexType>> = vec![Vec::new(); ncols];

        for row in 0..nrows {
            // Build a row key for deduplication
            let row_values: Vec<FlexType> = (0..ncols)
                .map(|c| batch.column(c).get(row).clone())
                .collect();
            let key = format!("{:?}", row_values);
            if seen.insert(key) {
                for (col, v) in row_values.into_iter().enumerate() {
                    col_vecs[col].push(v);
                }
            }
        }

        let dtypes = self.column_types();
        let result = SFrameRows::from_column_vecs(col_vecs, &dtypes)?;
        self.from_batch(result)
    }

    // === Phase 11.7: Tail ===

    /// Return the last n rows.
    pub fn tail(&self, n: usize) -> Result<SFrame> {
        let batch = self.materialize_batch()?;
        let nrows = batch.num_rows();
        if n >= nrows {
            return self.from_batch(batch);
        }
        let start = nrows - n;
        let ncols = batch.num_columns();
        let mut col_vecs: Vec<Vec<FlexType>> = vec![Vec::new(); ncols];
        for row in start..nrows {
            for col in 0..ncols {
                col_vecs[col].push(batch.column(col).get(row).clone());
            }
        }
        let dtypes = self.column_types();
        let result = SFrameRows::from_column_vecs(col_vecs, &dtypes)?;
        self.from_batch(result)
    }

    /// Materialize all lazy computations.
    pub fn materialize(&self) -> Result<SFrame> {
        let batch = self.materialize_batch()?;
        self.from_batch(batch)
    }

    /// Save to disk as an SFrame directory.
    ///
    /// Streams batches directly to the segment writer without collecting
    /// the entire dataset into memory first.
    pub fn save(&self, path: &str) -> Result<()> {
        let col_names: Vec<&str> = self.column_names.iter().map(|s| s.as_str()).collect();
        let dtypes = self.column_types();
        let mut writer = SFrameStreamWriter::new(path, &col_names, &dtypes)?;

        // Persist metadata
        for (key, value) in &self.metadata {
            writer.set_metadata(key, value);
        }

        let stream = self.compile_stream()?;
        for_each_batch_sync(stream, |batch| {
            writer.write_batch(&batch)
        })?;

        writer.finish()
    }

    /// Write to CSV file.
    pub fn to_csv(&self, path: &str, options: Option<CsvWriterOptions>) -> Result<()> {
        let opts = options.unwrap_or_default();
        let batch = self.materialize_batch()?;
        csv_writer::write_csv_file(path, &batch, &self.column_names, &opts)
    }

    /// Write to JSON Lines file.
    pub fn to_json(&self, path: &str) -> Result<()> {
        let batch = self.materialize_batch()?;
        json_io::write_json_file(path, &batch, &self.column_names)
    }

    /// Iterate over rows.
    pub fn iter_rows(&self) -> Result<Vec<Vec<FlexType>>> {
        let batch = self.materialize_batch()?;
        Ok(batch.to_rows())
    }

    // --- Internal helpers ---

    /// Get the shared plan if all columns use the same plan node.
    fn shared_plan(&self) -> Option<&Arc<PlannerNode>> {
        if self.columns.is_empty() {
            return None;
        }
        let first = self.columns[0].plan();
        if self.columns.iter().all(|c| Arc::ptr_eq(c.plan(), first)) {
            Some(first)
        } else {
            None
        }
    }

    /// Compile the SFrame's plan into a BatchStream.
    ///
    /// If all columns share a plan, projects to the needed columns and
    /// compiles once. Otherwise falls back to materializing all columns
    /// and wrapping the result.
    pub(crate) fn compile_stream(&self) -> Result<sframe_query::execute::BatchStream> {
        if self.columns.is_empty() {
            let plan = PlannerNode::materialized(SFrameRows::empty(&[]));
            return compile(&plan);
        }

        if let Some(plan) = self.shared_plan() {
            let indices: Vec<usize> =
                self.columns.iter().map(|c| c.column_index()).collect();
            let projected = PlannerNode::project(plan.clone(), indices);
            compile(&projected)
        } else {
            // Different plans: materialize all columns and wrap as a single batch
            let batch = self.materialize_batch()?;
            let plan = PlannerNode::materialized(batch);
            compile(&plan)
        }
    }

    fn column_index(&self, name: &str) -> Result<usize> {
        self.column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| SFrameError::Format(format!("Column '{}' not found", name)))
    }

    fn materialize_batch(&self) -> Result<SFrameRows> {
        if self.columns.is_empty() {
            return Ok(SFrameRows::empty(&[]));
        }

        // If all columns share the same plan, materialize once
        // Otherwise, materialize each column separately and combine
        let first_plan = self.columns[0].plan();
        let all_same = self.columns.iter().all(|c| Arc::ptr_eq(c.plan(), first_plan));

        if all_same {
            let stream = compile(first_plan)?;
            let batch = materialize_sync(stream)?;

            // Project to just the columns we need
            let indices: Vec<usize> = self.columns.iter().map(|c| c.column_index()).collect();
            batch.select_columns(&indices)
        } else {
            // Materialize each column independently
            let mut column_vecs: Vec<Vec<FlexType>> = Vec::new();
            for col in &self.columns {
                column_vecs.push(col.to_vec()?);
            }
            let dtypes: Vec<FlexTypeEnum> = self.columns.iter().map(|c| c.dtype()).collect();
            SFrameRows::from_column_vecs(column_vecs, &dtypes)
        }
    }

    fn from_batch(&self, batch: SFrameRows) -> Result<SFrame> {
        write_to_cache(batch, self.column_names.clone())
    }
}

impl std::fmt::Display for SFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let batch = match self.materialize_batch() {
            Ok(b) => b,
            Err(e) => return write!(f, "[SFrame error: {}]", e),
        };

        let nrows = batch.num_rows();
        let ncols = self.column_names.len();

        // Determine column widths
        let max_display = 10;
        let display_rows = nrows.min(max_display);

        let mut col_widths: Vec<usize> = self
            .column_names
            .iter()
            .map(|n| n.len())
            .collect();

        let mut cell_strings: Vec<Vec<String>> = Vec::new();
        for row_idx in 0..display_rows {
            let mut row_strs = Vec::new();
            for col_idx in 0..ncols {
                let val = batch.column(col_idx).get(row_idx);
                let s = format!("{}", val);
                let truncated = if s.len() > 30 {
                    format!("{}...", &s[..27])
                } else {
                    s
                };
                if truncated.len() > col_widths[col_idx] {
                    col_widths[col_idx] = truncated.len();
                }
                row_strs.push(truncated);
            }
            cell_strings.push(row_strs);
        }

        // Cap column widths
        for w in &mut col_widths {
            *w = (*w).min(30);
        }

        // Separator line
        let sep: String = col_widths
            .iter()
            .map(|w| format!("+{}", "-".repeat(w + 2)))
            .collect::<Vec<_>>()
            .join("")
            + "+";

        // Header
        writeln!(f, "{}", sep)?;
        let header: String = self
            .column_names
            .iter()
            .zip(col_widths.iter())
            .map(|(name, &w)| format!("| {:width$} ", name, width = w))
            .collect::<Vec<_>>()
            .join("")
            + "|";
        writeln!(f, "{}", header)?;
        writeln!(f, "{}", sep)?;

        // Data rows
        for row_strs in &cell_strings {
            let row: String = row_strs
                .iter()
                .zip(col_widths.iter())
                .map(|(s, &w)| format!("| {:width$} ", s, width = w))
                .collect::<Vec<_>>()
                .join("")
                + "|";
            writeln!(f, "{}", row)?;
        }

        if nrows > max_display {
            let dots: String = col_widths
                .iter()
                .map(|&w| format!("| {:width$} ", "...", width = w))
                .collect::<Vec<_>>()
                .join("")
                + "|";
            writeln!(f, "{}", dots)?;
        }

        writeln!(f, "{}", sep)?;
        write!(f, "[{} rows x {} columns]", nrows, ncols)
    }
}

/// Streaming SFrame writer that accepts `SFrameRows` batches.
///
/// Wraps the low-level [`SFrameWriter`] from `sframe-storage`, bridging
/// the columnar `SFrameRows` batch type to the storage writer's
/// column-vector interface. Data is buffered internally and flushed
/// to disk in blocks.
///
/// # Example
/// ```ignore
/// let mut writer = SFrameStreamWriter::new("output.sf", &["id", "name"], &[Integer, String])?;
/// writer.write_batch(&batch1)?;
/// writer.write_batch(&batch2)?;
/// writer.finish()?;
/// ```
pub struct SFrameStreamWriter {
    inner: SFrameWriter,
}

impl SFrameStreamWriter {
    /// Create a new streaming writer targeting the given directory.
    pub fn new(
        path: &str,
        column_names: &[&str],
        column_types: &[FlexTypeEnum],
    ) -> Result<Self> {
        let inner = SFrameWriter::new(path, column_names, column_types)?;
        Ok(SFrameStreamWriter { inner })
    }

    /// Write a batch of rows. Data is buffered internally and flushed
    /// to disk when enough rows accumulate for a full block.
    pub fn write_batch(&mut self, batch: &SFrameRows) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let col_vecs = batch.to_column_vecs();
        self.inner.write_columns(&col_vecs)
    }

    /// Set a metadata key-value pair to be persisted.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.inner.set_metadata(key, value);
    }

    /// Create a new streaming writer using a specific VFS backend.
    pub fn with_vfs(
        vfs: Arc<dyn VirtualFileSystem>,
        path: &str,
        column_names: &[&str],
        column_types: &[FlexTypeEnum],
    ) -> Result<Self> {
        let inner = SFrameWriter::with_vfs(vfs, path, column_names, column_types)?;
        Ok(SFrameStreamWriter { inner })
    }

    /// Finalize: flush remaining buffered data, write segment footer
    /// and metadata files.
    pub fn finish(self) -> Result<()> {
        self.inner.finish().map(|_| ())
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
    fn test_read_sframe() {
        let sf = SFrame::read(&format!("{}/business.sf", samples_dir())).unwrap();
        assert_eq!(sf.num_rows().unwrap(), 11536);
        assert_eq!(sf.num_columns(), 12);
        assert_eq!(sf.column_names()[0], "business_id");
    }

    #[test]
    fn test_column_access() {
        let sf = SFrame::read(&format!("{}/business.sf", samples_dir())).unwrap();
        let stars = sf.column("stars").unwrap();
        assert_eq!(stars.dtype(), FlexTypeEnum::Float);

        let head = stars.head(3).unwrap();
        assert_eq!(head.len(), 3);
        for v in &head {
            assert!(matches!(v, FlexType::Float(_)));
        }
    }

    #[test]
    fn test_select() {
        let sf = SFrame::read(&format!("{}/business.sf", samples_dir())).unwrap();
        let subset = sf.select(&["business_id", "stars"]).unwrap();
        assert_eq!(subset.num_columns(), 2);
        assert_eq!(subset.column_names(), &["business_id", "stars"]);
    }

    #[test]
    fn test_filter() {
        let sf = SFrame::read(&format!("{}/business.sf", samples_dir())).unwrap();
        let filtered = sf
            .filter("stars", Arc::new(|v| matches!(v, FlexType::Float(f) if *f >= 4.5)))
            .unwrap();

        assert_eq!(filtered.num_rows().unwrap(), 3020);
    }

    #[test]
    fn test_head() {
        let sf = SFrame::read(&format!("{}/business.sf", samples_dir())).unwrap();
        let head = sf.head(5).unwrap();
        assert_eq!(head.num_rows().unwrap(), 5);
        assert_eq!(head.num_columns(), 12);
    }

    #[test]
    fn test_sort() {
        let sf = SFrame::from_columns(vec![
            ("x", SArray::from_vec(
                vec![FlexType::Integer(3), FlexType::Integer(1), FlexType::Integer(2)],
                FlexTypeEnum::Integer,
            ).unwrap()),
        ]).unwrap();

        let sorted = sf.sort(&[("x", SortOrder::Ascending)]).unwrap();
        let rows = sorted.iter_rows().unwrap();
        assert_eq!(rows[0], vec![FlexType::Integer(1)]);
        assert_eq!(rows[1], vec![FlexType::Integer(2)]);
        assert_eq!(rows[2], vec![FlexType::Integer(3)]);
    }

    #[test]
    fn test_groupby() {
        let sf = SFrame::from_columns(vec![
            ("city", SArray::from_vec(
                vec![
                    FlexType::String("A".into()),
                    FlexType::String("B".into()),
                    FlexType::String("A".into()),
                    FlexType::String("B".into()),
                ],
                FlexTypeEnum::String,
            ).unwrap()),
            ("score", SArray::from_vec(
                vec![
                    FlexType::Integer(10),
                    FlexType::Integer(20),
                    FlexType::Integer(30),
                    FlexType::Integer(40),
                ],
                FlexTypeEnum::Integer,
            ).unwrap()),
        ]).unwrap();

        let grouped = sf
            .groupby(
                &["city"],
                vec![AggSpec::sum(1, "total"), AggSpec::count(1, "n")],
            )
            .unwrap();

        assert_eq!(grouped.num_rows().unwrap(), 2);
        assert_eq!(grouped.num_columns(), 3); // city, total, n
    }

    #[test]
    fn test_save_and_reload() {
        let sf = SFrame::from_columns(vec![
            ("id", SArray::from_vec(
                vec![FlexType::Integer(1), FlexType::Integer(2)],
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("name", SArray::from_vec(
                vec![FlexType::String("alice".into()), FlexType::String("bob".into())],
                FlexTypeEnum::String,
            ).unwrap()),
        ]).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.sf");
        let path_str = path.to_str().unwrap();

        sf.save(path_str).unwrap();

        let reloaded = SFrame::read(path_str).unwrap();
        assert_eq!(reloaded.num_rows().unwrap(), 2);
        assert_eq!(reloaded.num_columns(), 2);
        assert_eq!(reloaded.column_names(), &["id", "name"]);
    }

    #[test]
    fn test_display() {
        let sf = SFrame::from_columns(vec![
            ("x", SArray::from_vec(
                vec![FlexType::Integer(1), FlexType::Integer(2)],
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("y", SArray::from_vec(
                vec![FlexType::String("hello".into()), FlexType::String("world".into())],
                FlexTypeEnum::String,
            ).unwrap()),
        ]).unwrap();

        let s = format!("{}", sf);
        assert!(s.contains("x"));
        assert!(s.contains("y"));
        assert!(s.contains("hello"));
        assert!(s.contains("[2 rows x 2 columns]"));
    }

    #[test]
    fn test_from_csv() {
        let sf = SFrame::from_csv(
            &format!("{}/business.csv", samples_dir()),
            None,
        ).unwrap();

        assert_eq!(sf.num_rows().unwrap(), 11536);
        assert_eq!(sf.num_columns(), 12);
        assert_eq!(sf.column_names()[0], "business_id");
    }

    #[test]
    fn test_add_remove_column() {
        let sf = SFrame::from_columns(vec![
            ("x", SArray::from_vec(vec![FlexType::Integer(1)], FlexTypeEnum::Integer).unwrap()),
        ]).unwrap();

        let sf2 = sf.add_column(
            "y",
            SArray::from_vec(vec![FlexType::Float(1.5)], FlexTypeEnum::Float).unwrap(),
        ).unwrap();
        assert_eq!(sf2.num_columns(), 2);

        let sf3 = sf2.remove_column("x").unwrap();
        assert_eq!(sf3.num_columns(), 1);
        assert_eq!(sf3.column_names(), &["y"]);
    }

    #[test]
    fn test_stream_writer_batch_api() {
        use sframe_query::batch::{ColumnData, SFrameRows};

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("streamed.sf");
        let path_str = path.to_str().unwrap();

        let dtypes = &[FlexTypeEnum::Integer, FlexTypeEnum::String];
        let mut writer = SFrameStreamWriter::new(path_str, &["id", "name"], dtypes).unwrap();

        // Write two batches
        let batch1 = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1), Some(2)]),
            ColumnData::String(vec![Some("alice".into()), Some("bob".into())]),
        ]).unwrap();
        writer.write_batch(&batch1).unwrap();

        let batch2 = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(3)]),
            ColumnData::String(vec![Some("charlie".into())]),
        ]).unwrap();
        writer.write_batch(&batch2).unwrap();

        writer.finish().unwrap();

        // Verify via SFrame::read
        let sf = SFrame::read(path_str).unwrap();
        assert_eq!(sf.num_rows().unwrap(), 3);
        assert_eq!(sf.num_columns(), 2);

        let rows = sf.iter_rows().unwrap();
        assert_eq!(rows[0], vec![FlexType::Integer(1), FlexType::String("alice".into())]);
        assert_eq!(rows[1], vec![FlexType::Integer(2), FlexType::String("bob".into())]);
        assert_eq!(rows[2], vec![FlexType::Integer(3), FlexType::String("charlie".into())]);
    }

    // === Phase 11 Tests ===

    fn make_test_sf() -> SFrame {
        SFrame::from_columns(vec![
            ("id", SArray::from_vec(
                vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3), FlexType::Integer(4)],
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("name", SArray::from_vec(
                vec![
                    FlexType::String("alice".into()),
                    FlexType::String("bob".into()),
                    FlexType::Undefined,
                    FlexType::String("dave".into()),
                ],
                FlexTypeEnum::String,
            ).unwrap()),
            ("score", SArray::from_vec(
                vec![FlexType::Float(90.0), FlexType::Undefined, FlexType::Float(70.0), FlexType::Float(80.0)],
                FlexTypeEnum::Float,
            ).unwrap()),
        ]).unwrap()
    }

    #[test]
    fn test_replace_column() {
        let sf = make_test_sf();
        let new_col = SArray::from_vec(
            vec![FlexType::Integer(10), FlexType::Integer(20), FlexType::Integer(30), FlexType::Integer(40)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let sf2 = sf.replace_column("score", new_col).unwrap();
        assert_eq!(sf2.column_types()[2], FlexTypeEnum::Integer);
        let vals = sf2.column("score").unwrap().to_vec().unwrap();
        assert_eq!(vals[0], FlexType::Integer(10));
    }

    #[test]
    fn test_rename_columns() {
        let sf = make_test_sf();
        let mut mapping = HashMap::new();
        mapping.insert("id", "user_id");
        mapping.insert("name", "user_name");
        let sf2 = sf.rename(&mapping).unwrap();
        assert_eq!(sf2.column_names(), &["user_id", "user_name", "score"]);
    }

    #[test]
    fn test_swap_columns() {
        let sf = make_test_sf();
        let sf2 = sf.swap_columns("id", "score").unwrap();
        assert_eq!(sf2.column_names(), &["score", "name", "id"]);
        assert_eq!(sf2.column_types()[0], FlexTypeEnum::Float);
        assert_eq!(sf2.column_types()[2], FlexTypeEnum::Integer);
    }

    #[test]
    fn test_dropna_any() {
        let sf = make_test_sf();
        // Drop rows where ANY column is Undefined
        let cleaned = sf.dropna(None, "any").unwrap();
        assert_eq!(cleaned.num_rows().unwrap(), 2); // rows 0 and 3 survive
        let ids = cleaned.column("id").unwrap().to_vec().unwrap();
        assert_eq!(ids, vec![FlexType::Integer(1), FlexType::Integer(4)]);
    }

    #[test]
    fn test_dropna_all() {
        let sf = make_test_sf();
        // Drop rows where ALL columns are Undefined (none in our test data)
        let cleaned = sf.dropna(None, "all").unwrap();
        assert_eq!(cleaned.num_rows().unwrap(), 4);
    }

    #[test]
    fn test_dropna_single_column() {
        let sf = make_test_sf();
        let cleaned = sf.dropna(Some("name"), "any").unwrap();
        assert_eq!(cleaned.num_rows().unwrap(), 3); // row 2 dropped (name=Undefined)
    }

    #[test]
    fn test_fillna_sframe() {
        let sf = make_test_sf();
        let filled = sf.fillna("name", FlexType::String("unknown".into())).unwrap();
        let names = filled.column("name").unwrap().to_vec().unwrap();
        assert_eq!(names[2], FlexType::String("unknown".into()));
    }

    #[test]
    fn test_sample_sframe() {
        let sf = make_test_sf();
        let sampled = sf.sample(0.5, Some(123)).unwrap();
        assert!(sampled.num_rows().unwrap() <= 4);
        assert_eq!(sampled.num_columns(), 3);
    }

    #[test]
    fn test_random_split() {
        let sf = make_test_sf();
        let (train, test) = sf.random_split(0.5, Some(42)).unwrap();
        // All rows accounted for
        assert_eq!(
            train.num_rows().unwrap() + test.num_rows().unwrap(),
            4
        );
        assert_eq!(train.num_columns(), 3);
        assert_eq!(test.num_columns(), 3);
    }

    #[test]
    fn test_topk() {
        let sf = make_test_sf();
        let top2 = sf.topk("id", 2, false).unwrap();
        assert_eq!(top2.num_rows().unwrap(), 2);
        let ids = top2.column("id").unwrap().to_vec().unwrap();
        // Descending by default, so top 2 are 4 and 3
        assert_eq!(ids[0], FlexType::Integer(4));
        assert_eq!(ids[1], FlexType::Integer(3));
    }

    #[test]
    fn test_topk_reverse() {
        let sf = make_test_sf();
        let bottom2 = sf.topk("id", 2, true).unwrap();
        let ids = bottom2.column("id").unwrap().to_vec().unwrap();
        // Ascending (reverse=true), so bottom 2 are 1 and 2
        assert_eq!(ids[0], FlexType::Integer(1));
        assert_eq!(ids[1], FlexType::Integer(2));
    }

    #[test]
    fn test_multi_column_join() {
        let left = SFrame::from_columns(vec![
            ("dept", SArray::from_vec(
                vec![FlexType::String("eng".into()), FlexType::String("eng".into()), FlexType::String("sales".into())],
                FlexTypeEnum::String,
            ).unwrap()),
            ("region", SArray::from_vec(
                vec![FlexType::String("us".into()), FlexType::String("eu".into()), FlexType::String("us".into())],
                FlexTypeEnum::String,
            ).unwrap()),
            ("name", SArray::from_vec(
                vec![FlexType::String("alice".into()), FlexType::String("bob".into()), FlexType::String("charlie".into())],
                FlexTypeEnum::String,
            ).unwrap()),
        ]).unwrap();

        let right = SFrame::from_columns(vec![
            ("dept", SArray::from_vec(
                vec![FlexType::String("eng".into()), FlexType::String("eng".into())],
                FlexTypeEnum::String,
            ).unwrap()),
            ("region", SArray::from_vec(
                vec![FlexType::String("us".into()), FlexType::String("eu".into())],
                FlexTypeEnum::String,
            ).unwrap()),
            ("budget", SArray::from_vec(
                vec![FlexType::Float(100.0), FlexType::Float(80.0)],
                FlexTypeEnum::Float,
            ).unwrap()),
        ]).unwrap();

        let joined = left.join_on(&right, &[("dept", "dept"), ("region", "region")], JoinType::Inner).unwrap();
        assert_eq!(joined.num_rows().unwrap(), 2); // eng+us and eng+eu match
        assert_eq!(joined.num_columns(), 4); // dept, region, name, budget
        assert_eq!(joined.column_names(), &["dept", "region", "name", "budget"]);
    }

    #[test]
    fn test_unique_sframe() {
        let sf = SFrame::from_columns(vec![
            ("x", SArray::from_vec(
                vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(1), FlexType::Integer(3)],
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("y", SArray::from_vec(
                vec![
                    FlexType::String("a".into()),
                    FlexType::String("b".into()),
                    FlexType::String("a".into()),
                    FlexType::String("c".into()),
                ],
                FlexTypeEnum::String,
            ).unwrap()),
        ]).unwrap();

        let unique = sf.unique().unwrap();
        assert_eq!(unique.num_rows().unwrap(), 3); // (1,a) appears twice
    }

    #[test]
    fn test_tail_sframe() {
        let sf = make_test_sf();
        let last2 = sf.tail(2).unwrap();
        assert_eq!(last2.num_rows().unwrap(), 2);
        let ids = last2.column("id").unwrap().to_vec().unwrap();
        assert_eq!(ids[0], FlexType::Integer(3));
        assert_eq!(ids[1], FlexType::Integer(4));
    }

    // === Phase 11.4 + 7.5 Tests ===

    #[test]
    fn test_pack_columns() {
        let sf = SFrame::from_columns(vec![
            ("id", SArray::from_vec(
                vec![FlexType::Integer(1), FlexType::Integer(2)],
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("x", SArray::from_vec(
                vec![FlexType::Float(1.0), FlexType::Float(2.0)],
                FlexTypeEnum::Float,
            ).unwrap()),
            ("y", SArray::from_vec(
                vec![FlexType::Float(3.0), FlexType::Float(4.0)],
                FlexTypeEnum::Float,
            ).unwrap()),
        ]).unwrap();

        let packed = sf.pack_columns(&["x", "y"], "coords").unwrap();
        assert_eq!(packed.num_columns(), 2); // id + coords
        assert_eq!(packed.column_names(), &["id", "coords"]);
        assert_eq!(packed.column("coords").unwrap().dtype(), FlexTypeEnum::Dict);
    }

    #[test]
    fn test_unpack_dict_column() {
        // First pack, then unpack
        let sf = SFrame::from_columns(vec![
            ("id", SArray::from_vec(
                vec![FlexType::Integer(1), FlexType::Integer(2)],
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("x", SArray::from_vec(
                vec![FlexType::Float(1.0), FlexType::Float(2.0)],
                FlexTypeEnum::Float,
            ).unwrap()),
            ("y", SArray::from_vec(
                vec![FlexType::Float(3.0), FlexType::Float(4.0)],
                FlexTypeEnum::Float,
            ).unwrap()),
        ]).unwrap();

        let packed = sf.pack_columns(&["x", "y"], "coords").unwrap();
        let unpacked = packed.unpack_column("coords", Some("c")).unwrap();
        assert_eq!(unpacked.num_columns(), 3); // id + c.x + c.y
        assert!(unpacked.column_names().contains(&"c.x".to_string()));
        assert!(unpacked.column_names().contains(&"c.y".to_string()));
    }

    #[test]
    fn test_stack() {
        let sa = SArray::from_vec(
            vec![
                FlexType::List(Arc::from(vec![FlexType::Integer(1), FlexType::Integer(2)])),
                FlexType::List(Arc::from(vec![FlexType::Integer(3)])),
            ],
            FlexTypeEnum::List,
        ).unwrap();

        let sf = SFrame::from_columns(vec![
            ("id", SArray::from_vec(
                vec![FlexType::String("a".into()), FlexType::String("b".into())],
                FlexTypeEnum::String,
            ).unwrap()),
            ("items", sa),
        ]).unwrap();

        let stacked = sf.stack("items", "item").unwrap();
        assert_eq!(stacked.num_rows().unwrap(), 3); // 2 + 1 = 3 rows
        assert_eq!(stacked.column_names(), &["id", "item"]);
    }

    #[test]
    fn test_to_csv() {
        let sf = SFrame::from_columns(vec![
            ("id", SArray::from_vec(
                vec![FlexType::Integer(1), FlexType::Integer(2)],
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("name", SArray::from_vec(
                vec![FlexType::String("alice".into()), FlexType::String("bob".into())],
                FlexTypeEnum::String,
            ).unwrap()),
        ]).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.csv");
        let path_str = path.to_str().unwrap();

        sf.to_csv(path_str, None).unwrap();

        let content = std::fs::read_to_string(path_str).unwrap();
        assert!(content.contains("id,name"));
        assert!(content.contains("1,alice"));
        assert!(content.contains("2,bob"));
    }

    #[test]
    fn test_json_roundtrip() {
        let sf = SFrame::from_columns(vec![
            ("id", SArray::from_vec(
                vec![FlexType::Integer(1), FlexType::Integer(2)],
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("name", SArray::from_vec(
                vec![FlexType::String("alice".into()), FlexType::String("bob".into())],
                FlexTypeEnum::String,
            ).unwrap()),
            ("score", SArray::from_vec(
                vec![FlexType::Float(90.5), FlexType::Float(85.0)],
                FlexTypeEnum::Float,
            ).unwrap()),
        ]).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.jsonl");
        let path_str = path.to_str().unwrap();

        sf.to_json(path_str).unwrap();

        // Verify file content
        let content = std::fs::read_to_string(path_str).unwrap();
        assert!(content.contains("\"id\":1"));
        assert!(content.contains("\"name\":\"alice\""));

        // Read back
        let sf2 = SFrame::from_json(path_str).unwrap();
        assert_eq!(sf2.num_rows().unwrap(), 2);
        assert_eq!(sf2.num_columns(), 3);
    }

    // === Phase 13 Tests ===

    #[test]
    fn test_parallel_sort_large() {
        // Build a large enough SFrame to trigger parallel sort (>10K rows)
        let n = 15_000;
        let values: Vec<FlexType> = (0..n)
            .rev()
            .map(|i| FlexType::Integer(i))
            .collect();
        let sa = SArray::from_vec(values, FlexTypeEnum::Integer).unwrap();
        let sf = SFrame::from_columns(vec![("val", sa)]).unwrap();

        let sorted = sf.sort(&[("val", SortOrder::Ascending)]).unwrap();
        let rows = sorted.head(5).unwrap();
        assert_eq!(rows.num_rows().unwrap(), 5);

        let v = rows.column("val").unwrap().to_vec().unwrap();
        assert_eq!(v[0], FlexType::Integer(0));
        assert_eq!(v[1], FlexType::Integer(1));
        assert_eq!(v[4], FlexType::Integer(4));
    }

    // === Phase 16.1 Tests ===

    #[test]
    fn test_metadata_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.sf");
        let path_str = path.to_str().unwrap();

        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let mut sf = SFrame::from_columns(vec![("x", sa)]).unwrap();
        sf.set_metadata("author", "test_user");
        sf.set_metadata("version", "1.0");
        sf.save(path_str).unwrap();

        // Read back and verify metadata survived
        let sf2 = SFrame::read(path_str).unwrap();
        assert_eq!(sf2.get_metadata("author"), Some("test_user"));
        assert_eq!(sf2.get_metadata("version"), Some("1.0"));
        assert_eq!(sf2.get_metadata("nonexistent"), None);
        assert_eq!(sf2.num_rows().unwrap(), 2);
    }

    #[test]
    fn test_metadata_in_memory() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let mut sf = SFrame::from_columns(vec![("x", sa)]).unwrap();

        assert_eq!(sf.get_metadata("key"), None);
        sf.set_metadata("key", "value");
        assert_eq!(sf.get_metadata("key"), Some("value"));
        sf.remove_metadata("key");
        assert_eq!(sf.get_metadata("key"), None);
    }
}
