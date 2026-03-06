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
    compile, for_each_batch_sync, materialize_head_sync, materialize_sync, materialize_tail_sync,
};
use sframe_query::optimizer;
use sframe_query::planner::PlannerNode;
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
#[derive(Clone)]
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
        use sframe_storage::sframe_reader::SFrameMetadata;
        let meta = SFrameMetadata::open(path)?;
        let col_names: Vec<String> = meta.frame_index.column_names.clone();
        let col_types: Vec<FlexTypeEnum> = meta
            .group_index
            .columns
            .iter()
            .map(|c| c.dtype)
            .collect();
        let num_rows = meta.frame_index.nrows;

        let plan = PlannerNode::sframe_source(path, col_names.clone(), col_types.clone(), num_rows);

        let columns: Vec<SArray> = col_types
            .iter()
            .enumerate()
            .map(|(i, &dtype)| {
                SArray::from_plan(plan.clone(), dtype, Some(num_rows), i)
            })
            .collect();

        let mut sf = SFrame::new_with_columns(columns, col_names);
        sf.metadata = meta.frame_index.metadata;
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

        // Two-pass streaming: infer schema, then parse in parallel chunks
        let streaming = csv_parser::CsvStreamingParse::open(path, &opts)?;

        if streaming.column_names.is_empty() {
            let batch = SFrameRows::empty(&streaming.column_types);
            let plan = PlannerNode::materialized(batch);
            let columns: Vec<SArray> = streaming
                .column_types
                .iter()
                .enumerate()
                .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(0), i))
                .collect();
            return Ok(SFrame::new_with_columns(columns, streaming.column_names));
        }

        let mut builder = SFrameBuilder::anonymous(
            streaming.column_names.clone(),
            streaming.column_types.clone(),
        )?;

        streaming.parse_chunks(|col_vecs| builder.write_columns(&col_vecs))?;

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

    /// Read one or more Parquet files into an SFrame.
    ///
    /// The `path` may be a single file path or a glob pattern (e.g.
    /// `"data/*.parquet"`). Schemas are read from the first matched file;
    /// all files must share the same schema.
    ///
    /// Data is streamed through the parquet reader and written to an
    /// anonymous cache-backed SFrame in `DEFAULT_CHUNK_SIZE` chunks.
    pub fn from_parquet(path: &str) -> Result<Self> {
        let paths = sframe_parquet::parquet_reader::resolve_parquet_paths(path)?;
        Self::from_parquet_files_inner(&paths)
    }

    /// Read an explicit list of Parquet files into an SFrame.
    ///
    /// Schemas are read from the first file; all files must share the
    /// same schema.
    pub fn from_parquet_files(paths: &[&str]) -> Result<Self> {
        let path_bufs: Vec<std::path::PathBuf> =
            paths.iter().map(std::path::PathBuf::from).collect();
        Self::from_parquet_files_inner(&path_bufs)
    }

    /// Shared implementation for `from_parquet` and `from_parquet_files`.
    fn from_parquet_files_inner(paths: &[std::path::PathBuf]) -> Result<Self> {
        if paths.is_empty() {
            return Err(SFrameError::Format(
                "No parquet files provided".to_string(),
            ));
        }

        let (column_names, column_types) =
            sframe_parquet::parquet_reader::read_parquet_schema(paths[0].to_str().unwrap())?;

        if column_names.is_empty() {
            let batch = SFrameRows::empty(&column_types);
            let plan = PlannerNode::materialized(batch);
            let columns: Vec<SArray> = column_types
                .iter()
                .enumerate()
                .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(0), i))
                .collect();
            return Ok(SFrame::new_with_columns(columns, column_names));
        }

        let mut builder = SFrameBuilder::anonymous(column_names.clone(), column_types.clone())?;
        let mut iter = sframe_parquet::parquet_reader::read_parquet_batches(paths)?;
        while let Some(batch_result) = iter.next_batch() {
            let batch = batch_result?;
            builder.write_batch_chunked(&batch, DEFAULT_CHUNK_SIZE)?;
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

    /// Return the row count if cheaply available (no materialization).
    fn known_len(&self) -> Option<u64> {
        if self.columns.is_empty() {
            Some(0)
        } else {
            self.columns[0].known_len()
        }
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Compile the plan and stream batches through a bounded channel.
    ///
    /// A background thread compiles and drives the BatchIterator, sending
    /// each `SFrameRows` batch through the channel. Memory is bounded to
    /// `buffer + 1` batches.
    pub fn batch_channel(
        &self,
        buffer: usize,
    ) -> Result<std::sync::mpsc::Receiver<sframe_types::error::Result<SFrameRows>>> {
        let plan = self.fuse_plan()?;
        let (tx, rx) = std::sync::mpsc::sync_channel(buffer);
        std::thread::spawn(move || {
            let mut iter = match compile(&plan) {
                Ok(iter) => iter,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            while let Some(batch_result) = iter.next_batch() {
                if tx.send(batch_result).is_err() {
                    break; // receiver dropped
                }
            }
        });
        Ok(rx)
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

    /// Return a Graphviz DOT representation of the query plan.
    ///
    /// Shows both the logical plan (per-column) and the execution plan
    /// (fused/optimized) that will actually run. Shared subexpressions
    /// appear as a single node with multiple incoming edges.
    pub fn explain(&self) -> String {
        let mut buf = String::new();

        // --- Logical plan (what the user built) ---
        if let Some(plan) = self.shared_plan() {
            buf.push_str("Logical Plan:\n");
            buf.push_str(&plan.explain());
        } else {
            let named_plans: Vec<(&str, &Arc<PlannerNode>)> = self
                .column_names
                .iter()
                .zip(self.columns.iter())
                .map(|(name, col)| (name.as_str(), col.plan()))
                .collect();
            buf.push_str("Logical Plan:\n");
            buf.push_str(&PlannerNode::explain_multi(&named_plans));
        }

        // --- Execution plan (what will actually run) ---
        if let Ok(fused) = self.fuse_plan() {
            let optimized = optimizer::optimize(&fused);
            buf.push_str("\nExecution Plan:\n");
            buf.push_str(&optimized.explain());
        } else {
            buf.push_str("\nExecution Plan: columns will be materialized independently\n");
        }

        buf
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
                "Column '{name}' already exists"
            )));
        }

        if let Ok(fused) = self.fuse_plan() {
            let existing_ncols = self.columns.len();
            let col_plan = PlannerNode::project(col.plan().clone(), vec![col.column_index()]);

            let existing_len = fused.length();
            let new_len = col_plan.length();
            let compatible = match (existing_len, new_len) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            };

            if compatible {
                let unified = PlannerNode::column_union(vec![fused, col_plan]);
                let mut columns: Vec<SArray> = self
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(i, c)| SArray::from_plan(unified.clone(), c.dtype(), None, i))
                    .collect();
                columns.push(SArray::from_plan(
                    unified.clone(),
                    col.dtype(),
                    None,
                    existing_ncols,
                ));
                let mut names = self.column_names.clone();
                names.push(name.to_string());
                return Ok(SFrame::new_with_columns(columns, names));
            }
        }

        // Fallback: independent plans
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
        let fused = self.fuse_plan()?;

        let mask_source = PlannerNode::project(fused.clone(), vec![filter_col_idx]);
        let mask = PlannerNode::transform(
            mask_source,
            0,
            Arc::new(move |v: &FlexType| -> FlexType {
                if pred(v) {
                    FlexType::Integer(1)
                } else {
                    FlexType::Integer(0)
                }
            }),
            FlexTypeEnum::Integer,
        );

        let filtered_plan = PlannerNode::logical_filter(fused, mask);

        let columns: Vec<SArray> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| SArray::from_plan(filtered_plan.clone(), c.dtype(), None, i))
            .collect();

        Ok(SFrame::new_with_columns(columns, self.column_names.clone()))
    }

    /// Filter rows using a boolean mask SArray.
    ///
    /// Keeps rows where `mask` is non-zero. The mask must be an Integer
    /// SArray of the same length. This is the SArray-based alternative to
    /// `filter()` — useful when the mask is computed from scalar comparisons
    /// or logical combinations of SArrays.
    ///
    /// ```ignore
    /// let mask = sf.column("score").gt_scalar(500);
    /// let filtered = sf.logical_filter(mask)?;
    /// ```
    pub fn logical_filter(&self, mask: SArray) -> Result<SFrame> {
        let fused = self.fuse_plan()?;
        let filtered_plan = PlannerNode::logical_filter(fused, mask.plan().clone());

        let columns: Vec<SArray> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| SArray::from_plan(filtered_plan.clone(), c.dtype(), None, i))
            .collect();

        Ok(SFrame::new_with_columns(columns, self.column_names.clone()))
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

        let left = self.fuse_plan()?;
        let right = other.fuse_plan()?;
        let appended = PlannerNode::append(left, right);

        let columns: Vec<SArray> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| SArray::from_plan(appended.clone(), c.dtype(), None, i))
            .collect();

        Ok(SFrame::new_with_columns(columns, self.column_names.clone()))
    }

    /// Return the first n rows as a new SFrame.
    ///
    /// Only pulls enough batches from the stream to fill n rows, then stops.
    /// This avoids materializing the entire dataset for small head requests.
    pub fn head(&self, n: usize) -> Result<SFrame> {
        if n == 0 {
            let dtypes = self.column_types();
            let batch = SFrameRows::empty(&dtypes);
            return self.build_from_batch(batch);
        }
        // Try slicing for O(n) reads instead of streaming.
        if let Some(len) = self.known_len() {
            let end = (n as u64).min(len);
            if let Ok(sliced) = self.try_slice(0, end) {
                return Ok(sliced);
            }
        }
        let stream = self.compile_stream()?;
        let batch = materialize_head_sync(stream, n)?;
        self.build_from_batch(batch)
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
        let budget = sframe_config::global().sort_memory_budget;

        if estimated_size <= budget {
            self.sort_in_memory(&sort_keys)
        } else {
            crate::external_sort::external_sort(self, &sort_keys)
        }
    }

    /// Sort entirely in memory. Used when data fits within the sort memory budget.
    pub(crate) fn sort_in_memory(&self, sort_keys: &[SortKey]) -> Result<SFrame> {
        let stream = self.compile_stream()?;
        let (batch, indices) = sort::sort_indices(stream, sort_keys)?;

        if batch.num_rows() == 0 {
            return self.build_from_batch(batch);
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

        // Build output column names: all left cols + right cols (minus join keys)
        let mut names: Vec<String> = self.column_names.clone();
        let mut output_dtypes: Vec<FlexTypeEnum> = self.column_types();
        for (i, name) in other.column_names.iter().enumerate() {
            if !right_key_indices.contains(&i) {
                let out_name = if names.contains(name) {
                    format!("{name}.1")
                } else {
                    name.clone()
                };
                names.push(out_name);
                output_dtypes.push(other.columns[i].dtype());
            }
        }

        let mut join_stream = join::join(
            left_stream,
            right_stream,
            &JoinOn::multi(pairs),
            how,
        )?;

        // Consume the stream into a builder
        let mut builder = SFrameBuilder::anonymous(names.clone(), output_dtypes)?;
        while let Some(batch_result) = join_stream.next_batch() {
            let batch = batch_result?;
            builder.write_batch_chunked(&batch, DEFAULT_CHUNK_SIZE)?;
        }

        builder.finish()
    }

    /// Group by columns and aggregate.
    pub fn groupby(&self, key_names: &[&str], agg_specs: Vec<AggSpec>) -> Result<SFrame> {
        let key_indices: Vec<usize> = key_names
            .iter()
            .map(|name| self.column_index(name))
            .collect::<Result<_>>()?;

        let plan = self.fuse_plan()?;
        let input_types = self.column_types();

        // Build output column names: key columns + agg output names
        let mut out_names: Vec<String> = key_names.iter().map(|s| s.to_string()).collect();
        for spec in &agg_specs {
            out_names.push(spec.output_name.clone());
        }

        // Build output column types: key types + agg output types
        let mut out_types: Vec<FlexTypeEnum> = key_indices
            .iter()
            .map(|&i| input_types[i])
            .collect();
        for spec in &agg_specs {
            out_types.push(spec.aggregator.output_type(&[input_types[spec.column]]));
        }

        let cache_path =
            groupby::groupby(&plan, &key_indices, &agg_specs, &out_names, &out_types)?;

        // Build SFrame from the cache:// path produced by groupby
        let cache_fs = global_cache_fs();
        let vfs: Arc<dyn VirtualFileSystem> =
            Arc::new(ArcCacheFsVfs(cache_fs.clone()));
        let meta =
            sframe_storage::sframe_reader::SFrameMetadata::open_with_fs(&*vfs, &cache_path)?;
        let total_rows: u64 = if meta.group_index.columns.is_empty() {
            0
        } else {
            meta.group_index.columns[0]
                .segment_sizes
                .iter()
                .sum()
        };

        let store: Arc<dyn Send + Sync> = Arc::new(AnonymousStore {
            path: cache_path.clone(),
            cache_fs: cache_fs.clone(),
        });
        let plan = PlannerNode::sframe_source_cached(
            &cache_path,
            out_names.clone(),
            out_types.clone(),
            total_rows,
            store,
        );

        let columns: Vec<SArray> = out_types
            .iter()
            .enumerate()
            .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(total_rows), i))
            .collect();

        Ok(SFrame::new_with_columns(columns, out_names))
    }

    // === Phase 11.1: Column Mutation ===

    /// Replace a column with a new SArray (returns a new SFrame).
    pub fn replace_column(&self, name: &str, col: SArray) -> Result<SFrame> {
        let idx = self.column_index(name)?;

        if let Ok(fused) = self.fuse_plan() {
            let col_plan = PlannerNode::project(col.plan().clone(), vec![col.column_index()]);
            let existing_len = fused.length();
            let new_len = col_plan.length();
            let compatible = match (existing_len, new_len) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            };

            if compatible {
                let keep_indices: Vec<usize> =
                    (0..self.columns.len()).filter(|&i| i != idx).collect();

                if keep_indices.is_empty() {
                    // Only column being replaced
                    let columns = vec![SArray::from_plan(col_plan, col.dtype(), None, 0)];
                    return Ok(SFrame::new_with_columns(
                        columns,
                        self.column_names.clone(),
                    ));
                }

                let kept = PlannerNode::project(fused, keep_indices.clone());
                let unified = PlannerNode::column_union(vec![kept, col_plan]);

                let new_col_offset = keep_indices.len();
                let mut columns = Vec::with_capacity(self.columns.len());
                let mut kept_pos = 0;
                for i in 0..self.columns.len() {
                    if i == idx {
                        columns.push(SArray::from_plan(
                            unified.clone(),
                            col.dtype(),
                            None,
                            new_col_offset,
                        ));
                    } else {
                        columns.push(SArray::from_plan(
                            unified.clone(),
                            self.columns[i].dtype(),
                            None,
                            kept_pos,
                        ));
                        kept_pos += 1;
                    }
                }
                return Ok(SFrame::new_with_columns(
                    columns,
                    self.column_names.clone(),
                ));
            }
        }

        // Fallback: independent plans
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
                        "Column '{new_name}' already exists"
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
    ///   `"all"` drops rows where all specified columns are Undefined.
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
                for (col, col_vec) in col_vecs.iter_mut().enumerate().take(ncols) {
                    col_vec.push(batch.column(col).get(row).clone());
                }
            }
        }

        let dtypes = self.column_types();
        let result = SFrameRows::from_column_vecs(col_vecs, &dtypes)?;
        self.build_from_batch(result)
    }

    /// Fill missing values in a column.
    pub fn fillna(&self, column_name: &str, value: FlexType) -> Result<SFrame> {
        let idx = self.column_index(column_name)?;
        let filled = self.columns[idx].fillna(value);
        self.replace_column(column_name, filled)
    }

    // === Phase 11.3: Sampling & Splitting ===

    /// Build a boolean mask SArray for random sampling.
    ///
    /// Returns a lazy Integer SArray of 0/1 values where 1 indicates the row
    /// should be included (hash of seed + row index < threshold).
    fn make_sample_mask(&self, fraction: f64, seed: u64) -> Result<SArray> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n = self.num_rows()?;
        let threshold = (fraction * u64::MAX as f64) as u64;

        let indices = SArray::from_plan(
            PlannerNode::range(0, 1, n),
            FlexTypeEnum::Integer,
            Some(n),
            0,
        );
        Ok(indices.apply(
            Arc::new(move |v: &FlexType| {
                if let FlexType::Integer(i) = v {
                    let mut hasher = DefaultHasher::new();
                    (seed, *i as u64).hash(&mut hasher);
                    FlexType::Integer(if hasher.finish() < threshold { 1 } else { 0 })
                } else {
                    FlexType::Integer(0)
                }
            }),
            FlexTypeEnum::Integer,
        ))
    }

    /// Random sample of rows.
    ///
    /// Builds a lazy plan: `Range(0..n)` → seeded-hash transform → 0/1 mask
    /// → `logical_filter`. No data is materialized until the result is consumed.
    pub fn sample(&self, fraction: f64, seed: Option<u64>) -> Result<SFrame> {
        let mask = self.make_sample_mask(fraction, seed.unwrap_or(42))?;
        self.logical_filter(mask)
    }

    /// Random split into two SFrames.
    ///
    /// Returns `(train, test)` where `train` contains approximately `fraction`
    /// of the rows and `test` contains the rest. Both sides are lazy.
    pub fn random_split(&self, fraction: f64, seed: Option<u64>) -> Result<(SFrame, SFrame)> {
        let mask = self.make_sample_mask(fraction, seed.unwrap_or(42))?;

        let left = self.logical_filter(mask.clone())?;

        // Negate the mask for the other side
        let neg_mask = mask.apply(
            Arc::new(|v: &FlexType| match v {
                FlexType::Integer(i) => FlexType::Integer(if *i != 0 { 0 } else { 1 }),
                _ => FlexType::Integer(1),
            }),
            FlexTypeEnum::Integer,
        );
        let right = self.logical_filter(neg_mask)?;

        Ok((left, right))
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
                    let key_str = format!("{key}");
                    new_names.push(format!("{pfx}.{key_str}"));

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
                    new_names.push(format!("{pfx}.{pos}"));
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
                "Cannot unpack column '{column_name}' of type {col_dtype:?}; expected Dict or List"
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
                        for (c, col_vec) in col_vecs.iter_mut().enumerate().take(ncols) {
                            if c == col_idx {
                                col_vec.push(FlexType::Undefined);
                            } else {
                                col_vec.push(batch.column(c).get(row).clone());
                            }
                        }
                    } else {
                        for elem in elements {
                            for (c, col_vec) in col_vecs.iter_mut().enumerate().take(ncols) {
                                if c == col_idx {
                                    col_vec.push(elem.clone());
                                } else {
                                    col_vec.push(batch.column(c).get(row).clone());
                                }
                            }
                        }
                    }
                }

                let result = SFrameRows::from_column_vecs(col_vecs, &new_dtypes)?;
                write_to_cache(result, new_names)
            }
            _ => Err(SFrameError::Format(format!(
                "Cannot stack column '{column_name}' of type {col_dtype:?}; expected List or Vector"
            ))),
        }
    }

    // === Phase 11.5: Deduplication ===

    /// Remove duplicate rows.
    ///
    /// Implemented as a groupby on all columns with no aggregators.
    pub fn unique(&self) -> Result<SFrame> {
        let key_names: Vec<&str> = self.column_names.iter().map(|s| s.as_str()).collect();
        self.groupby(&key_names, vec![])
    }

    // === Phase 11.7: Tail ===

    /// Return the last n rows.
    ///
    /// Streams the plan but only retains the last n rows in memory.
    pub fn tail(&self, n: usize) -> Result<SFrame> {
        if n == 0 {
            let dtypes = self.column_types();
            let batch = SFrameRows::empty(&dtypes);
            return self.build_from_batch(batch);
        }
        // Try slicing for O(n) reads instead of streaming all rows.
        if let Some(len) = self.known_len() {
            let begin = len.saturating_sub(n as u64);
            if let Ok(sliced) = self.try_slice(begin, len) {
                return Ok(sliced);
            }
        }
        let stream = self.compile_stream()?;
        let batch = materialize_tail_sync(stream, n)?;
        self.build_from_batch(batch)
    }

    /// Lazily slice this SFrame to rows `[begin, end)`.
    ///
    /// All column plans must be fully linear (no Filter/LogicalFilter/Reduce).
    /// Returns an error for non-sliceable plans or out-of-bounds ranges.
    pub fn try_slice(&self, begin: u64, end: u64) -> Result<SFrame> {
        let new_columns: Vec<SArray> = self
            .columns
            .iter()
            .map(|col| col.try_slice(begin, end))
            .collect::<Result<Vec<_>>>()?;
        Ok(SFrame {
            columns: new_columns,
            column_names: self.column_names.clone(),
            metadata: self.metadata.clone(),
        })
    }

    /// Slice this SFrame to rows `[begin, end)`.
    ///
    /// Tries lazy plan rewriting first. If the plan is not sliceable,
    /// falls back to materializing.
    pub fn slice(&self, begin: u64, end: u64) -> Result<SFrame> {
        if let Ok(sf) = self.try_slice(begin, end) {
            return Ok(sf);
        }
        // Fallback: materialize the needed range via streaming.
        let len = self.num_rows()?;
        if begin > end || end > len {
            return Err(SFrameError::Format(format!(
                "Slice [{begin}, {end}) out of range for length {len}"
            )));
        }
        // Stream head(end), then take the tail portion.
        let stream = self.compile_stream()?;
        let batch = materialize_head_sync(stream, end as usize)?;
        let indices: Vec<usize> = (begin as usize..end as usize).collect();
        let sliced = batch.take(&indices)?;
        self.build_from_batch(sliced)
    }

    /// Materialize all lazy computations.
    pub fn materialize(&self) -> Result<SFrame> {
        let batch = self.materialize_batch()?;
        self.build_from_batch(batch)
    }

    /// Save to disk as an SFrame directory.
    ///
    /// When the query plan is parallel-sliceable, writes one segment per
    /// rayon thread in parallel. Otherwise falls back to sequential streaming.
    pub fn save(&self, path: &str) -> Result<()> {
        use sframe_query::execute::parallel_slice_row_count;

        let col_names: Vec<&str> = self.column_names.iter().map(|s| s.as_str()).collect();
        let dtypes = self.column_types();

        let plan = self.fuse_plan()?;

        if let Some(total_rows) = parallel_slice_row_count(&plan) {
            self.save_parallel(path, &plan, &col_names, &dtypes, total_rows)
        } else {
            self.save_sequential(path, &col_names, &dtypes)
        }
    }

    fn save_sequential(
        &self,
        path: &str,
        col_names: &[&str],
        dtypes: &[FlexTypeEnum],
    ) -> Result<()> {
        let mut writer = SFrameStreamWriter::new(path, col_names, dtypes)?;
        for (key, value) in &self.metadata {
            writer.set_metadata(key, value);
        }
        let stream = self.compile_stream()?;
        for_each_batch_sync(stream, |batch| writer.write_batch(&batch))?;
        writer.finish()
    }

    fn save_parallel(
        &self,
        path: &str,
        plan: &Arc<PlannerNode>,
        col_names: &[&str],
        dtypes: &[FlexTypeEnum],
        total_rows: u64,
    ) -> Result<()> {
        use rayon::prelude::*;
        use sframe_query::execute::consume_to_segment;
        use sframe_query::planner::clone_plan_with_row_range;
        use sframe_storage::segment_writer::BufferedSegmentWriter;
        use sframe_storage::sframe_writer::{
            assemble_sframe_from_segments, generate_hash, segment_filename,
        };

        let data_prefix = format!("m_{}", generate_hash(path));
        let n_workers = rayon::current_num_threads().max(1);

        std::fs::create_dir_all(path).map_err(SFrameError::Io)?;

        // Build per-worker plans with row ranges
        let worker_plans: Vec<(usize, Arc<PlannerNode>)> = (0..n_workers)
            .filter_map(|i| {
                let begin = (i as u64 * total_rows) / n_workers as u64;
                let end = ((i as u64 + 1) * total_rows) / n_workers as u64;
                if begin >= end {
                    return None;
                }
                Some((i, clone_plan_with_row_range(plan, begin, end)))
            })
            .collect();

        // Each worker writes its own segment file
        let results: Vec<Result<(String, Vec<u64>, u64)>> = worker_plans
            .into_par_iter()
            .map(|(i, worker_plan)| {
                let seg_file = segment_filename(&data_prefix, i);
                let seg_path = format!("{path}/{seg_file}");
                let file = std::fs::File::create(&seg_path).map_err(SFrameError::Io)?;
                let buf_writer = std::io::BufWriter::new(file);
                let seg_writer = BufferedSegmentWriter::new(buf_writer, dtypes);

                let mut iter = compile(&worker_plan)?;
                let (sizes, rows) = consume_to_segment(&mut iter, seg_writer, dtypes)?;
                Ok((seg_file, sizes, rows))
            })
            .collect();

        // Collect results
        let mut segment_files = Vec::new();
        let mut all_segment_sizes = Vec::new();
        let mut total_written = 0u64;
        for result in results {
            let (seg_file, sizes, rows) = result?;
            segment_files.push(seg_file);
            all_segment_sizes.push(sizes);
            total_written += rows;
        }

        // Assemble metadata
        let vfs = sframe_io::local_fs::LocalFileSystem;
        assemble_sframe_from_segments(
            &vfs,
            path,
            col_names,
            dtypes,
            &segment_files,
            &all_segment_sizes,
            total_written,
            &self.metadata,
        )?;

        Ok(())
    }

    /// Write to CSV file.
    ///
    /// Streams batches to disk — does not materialize the full frame in memory.
    pub fn to_csv(&self, path: &str, options: Option<CsvWriterOptions>) -> Result<()> {
        let opts = options.unwrap_or_default();
        let file = std::fs::File::create(path).map_err(SFrameError::Io)?;
        let mut writer = std::io::BufWriter::new(file);

        if opts.header {
            csv_writer::write_csv_header(&mut writer, &self.column_names, &opts)?;
        }

        let stream = self.compile_stream()?;
        for_each_batch_sync(stream, |batch| {
            csv_writer::write_csv_batch(&mut writer, &batch, &opts)
        })?;

        std::io::Write::flush(&mut writer).map_err(SFrameError::Io)
    }

    /// Write to JSON Lines file.
    pub fn to_json(&self, path: &str) -> Result<()> {
        let batch = self.materialize_batch()?;
        json_io::write_json_file(path, &batch, &self.column_names)
    }

    /// Write to a single Parquet file.
    ///
    /// Streams batches to disk — does not materialize the full frame in memory.
    pub fn to_parquet(&self, path: &str) -> Result<()> {
        let stream = self.compile_stream()?;
        let names = self.column_names.clone();
        let types = self.column_types();
        sframe_parquet::parquet_writer::write_parquet(
            stream,
            &names,
            &types,
            std::path::Path::new(path),
        )
    }

    /// Write to sharded Parquet files with parallel execution.
    ///
    /// Output files are named `{prefix}_{i}_of_{N}.parquet`.
    ///
    /// If the query plan is parallel-sliceable (no Reduce, Append, etc.),
    /// each shard is written in parallel using rayon. Otherwise, falls back
    /// to writing a single shard sequentially.
    pub fn to_parquet_sharded(&self, prefix: &str) -> Result<()> {
        use rayon::prelude::*;
        use sframe_query::execute::parallel_slice_row_count;
        use sframe_query::planner::clone_plan_with_row_range;

        let plan = self.fuse_plan()?;
        let names = self.column_names.clone();
        let types = self.column_types();

        // Check if parallel slicing is possible
        if let Some(total_rows) = parallel_slice_row_count(&plan) {
            let n_workers = rayon::current_num_threads().max(1);

            // Build per-worker plans with row ranges
            let worker_plans: Vec<(usize, Arc<PlannerNode>)> = (0..n_workers)
                .filter_map(|i| {
                    let begin = (i as u64 * total_rows) / n_workers as u64;
                    let end = ((i as u64 + 1) * total_rows) / n_workers as u64;
                    if begin >= end {
                        return None;
                    }
                    Some((i, clone_plan_with_row_range(&plan, begin, end)))
                })
                .collect();

            let actual_shards = worker_plans.len();

            // Execute in parallel
            let results: Vec<Result<()>> = worker_plans
                .into_par_iter()
                .map(|(i, worker_plan)| {
                    let iter = compile(&worker_plan)?;
                    sframe_parquet::parquet_writer::write_parquet_shard(
                        iter, &names, &types, prefix, i, actual_shards,
                    )
                })
                .collect();

            // Propagate any worker error
            for result in results {
                result?;
            }

            Ok(())
        } else {
            // Fallback: single-threaded sequential write to one shard
            let stream = self.compile_stream()?;
            sframe_parquet::parquet_writer::write_parquet_shard(
                stream, &names, &types, prefix, 0, 1,
            )
        }
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

    /// Try to produce a single unified plan for all columns.
    ///
    /// Output columns are always 0..N matching the SFrame's column order.
    /// Returns an error only if columns have truly incompatible plans (e.g.
    /// different source tables with different row counts).
    fn fuse_plan(&self) -> Result<Arc<PlannerNode>> {
        if self.columns.is_empty() {
            return Ok(PlannerNode::materialized(SFrameRows::empty(&[])));
        }

        // Fast path: all columns share the same plan Arc — just project.
        if let Some(plan) = self.shared_plan() {
            let indices: Vec<usize> =
                self.columns.iter().map(|c| c.column_index()).collect();
            return Ok(PlannerNode::project(plan.clone(), indices));
        }

        // Group columns by plan Arc, preserving original order.
        use std::collections::HashMap;
        let mut plan_to_group: HashMap<usize, usize> = HashMap::new();
        let mut groups: Vec<(Arc<PlannerNode>, Vec<usize>)> = Vec::new();
        // For each SFrame column, record (group_idx, position_within_group)
        let mut col_positions: Vec<(usize, usize)> = Vec::with_capacity(self.columns.len());

        for col in &self.columns {
            let ptr = Arc::as_ptr(col.plan()) as usize;
            let group_idx = if let Some(&gi) = plan_to_group.get(&ptr) {
                gi
            } else {
                let gi = groups.len();
                plan_to_group.insert(ptr, gi);
                groups.push((col.plan().clone(), Vec::new()));
                gi
            };
            let pos = groups[group_idx].1.len();
            groups[group_idx].1.push(col.column_index());
            col_positions.push((group_idx, pos));
        }

        // Build Project for each group
        let mut union_inputs: Vec<Arc<PlannerNode>> = Vec::new();
        let mut group_col_offsets: Vec<usize> = Vec::new();
        let mut offset = 0;
        for (plan, indices) in &groups {
            group_col_offsets.push(offset);
            offset += indices.len();
            union_inputs.push(PlannerNode::project(plan.clone(), indices.clone()));
        }

        let fused = if union_inputs.len() == 1 {
            union_inputs.remove(0)
        } else {
            // Verify length compatibility
            let known_lengths: Vec<u64> = union_inputs.iter()
                .filter_map(|p| p.length())
                .collect();
            if known_lengths.len() >= 2 && !known_lengths.windows(2).all(|w| w[0] == w[1]) {
                return Err(SFrameError::Format("Cannot fuse columns with different row counts".to_string()));
            }
            PlannerNode::column_union(union_inputs)
        };

        // Build final column order: map each original column to its position
        let final_indices: Vec<usize> = col_positions.iter()
            .map(|&(gi, pos)| group_col_offsets[gi] + pos)
            .collect();

        // If final_indices is already 0..N, skip the extra Project
        let is_identity = final_indices.iter().enumerate().all(|(i, &c)| c == i);
        if is_identity {
            Ok(fused)
        } else {
            Ok(PlannerNode::project(fused, final_indices))
        }
    }

    /// Compile the SFrame's plan into a BatchStream.
    pub(crate) fn compile_stream(&self) -> Result<sframe_query::execute::BatchStream> {
        compile(&self.fuse_plan()?)
    }

    fn column_index(&self, name: &str) -> Result<usize> {
        self.column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| SFrameError::Format(format!("Column '{name}' not found")))
    }

    fn materialize_batch(&self) -> Result<SFrameRows> {
        materialize_sync(self.compile_stream()?)
    }

    fn build_from_batch(&self, batch: SFrameRows) -> Result<SFrame> {
        write_to_cache(batch, self.column_names.clone())
    }
}

impl std::fmt::Display for SFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_display = 10;

        // Try slicing first to avoid reading the entire dataset.
        // Falls back to compile_stream + materialize_head if slicing isn't possible.
        let batch = match self.try_slice(0, (max_display as u64).min(self.known_len().unwrap_or(max_display as u64))) {
            Ok(sliced) => match sliced.compile_stream().and_then(materialize_sync) {
                Ok(b) => b,
                Err(e) => return write!(f, "[SFrame error: {e}]"),
            },
            Err(_) => match self
                .compile_stream()
                .and_then(|s| materialize_head_sync(s, max_display))
            {
                Ok(b) => b,
                Err(e) => return write!(f, "[SFrame error: {e}]"),
            },
        };

        // Get total row count cheaply (no materialization).
        let known_len: Option<u64> = self.known_len();
        let ncols = self.column_names.len();

        // Determine column widths
        let display_rows = batch.num_rows();

        let mut col_widths: Vec<usize> = self
            .column_names
            .iter()
            .map(|n| n.len())
            .collect();

        let mut cell_strings: Vec<Vec<String>> = Vec::new();
        for row_idx in 0..display_rows {
            let mut row_strs = Vec::new();
            for (col_idx, col_width) in col_widths.iter_mut().enumerate().take(ncols) {
                let val = batch.column(col_idx).get(row_idx);
                let s = format!("{val}");
                let truncated = if s.len() > 30 {
                    format!("{}...", &s[..27])
                } else {
                    s
                };
                if truncated.len() > *col_width {
                    *col_width = truncated.len();
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
        writeln!(f, "{sep}")?;
        let header: String = self
            .column_names
            .iter()
            .zip(col_widths.iter())
            .map(|(name, &w)| format!("| {name:w$} "))
            .collect::<Vec<_>>()
            .join("")
            + "|";
        writeln!(f, "{header}")?;
        writeln!(f, "{sep}")?;

        // Data rows
        for row_strs in &cell_strings {
            let row: String = row_strs
                .iter()
                .zip(col_widths.iter())
                .map(|(s, &w)| format!("| {s:w$} "))
                .collect::<Vec<_>>()
                .join("")
                + "|";
            writeln!(f, "{row}")?;
        }

        // Show "..." if there are (or might be) more rows than displayed.
        let has_more = match known_len {
            Some(n) => n as usize > display_rows,
            None => display_rows >= max_display,
        };
        if has_more {
            let dots: String = col_widths
                .iter()
                .map(|&w| format!("| {:width$} ", "...", width = w))
                .collect::<Vec<_>>()
                .join("")
                + "|";
            writeln!(f, "{dots}")?;
        }

        writeln!(f, "{sep}")?;
        let len_str = match known_len {
            Some(n) => n.to_string(),
            None if !has_more => display_rows.to_string(),
            None => "?".to_string(),
        };
        write!(f, "[{len_str} rows x {ncols} columns]")
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
        format!("{manifest}/../../samples")
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

        let s = format!("{sf}");
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
            .map(FlexType::Integer)
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

    #[test]
    fn test_join_correctness_all_types() {
        let n = 500i64;
        let left = SFrame::from_columns(vec![
            ("id", SArray::from_vec(
                (0..n).map(FlexType::Integer).collect(),
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("name", SArray::from_vec(
                (0..n).map(|i| FlexType::String(format!("left_{i}").into())).collect(),
                FlexTypeEnum::String,
            ).unwrap()),
        ]).unwrap();

        let right = SFrame::from_columns(vec![
            ("id", SArray::from_vec(
                (0..n).step_by(2).map(FlexType::Integer).collect(),
                FlexTypeEnum::Integer,
            ).unwrap()),
            ("score", SArray::from_vec(
                (0..n).step_by(2).map(|i| FlexType::Float(i as f64 * 1.5)).collect(),
                FlexTypeEnum::Float,
            ).unwrap()),
        ]).unwrap();

        // Inner join: should have n/2 rows (only even ids match)
        let inner = left.join(&right, "id", "id", JoinType::Inner).unwrap();
        assert_eq!(inner.num_rows().unwrap(), (n / 2) as u64);
        assert_eq!(inner.num_columns(), 3); // id, name, score

        // Left join: all left rows present
        let left_join = left.join(&right, "id", "id", JoinType::Left).unwrap();
        assert_eq!(left_join.num_rows().unwrap(), n as u64);
    }

    #[test]
    fn test_unique_uses_proper_hashing() {
        // Regression: ensure unique() uses FlexType Hash+Eq, not Debug string.
        // Floats with different representations must be treated correctly.
        let sf = SFrame::from_columns(vec![
            ("a", SArray::from_vec(
                vec![FlexType::Float(1.0), FlexType::Float(1.0), FlexType::Float(2.0)],
                FlexTypeEnum::Float,
            ).unwrap()),
            ("b", SArray::from_vec(
                vec![FlexType::Integer(10), FlexType::Integer(10), FlexType::Integer(20)],
                FlexTypeEnum::Integer,
            ).unwrap()),
        ]).unwrap();
        let result = sf.unique().unwrap();
        assert_eq!(result.num_rows().unwrap(), 2); // (1.0, 10) deduped, (2.0, 20) kept
    }

    // === Slice tests ===

    #[test]
    fn test_sframe_slice() {
        let vals: Vec<FlexType> = (0..100).map(FlexType::Integer).collect();
        let sa = SArray::from_vec(vals, FlexTypeEnum::Integer).unwrap();
        let sf = SFrame::from_columns(vec![("x", sa)]).unwrap();
        let sliced = sf.slice(10, 20).unwrap();
        assert_eq!(sliced.num_rows().unwrap(), 10);
        let col = sliced.column("x").unwrap();
        let result = col.to_vec().unwrap();
        let expected: Vec<FlexType> = (10..20).map(FlexType::Integer).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sframe_slice_multi_column() {
        let ints: Vec<FlexType> = (0..50).map(FlexType::Integer).collect();
        let floats: Vec<FlexType> = (0..50).map(|i| FlexType::Float(i as f64 * 0.5)).collect();
        let sf = SFrame::from_columns(vec![
            ("i", SArray::from_vec(ints, FlexTypeEnum::Integer).unwrap()),
            ("f", SArray::from_vec(floats, FlexTypeEnum::Float).unwrap()),
        ]).unwrap();
        let sliced = sf.slice(5, 10).unwrap();
        assert_eq!(sliced.num_rows().unwrap(), 5);
        assert_eq!(sliced.num_columns(), 2);
    }

    #[test]
    fn test_sframe_slice_out_of_bounds() {
        let vals: Vec<FlexType> = (0..10).map(FlexType::Integer).collect();
        let sf = SFrame::from_columns(vec![
            ("x", SArray::from_vec(vals, FlexTypeEnum::Integer).unwrap()),
        ]).unwrap();
        assert!(sf.slice(0, 11).is_err());
    }

    #[test]
    fn test_sframe_slice_preserves_laziness() {
        let vals: Vec<FlexType> = (0..100).map(FlexType::Integer).collect();
        let sa = SArray::from_vec(vals, FlexTypeEnum::Integer).unwrap();
        let sf = SFrame::from_columns(vec![("x", sa)]).unwrap();
        let added = sf.add_column(
            "y",
            sf.column("x").unwrap().add_scalar(FlexType::Integer(1)),
        ).unwrap();
        let sliced = added.slice(90, 100).unwrap();
        assert_eq!(sliced.num_rows().unwrap(), 10);
        let x = sliced.column("x").unwrap().to_vec().unwrap();
        let expected: Vec<FlexType> = (90..100).map(FlexType::Integer).collect();
        assert_eq!(x, expected);
    }

    // === ColumnUnion end-to-end tests ===

    #[test]
    fn test_add_column_then_filter() {
        // Scenario: add a derived column, then filter.
        // After add_column, fuse_plan should produce a ColumnUnion.
        // After filter, the result should still be lazily evaluated.
        let sf = make_test_sf(); // columns: id (Int), name (String), score (Float)
        let score = sf.column("score").unwrap();
        let doubled = score.apply(
            Arc::new(|v: &FlexType| match v {
                FlexType::Float(f) => FlexType::Float(f * 2.0),
                other => other.clone(),
            }),
            FlexTypeEnum::Float,
        );
        let sf2 = sf.add_column("score2", doubled).unwrap();

        // fuse_plan should succeed (tested via explain not saying "materialized independently")
        let explanation = sf2.explain();
        assert!(
            !explanation.contains("materialized independently"),
            "fuse_plan should succeed after add_column, but got:\n{explanation}",
        );

        // Filter on score > 75.0
        let filtered = sf2
            .filter(
                "score",
                Arc::new(|v: &FlexType| matches!(v, FlexType::Float(f) if *f > 75.0)),
            )
            .unwrap();

        // Verify data correctness
        let score_vals = filtered.column("score").unwrap().to_vec().unwrap();
        let score2_vals = filtered.column("score2").unwrap().to_vec().unwrap();
        assert!(!score_vals.is_empty(), "should have some filtered rows");
        assert_eq!(score_vals.len(), score2_vals.len());
        for (s, s2) in score_vals.iter().zip(score2_vals.iter()) {
            if let (FlexType::Float(a), FlexType::Float(b)) = (s, s2) {
                assert!(
                    (b - a * 2.0).abs() < 1e-10,
                    "score2 should be 2x score, got score={a} score2={b}",
                );
                assert!(*a > 75.0, "filter should have removed scores <= 75, got {a}");
            }
        }
    }

    #[test]
    fn test_filter_then_add_column() {
        // Scenario: filter first, then add a derived column.
        let sf = make_test_sf();

        let filtered = sf
            .filter(
                "score",
                Arc::new(|v: &FlexType| matches!(v, FlexType::Float(f) if *f > 75.0)),
            )
            .unwrap();

        let score = filtered.column("score").unwrap();
        let doubled = score.apply(
            Arc::new(|v: &FlexType| match v {
                FlexType::Float(f) => FlexType::Float(f * 2.0),
                other => other.clone(),
            }),
            FlexTypeEnum::Float,
        );
        let result = filtered.add_column("score2", doubled).unwrap();

        // fuse_plan should succeed (ColumnUnion-based)
        let explanation = result.explain();
        assert!(
            !explanation.contains("materialized independently"),
            "fuse_plan should succeed after filter + add_column, but got:\n{explanation}",
        );

        // Verify ColumnUnion appears in the explanation
        assert!(
            explanation.contains("ColumnUnion"),
            "Expected ColumnUnion in explanation:\n{explanation}",
        );

        // Verify data correctness
        let score_vals = result.column("score").unwrap().to_vec().unwrap();
        let score2_vals = result.column("score2").unwrap().to_vec().unwrap();
        assert!(!score_vals.is_empty(), "should have some filtered rows");
        assert_eq!(score_vals.len(), score2_vals.len());
        for (s, s2) in score_vals.iter().zip(score2_vals.iter()) {
            if let (FlexType::Float(a), FlexType::Float(b)) = (s, s2) {
                assert!(
                    (b - a * 2.0).abs() < 1e-10,
                    "score2 should be 2x score, got score={a} score2={b}",
                );
                assert!(*a > 75.0, "filter should have removed scores <= 75, got {a}");
            }
        }
    }

    #[test]
    fn test_replace_column_then_filter() {
        let sf = make_test_sf();
        let score = sf.column("score").unwrap();
        let negated = score.apply(
            Arc::new(|v: &FlexType| match v {
                FlexType::Float(f) => FlexType::Float(-f),
                other => other.clone(),
            }),
            FlexTypeEnum::Float,
        );
        let sf2 = sf.replace_column("score", negated).unwrap();

        // fuse_plan should succeed
        let explanation = sf2.explain();
        assert!(
            !explanation.contains("materialized independently"),
            "fuse_plan should succeed after replace_column, but got:\n{explanation}",
        );

        // Verify the replaced column has negated values
        let vals = sf2.column("score").unwrap().to_vec().unwrap();
        for v in &vals {
            if let FlexType::Float(f) = v {
                assert!(*f <= 0.0, "scores should be negated: {f}");
            }
        }

        // Now filter on the negated score (score < -75 means original > 75)
        let filtered = sf2
            .filter(
                "score",
                Arc::new(|v: &FlexType| matches!(v, FlexType::Float(f) if *f < -75.0)),
            )
            .unwrap();

        let filtered_scores = filtered.column("score").unwrap().to_vec().unwrap();
        assert!(!filtered_scores.is_empty(), "should have some filtered rows");
        for v in &filtered_scores {
            if let FlexType::Float(f) = v {
                assert!(*f < -75.0, "filter should keep only scores < -75, got {f}");
            }
        }

        // Verify other columns survived the pipeline
        let ids = filtered.column("id").unwrap().to_vec().unwrap();
        assert_eq!(ids.len(), filtered_scores.len());
    }

    // ===================== Parquet integration tests =====================

    /// Helper: write a test Parquet file with (id: i64, name: string) columns.
    fn write_test_parquet(path: &std::path::Path, ids: &[i64], names: &[&str]) {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::basic::Compression;
        use parquet::file::properties::WriterProperties;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let id_array = Int64Array::from(ids.to_vec());
        let name_array = StringArray::from(names.to_vec());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(name_array)],
        )
        .unwrap();

        let file = std::fs::File::create(path).unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn test_from_parquet_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(&path, &[10, 20, 30], &["alpha", "beta", "gamma"]);

        let sf = SFrame::from_parquet(path.to_str().unwrap()).unwrap();
        assert_eq!(sf.num_rows().unwrap(), 3);
        assert_eq!(sf.num_columns(), 2);
        assert_eq!(sf.column_names(), &["id", "name"]);
        assert_eq!(
            sf.column_types(),
            vec![FlexTypeEnum::Integer, FlexTypeEnum::String]
        );

        let rows = sf.iter_rows().unwrap();
        assert_eq!(rows[0][0], FlexType::Integer(10));
        assert_eq!(rows[1][0], FlexType::Integer(20));
        assert_eq!(rows[2][0], FlexType::Integer(30));
        assert_eq!(rows[0][1], FlexType::String("alpha".into()));
        assert_eq!(rows[1][1], FlexType::String("beta".into()));
        assert_eq!(rows[2][1], FlexType::String("gamma".into()));
    }

    #[test]
    fn test_from_parquet_glob_pattern() {
        let dir = tempfile::tempdir().unwrap();
        write_test_parquet(&dir.path().join("part0.parquet"), &[1, 2], &["a", "b"]);
        write_test_parquet(&dir.path().join("part1.parquet"), &[3, 4], &["c", "d"]);

        let pattern = format!("{}/*.parquet", dir.path().to_str().unwrap());
        let sf = SFrame::from_parquet(&pattern).unwrap();
        assert_eq!(sf.num_rows().unwrap(), 4);
        assert_eq!(sf.num_columns(), 2);

        let rows = sf.iter_rows().unwrap();
        let ids: Vec<i64> = rows
            .iter()
            .map(|r| match &r[0] {
                FlexType::Integer(v) => *v,
                _ => panic!("expected integer"),
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_from_parquet_files_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("a.parquet");
        let p2 = dir.path().join("b.parquet");
        write_test_parquet(&p1, &[100], &["x"]);
        write_test_parquet(&p2, &[200], &["y"]);

        let sf = SFrame::from_parquet_files(&[
            p1.to_str().unwrap(),
            p2.to_str().unwrap(),
        ])
        .unwrap();
        assert_eq!(sf.num_rows().unwrap(), 2);
        assert_eq!(sf.num_columns(), 2);

        let rows = sf.iter_rows().unwrap();
        assert_eq!(rows[0][0], FlexType::Integer(100));
        assert_eq!(rows[1][0], FlexType::Integer(200));
    }

    #[test]
    fn test_to_parquet_roundtrip() {
        let sf = SFrame::from_columns(vec![
            (
                "x",
                SArray::from_vec(
                    vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)],
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "y",
                SArray::from_vec(
                    vec![
                        FlexType::String("a".into()),
                        FlexType::String("b".into()),
                        FlexType::String("c".into()),
                    ],
                    FlexTypeEnum::String,
                )
                .unwrap(),
            ),
        ])
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output.parquet");
        sf.to_parquet(path.to_str().unwrap()).unwrap();

        // Read back
        let sf2 = SFrame::from_parquet(path.to_str().unwrap()).unwrap();
        assert_eq!(sf2.num_rows().unwrap(), 3);
        assert_eq!(sf2.num_columns(), 2);
        assert_eq!(sf2.column_names(), &["x", "y"]);

        let rows = sf2.iter_rows().unwrap();
        assert_eq!(rows[0][0], FlexType::Integer(1));
        assert_eq!(rows[1][0], FlexType::Integer(2));
        assert_eq!(rows[2][0], FlexType::Integer(3));
        assert_eq!(rows[0][1], FlexType::String("a".into()));
        assert_eq!(rows[1][1], FlexType::String("b".into()));
        assert_eq!(rows[2][1], FlexType::String("c".into()));
    }

    #[test]
    fn test_to_parquet_sharded_roundtrip() {
        // Use a materialized SFrame (from_columns creates MaterializedSource),
        // which is not parallel-sliceable, so this tests the fallback path.
        let sf = SFrame::from_columns(vec![(
            "val",
            SArray::from_vec(
                vec![
                    FlexType::Integer(10),
                    FlexType::Integer(20),
                    FlexType::Integer(30),
                ],
                FlexTypeEnum::Integer,
            )
            .unwrap(),
        )])
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("shard").to_str().unwrap().to_string();
        sf.to_parquet_sharded(&prefix).unwrap();

        // Since from_columns is not sliceable, should produce a single shard
        let shard_path = format!("{prefix}_0_of_1.parquet");
        assert!(
            std::path::Path::new(&shard_path).exists(),
            "Expected shard file at {shard_path}"
        );

        // Read back via glob
        let pattern = format!("{prefix}_*.parquet");
        let sf2 = SFrame::from_parquet(&pattern).unwrap();
        assert_eq!(sf2.num_rows().unwrap(), 3);

        let rows = sf2.iter_rows().unwrap();
        assert_eq!(rows[0][0], FlexType::Integer(10));
        assert_eq!(rows[1][0], FlexType::Integer(20));
        assert_eq!(rows[2][0], FlexType::Integer(30));
    }

    #[test]
    fn test_to_parquet_sharded_parallel_path() {
        // Create an SFrame from an on-disk SFrame source, which IS
        // parallel-sliceable.
        let dir = tempfile::tempdir().unwrap();

        // First, build an SFrame on disk
        let sf = SFrame::from_columns(vec![
            (
                "id",
                SArray::from_vec(
                    (0..100).map(FlexType::Integer).collect(),
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "val",
                SArray::from_vec(
                    (0..100).map(|i| FlexType::Float(i as f64 * 0.5)).collect(),
                    FlexTypeEnum::Float,
                )
                .unwrap(),
            ),
        ])
        .unwrap();

        let sf_path = dir.path().join("source.sf");
        sf.save(sf_path.to_str().unwrap()).unwrap();

        // Read it back — now it has an SFrameSource plan (parallel-sliceable)
        let sf_disk = SFrame::read(sf_path.to_str().unwrap()).unwrap();

        let prefix = dir.path().join("out").to_str().unwrap().to_string();
        sf_disk.to_parquet_sharded(&prefix).unwrap();

        // Should produce multiple shards (at least 1)
        let pattern = format!("{prefix}_*.parquet");
        let shard_files = sframe_parquet::parquet_reader::resolve_parquet_paths(&pattern).unwrap();
        assert!(
            !shard_files.is_empty(),
            "Expected at least one shard file"
        );

        // Read all shards and verify total row count
        let sf_back = SFrame::from_parquet(&pattern).unwrap();
        assert_eq!(sf_back.num_rows().unwrap(), 100);

        // Verify all 100 ids are present (order may vary due to shard
        // filename lexicographic sorting not matching shard index order
        // when there are >= 10 shards).
        let rows = sf_back.iter_rows().unwrap();
        let mut ids: Vec<i64> = rows
            .iter()
            .map(|r| match &r[0] {
                FlexType::Integer(v) => *v,
                _ => panic!("expected integer"),
            })
            .collect();
        ids.sort();
        let expected: Vec<i64> = (0..100).collect();
        assert_eq!(ids, expected);

        // Verify id-value pairs are consistent
        for row in &rows {
            if let (FlexType::Integer(id), FlexType::Float(val)) = (&row[0], &row[1]) {
                assert_eq!(*val, *id as f64 * 0.5, "id={id} val={val}");
            }
        }
    }

    #[test]
    fn test_parquet_roundtrip_with_floats_and_nulls() {
        let sf = SFrame::from_columns(vec![
            (
                "i",
                SArray::from_vec(
                    vec![FlexType::Integer(1), FlexType::Undefined, FlexType::Integer(3)],
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "f",
                SArray::from_vec(
                    vec![FlexType::Float(1.5), FlexType::Float(2.5), FlexType::Undefined],
                    FlexTypeEnum::Float,
                )
                .unwrap(),
            ),
            (
                "s",
                SArray::from_vec(
                    vec![
                        FlexType::Undefined,
                        FlexType::String("hello".into()),
                        FlexType::String("world".into()),
                    ],
                    FlexTypeEnum::String,
                )
                .unwrap(),
            ),
        ])
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.parquet");
        sf.to_parquet(path.to_str().unwrap()).unwrap();

        let sf2 = SFrame::from_parquet(path.to_str().unwrap()).unwrap();
        assert_eq!(sf2.num_rows().unwrap(), 3);
        assert_eq!(sf2.num_columns(), 3);

        let rows = sf2.iter_rows().unwrap();
        // Row 0: (1, 1.5, Undefined)
        assert_eq!(rows[0][0], FlexType::Integer(1));
        assert_eq!(rows[0][1], FlexType::Float(1.5));
        assert_eq!(rows[0][2], FlexType::Undefined);
        // Row 1: (Undefined, 2.5, "hello")
        assert_eq!(rows[1][0], FlexType::Undefined);
        assert_eq!(rows[1][1], FlexType::Float(2.5));
        assert_eq!(rows[1][2], FlexType::String("hello".into()));
        // Row 2: (3, Undefined, "world")
        assert_eq!(rows[2][0], FlexType::Integer(3));
        assert_eq!(rows[2][1], FlexType::Undefined);
        assert_eq!(rows[2][2], FlexType::String("world".into()));
    }

    #[test]
    fn test_csv_to_parquet_roundtrip() {
        // Read business.csv
        let csv_path = format!("{}/business.csv", samples_dir());
        let sf = SFrame::from_csv(&csv_path, None).unwrap();
        let original_rows = sf.num_rows().unwrap();
        let original_names = sf.column_names().to_vec();

        // Write to Parquet
        let dir = tempfile::tempdir().unwrap();
        let parquet_path = dir.path().join("business.parquet");
        sf.to_parquet(parquet_path.to_str().unwrap()).unwrap();

        // Read back and compare
        let sf2 = SFrame::from_parquet(parquet_path.to_str().unwrap()).unwrap();
        assert_eq!(sf2.num_rows().unwrap(), original_rows);
        assert_eq!(sf2.column_names(), &original_names);

        // Spot-check first few rows
        let head1 = sf.head(5).unwrap();
        let head2 = sf2.head(5).unwrap();
        assert_eq!(head1.num_rows().unwrap(), head2.num_rows().unwrap());
    }

    #[test]
    fn test_parquet_sharded_roundtrip_with_real_data() {
        let csv_path = format!("{}/business.csv", samples_dir());
        let sf = SFrame::from_csv(&csv_path, None).unwrap();
        let original_rows = sf.num_rows().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("sharded").to_str().unwrap().to_string();
        sf.to_parquet_sharded(&prefix).unwrap();

        // Read back via glob
        let pattern = format!("{prefix}_*.parquet");
        let sf2 = SFrame::from_parquet(&pattern).unwrap();
        assert_eq!(sf2.num_rows().unwrap(), original_rows);
    }
}
