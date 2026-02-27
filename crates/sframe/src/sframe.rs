//! SFrame — user-facing dataframe API with lazy evaluation.
//!
//! Operations build PlannerNode DAGs. Materialization happens on
//! `.head()`, `.iter_rows()`, `.save()`, `.materialize()`, or `Display`.

use std::sync::Arc;

use sframe_query::algorithms::aggregators::AggSpec;
use sframe_query::algorithms::csv_parser::{read_csv, CsvOptions};
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

/// A columnar dataframe with lazy evaluation.
pub struct SFrame {
    columns: Vec<SArray>,
    column_names: Vec<String>,
}

impl SFrame {
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

        Ok(SFrame {
            columns,
            column_names: col_names,
        })
    }

    /// Read a CSV file into an SFrame.
    pub fn from_csv(path: &str, options: Option<CsvOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();
        let (col_names, batch) = read_csv(path, &opts)?;

        let plan = PlannerNode::materialized(batch.clone());
        let num_rows = batch.num_rows() as u64;
        let dtypes = batch.dtypes();

        let columns: Vec<SArray> = dtypes
            .iter()
            .enumerate()
            .map(|(i, &dtype)| {
                SArray::from_plan(plan.clone(), dtype, Some(num_rows), i)
            })
            .collect();

        Ok(SFrame {
            columns,
            column_names: col_names,
        })
    }

    /// Build an SFrame from named columns.
    pub fn from_columns(cols: Vec<(&str, SArray)>) -> Result<Self> {
        let column_names: Vec<String> = cols.iter().map(|(name, _)| name.to_string()).collect();
        let columns: Vec<SArray> = cols.into_iter().map(|(_, col)| col).collect();
        Ok(SFrame {
            columns,
            column_names,
        })
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
        Ok(SFrame {
            columns: new_columns,
            column_names: new_names,
        })
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
        Ok(SFrame {
            columns,
            column_names: names,
        })
    }

    /// Remove a column (returns a new SFrame).
    pub fn remove_column(&self, name: &str) -> Result<SFrame> {
        let idx = self.column_index(name)?;
        let mut columns = self.columns.clone();
        let mut names = self.column_names.clone();
        columns.remove(idx);
        names.remove(idx);
        Ok(SFrame {
            columns,
            column_names: names,
        })
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

            return Ok(SFrame {
                columns,
                column_names: self.column_names.clone(),
            });
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

            return Ok(SFrame {
                columns,
                column_names: self.column_names.clone(),
            });
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

        let stream = self.compile_stream()?;
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SFrameError::Format(format!("Runtime error: {}", e)))?;
        let sorted = rt.block_on(sort::sort(stream, &sort_keys))?;
        self.from_batch(sorted)
    }

    /// Join with another SFrame.
    pub fn join(
        &self,
        other: &SFrame,
        left_col: &str,
        right_col: &str,
        how: JoinType,
    ) -> Result<SFrame> {
        let left_idx = self.column_index(left_col)?;
        let right_idx = other.column_index(right_col)?;

        let left_stream = self.compile_stream()?;
        let right_stream = other.compile_stream()?;

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SFrameError::Format(format!("Runtime error: {}", e)))?;

        let joined = rt.block_on(join::join(
            left_stream,
            right_stream,
            &JoinOn::new(left_idx, right_idx),
            how,
        ))?;

        // Build output column names: all left cols + right cols (minus join key)
        let mut names: Vec<String> = self.column_names.clone();
        for (i, name) in other.column_names.iter().enumerate() {
            if i != right_idx {
                // Disambiguate if needed
                let out_name = if names.contains(name) {
                    format!("{}.1", name)
                } else {
                    name.clone()
                };
                names.push(out_name);
            }
        }

        let dtypes = joined.dtypes();
        let plan = PlannerNode::materialized(joined.clone());
        let num_rows = joined.num_rows() as u64;

        let columns: Vec<SArray> = dtypes
            .iter()
            .enumerate()
            .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(num_rows), i))
            .collect();

        Ok(SFrame {
            columns,
            column_names: names,
        })
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

        let dtypes = grouped.dtypes();
        let plan = PlannerNode::materialized(grouped.clone());
        let num_rows = grouped.num_rows() as u64;

        let columns: Vec<SArray> = dtypes
            .iter()
            .enumerate()
            .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(num_rows), i))
            .collect();

        Ok(SFrame {
            columns,
            column_names: names,
        })
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

        let stream = self.compile_stream()?;
        for_each_batch_sync(stream, |batch| {
            writer.write_batch(&batch)
        })?;

        writer.finish()
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
    fn compile_stream(&self) -> Result<sframe_query::execute::BatchStream> {
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
        let plan = PlannerNode::materialized(batch.clone());
        let num_rows = batch.num_rows() as u64;
        let dtypes = batch.dtypes();

        let columns: Vec<SArray> = dtypes
            .iter()
            .enumerate()
            .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(num_rows), i))
            .collect();

        Ok(SFrame {
            columns,
            column_names: self.column_names.clone(),
        })
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

    /// Finalize: flush remaining buffered data, write segment footer
    /// and metadata files.
    pub fn finish(self) -> Result<()> {
        self.inner.finish()
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
}
