use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use sframe_core::{SArray, SFrame};
use sframe_query::algorithms::aggregators::AggSpec;
use sframe_query::algorithms::csv_parser::CsvOptions;
use sframe_query::algorithms::csv_writer::CsvWriterOptions;
use sframe_query::algorithms::join::JoinType;
use sframe_query::algorithms::sort::SortOrder;
use sframe_types::flex_type::FlexType;

use crate::conversion::{dtype_to_py_str, flextype_to_py, py_to_flextype};
use crate::error::IntoPyResult;
use crate::py_sarray::PySArray;

/// Release the GIL, run a closure that returns `Result<T>`, reacquire GIL.
fn allow<T: Send>(
    py: Python<'_>,
    f: impl FnOnce() -> sframe_types::error::Result<T> + Send,
) -> PyResult<T> {
    py.allow_threads(f).into_pyresult()
}

/// A lazy dataframe backed by the Rust SFrame.
#[pyclass(name = "SFrame")]
pub struct PySFrame {
    pub(crate) inner: SFrame,
}

impl PySFrame {
    pub fn new(inner: SFrame) -> Self {
        PySFrame { inner }
    }
}

#[pymethods]
impl PySFrame {
    // ── Constructors / I/O ──────────────────────────────────────────

    #[staticmethod]
    fn read(path: &str, py: Python<'_>) -> PyResult<Self> {
        let path = path.to_string();
        let sf = allow(py, move || SFrame::read(&path))?;
        Ok(PySFrame::new(sf))
    }

    #[staticmethod]
    #[pyo3(signature = (path, delimiter=None))]
    fn from_csv(path: &str, delimiter: Option<&str>, py: Python<'_>) -> PyResult<Self> {
        let path = path.to_string();
        let opts = delimiter.map(|d| {
            let mut o = CsvOptions::default();
            o.delimiter = d.to_string();
            o
        });
        let sf = allow(py, move || SFrame::from_csv(&path, opts))?;
        Ok(PySFrame::new(sf))
    }

    #[staticmethod]
    fn from_json(path: &str, py: Python<'_>) -> PyResult<Self> {
        let path = path.to_string();
        let sf = allow(py, move || SFrame::from_json(&path))?;
        Ok(PySFrame::new(sf))
    }

    #[staticmethod]
    #[pyo3(signature = (path))]
    fn from_parquet(path: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        if let Ok(list) = path.downcast::<PyList>() {
            let paths: Vec<String> = list
                .iter()
                .map(|item| item.extract::<String>())
                .collect::<PyResult<_>>()?;
            let sf = allow(py, move || {
                let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
                SFrame::from_parquet_files(&path_refs)
            })?;
            Ok(PySFrame::new(sf))
        } else {
            let path_str: String = path.extract()?;
            let sf = allow(py, move || SFrame::from_parquet(&path_str))?;
            Ok(PySFrame::new(sf))
        }
    }

    #[staticmethod]
    fn from_columns(cols: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut names: Vec<String> = Vec::new();
        let mut arrays: Vec<SArray> = Vec::new();
        for (key, val) in cols.iter() {
            let name: String = key.extract()?;
            let sa: PyRef<PySArray> = val.extract()?;
            names.push(name);
            arrays.push(sa.inner.clone());
        }
        let rust_cols: Vec<(&str, SArray)> = names
            .iter()
            .zip(arrays.into_iter())
            .map(|(n, a)| (n.as_str(), a))
            .collect();
        let sf = SFrame::from_columns(rust_cols).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    fn save(&self, path: &str, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let path = path.to_string();
        allow(py, move || inner.save(&path))
    }

    #[pyo3(signature = (path, delimiter=None))]
    fn to_csv(&self, path: &str, delimiter: Option<&str>, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let path = path.to_string();
        let opts = delimiter.map(|d| {
            let mut o = CsvWriterOptions::default();
            o.delimiter = d.to_string();
            o
        });
        allow(py, move || inner.to_csv(&path, opts))
    }

    fn to_json(&self, path: &str, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let path = path.to_string();
        allow(py, move || inner.to_json(&path))
    }

    #[pyo3(signature = (path, sharded=false))]
    fn to_parquet(&self, path: &str, sharded: bool, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let path = path.to_string();
        if sharded {
            allow(py, move || inner.to_parquet_sharded(&path))
        } else {
            allow(py, move || inner.to_parquet(&path))
        }
    }

    // ── Schema / metadata ───────────────────────────────────────────

    fn num_rows(&self, py: Python<'_>) -> PyResult<u64> {
        let inner = self.inner.clone();
        allow(py, move || inner.num_rows())
    }

    fn num_columns(&self) -> usize {
        self.inner.num_columns()
    }

    fn column_names(&self) -> Vec<String> {
        self.inner.column_names().to_vec()
    }

    fn column_types(&self) -> Vec<&'static str> {
        self.inner
            .column_types()
            .into_iter()
            .map(dtype_to_py_str)
            .collect()
    }

    fn schema(&self) -> Vec<(String, &'static str)> {
        self.inner
            .schema()
            .into_iter()
            .map(|(name, dt)| (name, dtype_to_py_str(dt)))
            .collect()
    }

    fn explain(&self) -> String {
        self.inner.explain()
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        let inner = self.inner.clone();
        Ok(allow(py, move || inner.num_rows())? as usize)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let inner = self.inner.clone();
        py.allow_threads(move || format!("{}", inner))
    }

    fn __str__(&self, py: Python<'_>) -> String {
        let inner = self.inner.clone();
        py.allow_threads(move || format!("{}", inner))
    }

    // ── Column access ───────────────────────────────────────────────

    fn column(&self, name: &str) -> PyResult<PySArray> {
        let sa = self.inner.column(name).into_pyresult()?;
        Ok(PySArray::new(sa.clone()))
    }

    fn select(&self, names: Vec<String>) -> PyResult<PySFrame> {
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let sf = self.inner.select(&refs).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    fn add_column(&self, name: &str, col: &PySArray) -> PyResult<PySFrame> {
        let sf = self
            .inner
            .add_column(name, col.inner.clone())
            .into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    fn remove_column(&self, name: &str) -> PyResult<PySFrame> {
        let sf = self.inner.remove_column(name).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    fn replace_column(&self, name: &str, col: &PySArray) -> PyResult<PySFrame> {
        let sf = self
            .inner
            .replace_column(name, col.inner.clone())
            .into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    fn rename(&self, mapping: &Bound<'_, PyDict>) -> PyResult<PySFrame> {
        let mut owned: Vec<(String, String)> = Vec::new();
        for (k, v) in mapping.iter() {
            let old: String = k.extract()?;
            let new: String = v.extract()?;
            owned.push((old, new));
        }
        let map: HashMap<&str, &str> = owned
            .iter()
            .map(|(o, n)| (o.as_str(), n.as_str()))
            .collect();
        let sf = self.inner.rename(&map).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    fn swap_columns(&self, name1: &str, name2: &str) -> PyResult<PySFrame> {
        let sf = self.inner.swap_columns(name1, name2).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    // ── Indexing ────────────────────────────────────────────────────

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = key.py();

        // Integer index -> single row as dict
        if let Ok(index) = key.extract::<isize>() {
            return self.getitem_int(index, py);
        }

        // Slice -> lazy SFrame
        if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            return self.getitem_slice(slice, py);
        }

        // String key -> return column as PySArray
        if let Ok(name) = key.extract::<String>() {
            let sa = self.inner.column(&name).into_pyresult()?;
            return Ok(PySArray::new(sa.clone())
                .into_pyobject(py)?
                .into_any()
                .unbind());
        }

        // List of strings -> select columns
        if let Ok(list) = key.downcast::<PyList>() {
            if let Ok(names) = list
                .iter()
                .map(|item| item.extract::<String>())
                .collect::<PyResult<Vec<String>>>()
            {
                let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
                let sf = self.inner.select(&refs).into_pyresult()?;
                return Ok(PySFrame::new(sf).into_pyobject(py)?.into_any().unbind());
            }
        }

        // PySArray mask -> logical_filter
        if let Ok(mask) = key.extract::<PyRef<PySArray>>() {
            let sf = self
                .inner
                .logical_filter(mask.inner.clone())
                .into_pyresult()?;
            return Ok(PySFrame::new(sf).into_pyobject(py)?.into_any().unbind());
        }

        Err(PyTypeError::new_err(
            "SFrame index must be an int, slice, column name (str), list of column names, or a boolean SArray mask",
        ))
    }

    fn __setitem__(&mut self, key: &str, value: &PySArray) -> PyResult<()> {
        let names = self.inner.column_names();
        if names.contains(&key.to_string()) {
            self.inner = self
                .inner
                .replace_column(key, value.inner.clone())
                .into_pyresult()?;
        } else {
            self.inner = self
                .inner
                .add_column(key, value.inner.clone())
                .into_pyresult()?;
        }
        Ok(())
    }

    fn __contains__(&self, key: &str) -> bool {
        self.inner.column_names().contains(&key.to_string())
    }

    // ── Filtering & selection ───────────────────────────────────────

    #[pyo3(signature = (n=10))]
    fn head(&self, n: usize, py: Python<'_>) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let sf = allow(py, move || inner.head(n))?;
        Ok(PySFrame::new(sf))
    }

    #[pyo3(signature = (n=10))]
    fn tail(&self, n: usize, py: Python<'_>) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let sf = allow(py, move || inner.tail(n))?;
        Ok(PySFrame::new(sf))
    }

    #[pyo3(name = "filter")]
    fn py_filter(&self, column_name: &str, func: PyObject) -> PyResult<PySFrame> {
        let pred: Arc<dyn Fn(&FlexType) -> bool + Send + Sync> =
            Arc::new(move |v: &FlexType| {
                Python::with_gil(|py| {
                    let py_val = flextype_to_py(py, v);
                    match func.call1(py, (py_val,)) {
                        Ok(r) => r.is_truthy(py).unwrap_or(false),
                        Err(_) => false,
                    }
                })
            });
        let sf = self.inner.filter(column_name, pred).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    fn logical_filter(&self, mask: &PySArray) -> PyResult<PySFrame> {
        let sf = self
            .inner
            .logical_filter(mask.inner.clone())
            .into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    #[pyo3(signature = (fraction, seed=None))]
    fn sample(&self, fraction: f64, seed: Option<u64>, py: Python<'_>) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let sf = allow(py, move || inner.sample(fraction, seed))?;
        Ok(PySFrame::new(sf))
    }

    #[pyo3(signature = (fraction, seed=None))]
    fn random_split(
        &self,
        fraction: f64,
        seed: Option<u64>,
        py: Python<'_>,
    ) -> PyResult<(PySFrame, PySFrame)> {
        let inner = self.inner.clone();
        let (a, b) = allow(py, move || inner.random_split(fraction, seed))?;
        Ok((PySFrame::new(a), PySFrame::new(b)))
    }

    #[pyo3(signature = (column, k, reverse=false))]
    fn topk(&self, column: &str, k: usize, reverse: bool, py: Python<'_>) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let col = column.to_string();
        let sf = allow(py, move || inner.topk(&col, k, reverse))?;
        Ok(PySFrame::new(sf))
    }

    // ── Data manipulation ───────────────────────────────────────────

    fn append(&self, other: &PySFrame) -> PyResult<PySFrame> {
        let sf = self.inner.append(&other.inner).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    fn unique(&self, py: Python<'_>) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let sf = allow(py, move || inner.unique())?;
        Ok(PySFrame::new(sf))
    }

    #[pyo3(signature = (column=None, how="any"))]
    fn dropna(&self, column: Option<&str>, how: &str, py: Python<'_>) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let col = column.map(|s| s.to_string());
        let how = how.to_string();
        let sf = allow(py, move || inner.dropna(col.as_deref(), &how))?;
        Ok(PySFrame::new(sf))
    }

    fn fillna(&self, column: &str, value: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PySFrame> {
        let val = py_to_flextype(value)?;
        let inner = self.inner.clone();
        let col = column.to_string();
        let sf = allow(py, move || inner.fillna(&col, val))?;
        Ok(PySFrame::new(sf))
    }

    fn materialize(&self, py: Python<'_>) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let sf = allow(py, move || inner.materialize())?;
        Ok(PySFrame::new(sf))
    }

    // ── Sort ────────────────────────────────────────────────────────

    #[pyo3(signature = (column_or_columns, ascending=true))]
    fn sort(
        &self,
        column_or_columns: &Bound<'_, PyAny>,
        ascending: bool,
        py: Python<'_>,
    ) -> PyResult<PySFrame> {
        let keys = parse_sort_keys(column_or_columns, ascending)?;
        let inner = self.inner.clone();
        let sf = allow(py, move || {
            let key_refs: Vec<(&str, SortOrder)> = keys
                .iter()
                .map(|(name, order)| (name.as_str(), *order))
                .collect();
            inner.sort(&key_refs)
        })?;
        Ok(PySFrame::new(sf))
    }

    // ── Join ────────────────────────────────────────────────────────

    #[pyo3(signature = (other, on, how="inner"))]
    fn join(
        &self,
        other: &PySFrame,
        on: &Bound<'_, PyAny>,
        how: &str,
        py: Python<'_>,
    ) -> PyResult<PySFrame> {
        let join_type = match how {
            "inner" => JoinType::Inner,
            "left" => JoinType::Left,
            "right" => JoinType::Right,
            "full" => JoinType::Full,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown join type: '{}'",
                    how
                )))
            }
        };

        if let Ok(col) = on.extract::<String>() {
            let left = self.inner.clone();
            let right = other.inner.clone();
            let sf = allow(py, move || left.join(&right, &col, &col, join_type))?;
            return Ok(PySFrame::new(sf));
        }

        if let Ok(d) = on.downcast::<PyDict>() {
            let pairs: Vec<(String, String)> = d
                .iter()
                .map(|(k, v)| {
                    let left: String = k.extract()?;
                    let right: String = v.extract()?;
                    Ok((left, right))
                })
                .collect::<PyResult<_>>()?;
            let left = self.inner.clone();
            let right = other.inner.clone();
            let sf = allow(py, move || {
                let pair_refs: Vec<(&str, &str)> = pairs
                    .iter()
                    .map(|(l, r)| (l.as_str(), r.as_str()))
                    .collect();
                left.join_on(&right, &pair_refs, join_type)
            })?;
            return Ok(PySFrame::new(sf));
        }

        Err(PyTypeError::new_err(
            "'on' must be a column name (str) or a dict mapping left->right column names",
        ))
    }

    // ── Groupby ─────────────────────────────────────────────────────

    fn groupby(
        &self,
        keys: Vec<String>,
        operations: &Bound<'_, PyDict>,
        py: Python<'_>,
    ) -> PyResult<PySFrame> {
        let mut agg_specs: Vec<AggSpec> = Vec::new();

        for (output_name, spec) in operations.iter() {
            let output: String = output_name.extract()?;
            let py_spec: PyRef<crate::PyAggSpec> = spec.extract()?;

            let col_name = &py_spec.column;
            let col_idx = self
                .inner
                .column_names()
                .iter()
                .position(|n| n == col_name)
                .ok_or_else(|| {
                    PyKeyError::new_err(format!("Column '{}' not found", col_name))
                })?;

            let aggregator = py_spec.make_aggregator()?;
            agg_specs.push(AggSpec::new(col_idx, aggregator, &output));
        }

        let inner = self.inner.clone();
        let sf = allow(py, move || {
            let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
            inner.groupby(&key_refs, agg_specs)
        })?;
        Ok(PySFrame::new(sf))
    }

    // ── Pack/Unpack/Stack ───────────────────────────────────────────

    fn pack_columns(
        &self,
        columns: Vec<String>,
        output_name: &str,
        py: Python<'_>,
    ) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let out = output_name.to_string();
        let sf = allow(py, move || {
            let refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            inner.pack_columns(&refs, &out)
        })?;
        Ok(PySFrame::new(sf))
    }

    #[pyo3(signature = (column_name, prefix=None))]
    fn unpack_column(
        &self,
        column_name: &str,
        prefix: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let col = column_name.to_string();
        let pfx = prefix.map(|s| s.to_string());
        let sf = allow(py, move || inner.unpack_column(&col, pfx.as_deref()))?;
        Ok(PySFrame::new(sf))
    }

    fn stack(
        &self,
        column_name: &str,
        new_column_name: &str,
        py: Python<'_>,
    ) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let col = column_name.to_string();
        let new_col = new_column_name.to_string();
        let sf = allow(py, move || inner.stack(&col, &new_col))?;
        Ok(PySFrame::new(sf))
    }

    // ── Iteration ───────────────────────────────────────────────────

    fn iter_rows(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let rows = allow(py, move || inner.iter_rows())?;
        let names = self.inner.column_names();
        let list = PyList::empty(py);
        for row in &rows {
            let dict = PyDict::new(py);
            for (name, val) in names.iter().zip(row.iter()) {
                dict.set_item(name, flextype_to_py(py, val))?;
            }
            list.append(dict)?;
        }
        Ok(list.into_any().unbind())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PySFrameIter> {
        let inner = slf.inner.clone();
        let rx = slf
            .py()
            .allow_threads(move || inner.batch_channel(2))
            .into_pyresult()?;
        let names = slf.inner.column_names().to_vec();
        Ok(PySFrameIter {
            receiver: std::sync::Mutex::new(rx),
            names,
            current_batch: None,
            batch_offset: 0,
        })
    }
}

impl PySFrame {
    fn getitem_int(&self, index: isize, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let len = allow(py, move || inner.num_rows())? as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx < 0 || idx >= len {
            return Err(PyIndexError::new_err("SFrame index out of range"));
        }
        let begin = idx as u64;
        let inner = self.inner.clone();
        let sliced_sf = allow(py, move || inner.slice(begin, begin + 1))?;
        // Convert single row to dict
        let names = sliced_sf.column_names();
        let dict = PyDict::new(py);
        for name in names {
            let col = sliced_sf.column(name).into_pyresult()?;
            let vals = col.to_vec().into_pyresult()?;
            if let Some(v) = vals.into_iter().next() {
                dict.set_item(name, flextype_to_py(py, &v))?;
            }
        }
        Ok(dict.into_any().unbind())
    }

    fn getitem_slice(
        &self,
        slice: &Bound<'_, pyo3::types::PySlice>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let len = allow(py, move || inner.num_rows())? as isize;

        let indices = slice.indices(len)?;
        let start = indices.start;
        let stop = indices.stop;
        let step = indices.step;

        if step != 1 {
            return Err(PyValueError::new_err(
                "SFrame slicing with step != 1 is not supported",
            ));
        }

        if start >= stop {
            let inner = self.inner.clone();
            let empty = allow(py, move || inner.head(0))?;
            return Ok(PySFrame::new(empty)
                .into_pyobject(py)?
                .into_any()
                .unbind());
        }

        let begin = start as u64;
        let end = stop as u64;
        let inner = self.inner.clone();
        let sliced = allow(py, move || inner.slice(begin, end))?;
        Ok(PySFrame::new(sliced)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }
}

/// Streaming iterator for PySFrame rows (yields dicts).
///
/// Pulls batches on demand through a bounded channel so memory stays
/// O(batch_size) instead of O(total_rows).
#[pyclass]
pub struct PySFrameIter {
    receiver: std::sync::Mutex<std::sync::mpsc::Receiver<sframe_types::error::Result<sframe_query::batch::SFrameRows>>>,
    names: Vec<String>,
    current_batch: Option<sframe_query::batch::SFrameRows>,
    batch_offset: usize,
}

#[pymethods]
impl PySFrameIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        loop {
            // Yield from current batch if available.
            if let Some(batch) = &self.current_batch {
                if self.batch_offset < batch.num_rows() {
                    let dict = PyDict::new(py);
                    for (col_idx, name) in self.names.iter().enumerate() {
                        let val = batch.column(col_idx).get(self.batch_offset);
                        dict.set_item(name, flextype_to_py(py, &val))?;
                    }
                    self.batch_offset += 1;
                    return Ok(Some(dict.into_any().unbind()));
                }
                // Current batch exhausted.
                self.current_batch = None;
                self.batch_offset = 0;
            }

            // Pull next batch. The heavy I/O happens in the background
            // thread; recv typically returns immediately from the prefetch
            // buffer.
            match self.receiver.lock().unwrap().recv() {
                Ok(Ok(batch)) => {
                    self.current_batch = Some(batch);
                }
                Ok(Err(e)) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)));
                }
                Err(_) => return Ok(None), // Channel closed, stream finished.
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn parse_sort_keys(
    spec: &Bound<'_, PyAny>,
    default_ascending: bool,
) -> PyResult<Vec<(String, SortOrder)>> {
    let order = if default_ascending {
        SortOrder::Ascending
    } else {
        SortOrder::Descending
    };

    if let Ok(name) = spec.extract::<String>() {
        return Ok(vec![(name, order)]);
    }

    if let Ok(list) = spec.downcast::<PyList>() {
        let mut keys = Vec::new();
        for item in list.iter() {
            if let Ok(tup) = item.downcast::<PyTuple>() {
                if tup.len() == 2 {
                    let name: String = tup.get_item(0)?.extract()?;
                    let asc: bool = tup.get_item(1)?.extract()?;
                    let ord = if asc {
                        SortOrder::Ascending
                    } else {
                        SortOrder::Descending
                    };
                    keys.push((name, ord));
                    continue;
                }
            }
            let name: String = item.extract()?;
            keys.push((name, order));
        }
        return Ok(keys);
    }

    Err(PyTypeError::new_err(
        "sort key must be a column name, list of column names, or list of (name, ascending) tuples",
    ))
}
