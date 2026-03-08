use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use sframe_core::{SArray, SFrame};
use sframe_query::algorithms::aggregators::AggSpec;
use sframe_query::algorithms::csv_parser::CsvOptions;
use sframe_query::algorithms::csv_writer::{CsvWriterOptions, QuoteStyle};
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
    py.detach(f).into_pyresult()
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
    // ── Constructor ─────────────────────────────────────────────────

    /// Construct an SFrame from various Python types:
    ///   - dict {str: list|SArray} → columns
    ///   - list of dicts → rows
    ///   - pandas DataFrame
    ///   - None / no arg → empty SFrame
    #[new]
    #[pyo3(signature = (data=None))]
    fn py_new(data: Option<&Bound<'_, PyAny>>, _py: Python<'_>) -> PyResult<Self> {
        let data = match data {
            None => return Ok(PySFrame::new(SFrame::from_columns(Vec::<(&str, SArray)>::new()).into_pyresult()?)),
            Some(d) => d,
        };

        // Dict → from_columns
        if let Ok(d) = data.cast::<PyDict>() {
            let mut names: Vec<String> = Vec::new();
            let mut arrays: Vec<SArray> = Vec::new();
            for (key, val) in d.iter() {
                let name: String = key.extract()?;
                // Value can be SArray or list
                if let Ok(sa) = val.extract::<PyRef<PySArray>>() {
                    names.push(name);
                    arrays.push(sa.inner.clone());
                } else if let Ok(list) = val.cast::<PyList>() {
                    let values: Vec<FlexType> = list
                        .iter()
                        .map(|item| py_to_flextype(&item))
                        .collect::<PyResult<_>>()?;
                    let dt = values
                        .iter()
                        .find(|v| !matches!(v, FlexType::Undefined))
                        .map(|v| v.type_enum())
                        .unwrap_or(sframe_types::flex_type::FlexTypeEnum::Undefined);
                    let sa = SArray::from_vec(values, dt).into_pyresult()?;
                    names.push(name);
                    arrays.push(sa);
                } else {
                    return Err(PyTypeError::new_err(format!(
                        "SFrame dict values must be lists or SArrays, got {} for column '{name}'",
                        val.get_type().name()?
                    )));
                }
            }
            let rust_cols: Vec<(&str, SArray)> = names
                .iter()
                .zip(arrays)
                .map(|(n, a)| (n.as_str(), a))
                .collect();
            let sf = SFrame::from_columns(rust_cols).into_pyresult()?;
            return Ok(PySFrame::new(sf));
        }

        // List of dicts → row-oriented construction
        if let Ok(list) = data.cast::<PyList>() {
            if list.is_empty() {
                return Ok(PySFrame::new(SFrame::from_columns(Vec::<(&str, SArray)>::new()).into_pyresult()?));
            }
            // Peek at first element to check if it's a dict
            let first = list.get_item(0)?;
            if first.cast::<PyDict>().is_ok() {
                // Collect column names from all dicts (preserving order from first)
                let mut col_order: Vec<String> = Vec::new();
                let mut col_set: std::collections::HashSet<String> = std::collections::HashSet::new();
                let mut col_data: HashMap<String, Vec<FlexType>> = HashMap::new();
                let n = list.len();

                for i in 0..n {
                    let row = list.get_item(i)?.cast::<PyDict>().map_err(|_| {
                        PyTypeError::new_err("All elements in list must be dicts")
                    })?.clone();
                    for (k, v) in row.iter() {
                        let name: String = k.extract()?;
                        if col_set.insert(name.clone()) {
                            col_order.push(name.clone());
                            // Back-fill with Undefined for prior rows
                            col_data.insert(name.clone(), vec![FlexType::Undefined; i]);
                        }
                        col_data.get_mut(&name).unwrap().push(py_to_flextype(&v)?);
                    }
                    // Fill Undefined for columns not in this row
                    for name in &col_order {
                        let vals = col_data.get_mut(name).unwrap();
                        if vals.len() <= i {
                            vals.push(FlexType::Undefined);
                        }
                    }
                }

                let mut rust_cols: Vec<(String, SArray)> = Vec::new();
                for name in &col_order {
                    let values = col_data.remove(name).unwrap();
                    let dt = values
                        .iter()
                        .find(|v| !matches!(v, FlexType::Undefined))
                        .map(|v| v.type_enum())
                        .unwrap_or(sframe_types::flex_type::FlexTypeEnum::Undefined);
                    let sa = SArray::from_vec(values, dt).into_pyresult()?;
                    rust_cols.push((name.clone(), sa));
                }
                let col_refs: Vec<(&str, SArray)> = rust_cols
                    .iter()
                    .map(|(n, a)| (n.as_str(), a.clone()))
                    .collect();
                let sf = SFrame::from_columns(col_refs).into_pyresult()?;
                return Ok(PySFrame::new(sf));
            }
        }

        // pandas DataFrame
        let type_name: String = data.get_type().name()?.extract()?;
        if type_name.contains("DataFrame") {
            // Extract column names
            let columns = data.getattr("columns")?;
            let col_list: Vec<String> = columns.call_method0("tolist")?.extract()?;
            let mut rust_cols: Vec<(String, SArray)> = Vec::new();
            for name in &col_list {
                let series = data.get_item(name)?;
                let values_list = series.call_method0("tolist")?;
                let pylist = values_list.cast::<PyList>()?;
                let values: Vec<FlexType> = pylist
                    .iter()
                    .map(|item| py_to_flextype(&item))
                    .collect::<PyResult<_>>()?;
                let dt = values
                    .iter()
                    .find(|v| !matches!(v, FlexType::Undefined))
                    .map(|v| v.type_enum())
                    .unwrap_or(sframe_types::flex_type::FlexTypeEnum::Undefined);
                let sa = SArray::from_vec(values, dt).into_pyresult()?;
                rust_cols.push((name.clone(), sa));
            }
            let col_refs: Vec<(&str, SArray)> = rust_cols
                .iter()
                .map(|(n, a)| (n.as_str(), a.clone()))
                .collect();
            let sf = SFrame::from_columns(col_refs).into_pyresult()?;
            return Ok(PySFrame::new(sf));
        }

        Err(PyTypeError::new_err(
            "SFrame() expects a dict, list of dicts, pandas DataFrame, or None",
        ))
    }

    // ── Constructors / I/O ──────────────────────────────────────────

    #[staticmethod]
    fn read(path: &str, py: Python<'_>) -> PyResult<Self> {
        let path = path.to_string();
        let sf = allow(py, move || SFrame::read(&path))?;
        Ok(PySFrame::new(sf))
    }

    #[staticmethod]
    #[pyo3(signature = (
        path,
        delimiter = ",",
        header = true,
        comment_char = "#",
        escape_char = "\\",
        double_quote = true,
        quote_char = "\"",
        skip_initial_space = true,
        column_type_hints = None,
        na_values = None,
        line_terminator = "\n",
        usecols = None,
        nrows = None,
        skiprows = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_csv(
        path: &str,
        delimiter: &str,
        header: bool,
        comment_char: &str,
        escape_char: &str,
        double_quote: bool,
        quote_char: &str,
        skip_initial_space: bool,
        column_type_hints: Option<&Bound<'_, PyDict>>,
        na_values: Option<Vec<String>>,
        line_terminator: &str,
        usecols: Option<Vec<String>>,
        nrows: Option<usize>,
        skiprows: usize,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let type_hints = match column_type_hints {
            Some(d) => {
                let mut hints = Vec::new();
                for (k, v) in d.iter() {
                    let name: String = k.extract()?;
                    let dtype_str: String = v.extract()?;
                    let dtype = crate::conversion::py_str_to_dtype(&dtype_str)?;
                    hints.push((name, dtype));
                }
                hints
            }
            None => Vec::new(),
        };
        let comment = if comment_char.is_empty() {
            None
        } else {
            Some(
                comment_char
                    .chars()
                    .next()
                    .ok_or_else(|| PyValueError::new_err("comment_char must be a single character"))?,
            )
        };
        let esc = escape_char
            .chars()
            .next()
            .ok_or_else(|| PyValueError::new_err("escape_char must be a single character"))?;
        let qc = quote_char
            .chars()
            .next()
            .ok_or_else(|| PyValueError::new_err("quote_char must be a single character"))?;

        let opts = CsvOptions {
            has_header: header,
            delimiter: delimiter.to_string(),
            type_hints,
            na_values: na_values.unwrap_or_default(),
            comment_char: comment,
            skip_rows: skiprows,
            row_limit: nrows,
            output_columns: usecols,
            use_custom_tokenizer: true,
            double_quote,
            line_terminator: line_terminator.to_string(),
            escape_char: esc,
            skip_initial_space,
            quote_char: qc,
        };

        let path = path.to_string();
        let sf = allow(py, move || SFrame::from_csv(&path, Some(opts)))?;
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
        if let Ok(list) = path.cast::<PyList>() {
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
            .zip(arrays)
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

    #[pyo3(signature = (
        path,
        delimiter = ",",
        quote_char = "\"",
        escape_char = "\\",
        line_terminator = "\n",
        na_rep = "",
        header = true,
        quoting = "minimal",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn to_csv(
        &self,
        path: &str,
        delimiter: &str,
        quote_char: &str,
        escape_char: &str,
        line_terminator: &str,
        na_rep: &str,
        header: bool,
        quoting: &str,
        py: Python<'_>,
    ) -> PyResult<()> {
        let qc = quote_char
            .chars()
            .next()
            .ok_or_else(|| PyValueError::new_err("quote_char must be a single character"))?;
        let esc = escape_char
            .chars()
            .next()
            .ok_or_else(|| PyValueError::new_err("escape_char must be a single character"))?;
        let quote_style = match quoting {
            "minimal" => QuoteStyle::Minimal,
            "all" => QuoteStyle::All,
            "nonnumeric" => QuoteStyle::NonNumeric,
            "none" => QuoteStyle::None,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown quoting style: '{quoting}'. Use 'minimal', 'all', 'nonnumeric', or 'none'"
                )))
            }
        };

        let opts = CsvWriterOptions {
            delimiter: delimiter.to_string(),
            quote_char: qc,
            escape_char: esc,
            line_terminator: line_terminator.to_string(),
            na_rep: na_rep.to_string(),
            header,
            quoting: quote_style,
        };

        let inner = self.inner.clone();
        let path = path.to_string();
        allow(py, move || inner.to_csv(&path, Some(opts)))
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

    /// Alias for num_columns (C++ API compat).
    fn num_cols(&self) -> usize {
        self.inner.num_columns()
    }

    /// (num_rows, num_columns) tuple.
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<(u64, usize)> {
        let inner = self.inner.clone();
        let nrows = allow(py, move || inner.num_rows())?;
        Ok((nrows, self.inner.num_columns()))
    }

    /// Column name → type string mapping.
    #[getter(dtype)]
    fn py_dtype(&self) -> HashMap<String, &'static str> {
        self.inner
            .column_names()
            .iter()
            .zip(self.inner.column_types())
            .map(|(n, dt)| (n.clone(), dtype_to_py_str(dt)))
            .collect()
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
        py.detach(move || format!("{inner}"))
    }

    fn __str__(&self, py: Python<'_>) -> String {
        let inner = self.inner.clone();
        py.detach(move || format!("{inner}"))
    }

    // ── Column access ───────────────────────────────────────────────

    // ── Aliases (C++ API compat) ───────────────────────────────────

    #[staticmethod]
    #[pyo3(signature = (
        url,
        delimiter = ",",
        header = true,
        comment_char = "#",
        escape_char = "\\",
        double_quote = true,
        quote_char = "\"",
        skip_initial_space = true,
        column_type_hints = None,
        na_values = None,
        line_terminator = "\n",
        usecols = None,
        nrows = None,
        skiprows = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn read_csv(
        url: &str,
        delimiter: &str,
        header: bool,
        comment_char: &str,
        escape_char: &str,
        double_quote: bool,
        quote_char: &str,
        skip_initial_space: bool,
        column_type_hints: Option<&Bound<'_, PyDict>>,
        na_values: Option<Vec<String>>,
        line_terminator: &str,
        usecols: Option<Vec<String>>,
        nrows: Option<usize>,
        skiprows: usize,
        py: Python<'_>,
    ) -> PyResult<Self> {
        Self::from_csv(
            url, delimiter, header, comment_char, escape_char, double_quote,
            quote_char, skip_initial_space, column_type_hints, na_values,
            line_terminator, usecols, nrows, skiprows, py,
        )
    }

    #[staticmethod]
    fn read_json(url: &str, py: Python<'_>) -> PyResult<Self> {
        Self::from_json(url, py)
    }

    #[pyo3(signature = (
        filename,
        delimiter = ",",
        quote_char = "\"",
        escape_char = "\\",
        line_terminator = "\n",
        na_rep = "",
        header = true,
        quoting = "minimal",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn export_csv(
        &self,
        filename: &str,
        delimiter: &str,
        quote_char: &str,
        escape_char: &str,
        line_terminator: &str,
        na_rep: &str,
        header: bool,
        quoting: &str,
        py: Python<'_>,
    ) -> PyResult<()> {
        self.to_csv(filename, delimiter, quote_char, escape_char, line_terminator, na_rep, header, quoting, py)
    }

    fn export_json(&self, path: &str, py: Python<'_>) -> PyResult<()> {
        self.to_json(path, py)
    }

    /// Alias for `select` (C++ API compat).
    fn select_columns(&self, names: Vec<String>) -> PyResult<PySFrame> {
        self.select(names)
    }

    /// Alias for `column` (C++ API compat).
    fn select_column(&self, name: &str) -> PyResult<PySArray> {
        self.column(name)
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

    /// Add multiple columns at once.
    fn add_columns(&self, data: &Bound<'_, PyAny>) -> PyResult<PySFrame> {
        // Accept dict {name: SArray} or list of SArrays with namelist
        let mut sf = self.inner.clone();
        if let Ok(d) = data.cast::<PyDict>() {
            for (k, v) in d.iter() {
                let name: String = k.extract()?;
                let sa: PyRef<PySArray> = v.extract()?;
                sf = sf.add_column(&name, sa.inner.clone()).into_pyresult()?;
            }
            return Ok(PySFrame::new(sf));
        }
        // Also accept list of (name, SArray) tuples
        if let Ok(list) = data.cast::<PyList>() {
            for item in list.iter() {
                let tup = item.cast::<PyTuple>().map_err(|_| {
                    PyTypeError::new_err("add_columns expects a dict or list of (name, SArray) tuples")
                })?;
                let name: String = tup.get_item(0)?.extract()?;
                let sa: PyRef<PySArray> = tup.get_item(1)?.extract()?;
                sf = sf.add_column(&name, sa.inner.clone()).into_pyresult()?;
            }
            return Ok(PySFrame::new(sf));
        }
        Err(PyTypeError::new_err(
            "add_columns expects a dict {name: SArray} or list of (name, SArray) tuples",
        ))
    }

    fn remove_column(&self, name: &str) -> PyResult<PySFrame> {
        let sf = self.inner.remove_column(name).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    /// Remove multiple columns at once.
    fn remove_columns(&self, column_names: Vec<String>) -> PyResult<PySFrame> {
        let mut sf = self.inner.clone();
        for name in &column_names {
            sf = sf.remove_column(name).into_pyresult()?;
        }
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

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = key.py();

        // Integer index -> single row as dict
        if let Ok(index) = key.extract::<isize>() {
            return self.getitem_int(index, py);
        }

        // Slice -> lazy SFrame
        if let Ok(slice) = key.cast::<pyo3::types::PySlice>() {
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
        if let Ok(list) = key.cast::<PyList>() {
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

    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self.inner = self.inner.remove_column(key).into_pyresult()?;
        Ok(())
    }

    /// Return self (SFrames are lazy/immutable plans, so copy is trivial).
    fn copy(&self) -> PySFrame {
        PySFrame::new(self.inner.clone())
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
    fn py_filter(&self, column_name: &str, func: Py<PyAny>) -> PyResult<PySFrame> {
        let pred: Arc<dyn Fn(&FlexType) -> bool + Send + Sync> =
            Arc::new(move |v: &FlexType| {
                Python::attach(|py| {
                    let py_val = flextype_to_py(py, v);
                    match func.call1(py, (py_val,)) {
                        Ok(r) => r.bind(py).is_truthy().unwrap_or(false),
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

    /// Filter rows where column_name's value is in (or not in) values.
    #[pyo3(signature = (values, column_name, exclude=false))]
    fn filter_by(
        &self,
        values: &Bound<'_, PyAny>,
        column_name: &str,
        exclude: bool,
        py: Python<'_>,
    ) -> PyResult<PySFrame> {
        // Collect values into a HashSet for O(1) lookup.
        let val_list: Vec<FlexType> = if let Ok(sa) = values.extract::<PyRef<PySArray>>() {
            let sa_inner = sa.inner.clone();
            allow(py, move || sa_inner.to_vec())?
        } else if let Ok(list) = values.cast::<PyList>() {
            list.iter()
                .map(|item| py_to_flextype(&item))
                .collect::<PyResult<_>>()?
        } else {
            return Err(PyTypeError::new_err(
                "filter_by values must be a list or SArray",
            ));
        };
        let val_set: std::collections::HashSet<FlexType> = val_list.into_iter().collect();
        let pred: Arc<dyn Fn(&FlexType) -> bool + Send + Sync> = if exclude {
            Arc::new(move |v: &FlexType| !val_set.contains(v))
        } else {
            Arc::new(move |v: &FlexType| val_set.contains(v))
        };
        let sf = self.inner.filter(column_name, pred).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    /// Add a sequential row number column.
    #[pyo3(signature = (column_name="id", start=0))]
    fn add_row_number(
        &self,
        column_name: &str,
        start: i64,
        py: Python<'_>,
    ) -> PyResult<PySFrame> {
        let inner = self.inner.clone();
        let nrows = allow(py, move || inner.num_rows())?;
        let sa = SArray::from_range(start, 1, nrows);
        let sf = self
            .inner
            .add_column(column_name, sa)
            .into_pyresult()?;
        // Move the new column to position 0
        let mut names = sf.column_names().to_vec();
        let pos = names.iter().position(|n| n == column_name).unwrap();
        names.remove(pos);
        names.insert(0, column_name.to_string());
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let sf = sf.select(&refs).into_pyresult()?;
        Ok(PySFrame::new(sf))
    }

    /// Split into (clean, dirty) SFrames based on missing values.
    #[pyo3(signature = (columns=None, how="any"))]
    fn dropna_split(
        &self,
        columns: Option<Vec<String>>,
        how: &str,
        py: Python<'_>,
    ) -> PyResult<(PySFrame, PySFrame)> {
        let inner = self.inner.clone();
        let inner2 = self.inner.clone();
        let col_names: Vec<String> = match &columns {
            Some(c) => c.clone(),
            None => inner.column_names().to_vec(),
        };
        let col_names2 = col_names.clone();
        let how_str = how.to_string();
        let how_str2 = how_str.clone();
        // Clean: dropna(None, how) checks all columns when column is None
        let clean = allow(py, move || {
            // If specific columns requested, we need to build a mask ourselves.
            // Otherwise use dropna(None) which checks all.
            if columns.is_some() {
                let mut mask: Option<SArray> = None;
                for cname in &col_names {
                    let col = inner.column(cname)?;
                    let na = col.is_na();
                    mask = Some(match mask {
                        None => na,
                        Some(prev) => {
                            if how_str == "all" {
                                prev.and(&na)?
                            } else {
                                prev.or(&na)?
                            }
                        }
                    });
                }
                match mask {
                    // NOT mask → clean rows (no missing)
                    Some(m) => {
                        let not_mask: SArray = m.eq_scalar(FlexType::Integer(0));
                        inner.logical_filter(not_mask)
                    }
                    None => Ok(inner.clone()),
                }
            } else {
                inner.dropna(None, &how_str)
            }
        })?;
        // Dirty: rows WITH missing values (complement of clean)
        let dirty = allow(py, move || {
            let mut mask: Option<SArray> = None;
            for cname in &col_names2 {
                let col = inner2.column(cname)?;
                let na = col.is_na();
                mask = Some(match mask {
                    None => na,
                    Some(prev) => {
                        if how_str2 == "all" {
                            prev.and(&na)?
                        } else {
                            prev.or(&na)?
                        }
                    }
                });
            }
            match mask {
                Some(m) => inner2.logical_filter(m),
                None => inner2.head(0),
            }
        })?;
        Ok((PySFrame::new(clean), PySFrame::new(dirty)))
    }

    /// Formatted printing with configurable dimensions.
    #[pyo3(signature = (num_rows=10, num_columns=40, max_column_width=30))]
    fn print_rows(&self, num_rows: usize, num_columns: usize, max_column_width: usize, py: Python<'_>) -> PyResult<String> {
        let inner = self.inner.clone();
        let sf = allow(py, move || inner.head(num_rows))?;
        let names = sf.column_names();
        let display_cols = names.len().min(num_columns);

        let mut lines: Vec<String> = Vec::new();

        // Header
        let mut header = String::new();
        for name in names.iter().take(display_cols) {
            let truncated: String = name.chars().take(max_column_width).collect();
            header.push_str(&format!("{:<width$} ", truncated, width = max_column_width.min(truncated.len() + 2)));
        }
        lines.push(header);
        lines.push("-".repeat(lines[0].len()));

        // Rows
        let rows = sf.iter_rows().into_pyresult()?;
        for row in &rows {
            let mut line = String::new();
            for val in row.iter().take(display_cols) {
                let s = format!("{val}");
                let truncated: String = s.chars().take(max_column_width).collect();
                line.push_str(&format!("{:<width$} ", truncated, width = max_column_width.min(truncated.len() + 2)));
            }
            lines.push(line);
        }

        let result = lines.join("\n");
        Ok(result)
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
                    "Unknown join type: '{how}'"
                )))
            }
        };

        if let Ok(col) = on.extract::<String>() {
            let left = self.inner.clone();
            let right = other.inner.clone();
            let sf = allow(py, move || left.join(&right, &col, &col, join_type))?;
            return Ok(PySFrame::new(sf));
        }

        if let Ok(d) = on.cast::<PyDict>() {
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
                    PyKeyError::new_err(format!("Column '{col_name}' not found"))
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

    fn iter_rows(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
            .detach(move || inner.batch_channel(2))
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
    fn getitem_int(&self, index: isize, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
    ) -> PyResult<Py<PyAny>> {
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

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
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
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")));
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

    if let Ok(list) = spec.cast::<PyList>() {
        let mut keys = Vec::new();
        for item in list.iter() {
            if let Ok(tup) = item.cast::<PyTuple>() {
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
