use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use sframe_core::SArray;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::conversion::{dtype_to_py_str, flextype_to_py, py_str_to_dtype, py_to_flextype};
use crate::error::IntoPyResult;

/// A lazy columnar array backed by the Rust SArray.
#[pyclass(name = "SArray")]
pub struct PySArray {
    pub(crate) inner: SArray,
}

impl PySArray {
    pub fn new(inner: SArray) -> Self {
        PySArray { inner }
    }
}

/// Release the GIL, run a closure that returns `Result<T>`, reacquire GIL.
/// This prevents deadlocks when Rust parallel execution needs to call back
/// into Python (e.g. Python lambdas in apply/filter).
fn allow<T: Send>(py: Python<'_>, f: impl FnOnce() -> sframe_types::error::Result<T> + Send) -> PyResult<T> {
    py.allow_threads(f).into_pyresult()
}

#[pymethods]
impl PySArray {
    /// Construct an SArray from a Python list with optional dtype.
    #[new]
    #[pyo3(signature = (data, dtype=None))]
    fn py_new(data: &Bound<'_, PyList>, dtype: Option<&str>) -> PyResult<Self> {
        let values: Vec<FlexType> = data
            .iter()
            .map(|item| py_to_flextype(&item))
            .collect::<PyResult<_>>()?;

        let target_dt = dtype.map(py_str_to_dtype).transpose()?;

        // Infer from first non-None value
        let inferred_dt = values
            .iter()
            .find(|v| !matches!(v, FlexType::Undefined))
            .map(|v| v.type_enum())
            .unwrap_or(FlexTypeEnum::Undefined);

        let dt = target_dt.unwrap_or(inferred_dt);
        let sa = SArray::from_vec(values, inferred_dt).into_pyresult()?;

        // If explicit dtype differs, cast
        if dt != inferred_dt {
            Ok(PySArray { inner: sa.astype(dt, false) })
        } else {
            Ok(PySArray { inner: sa })
        }
    }

    /// The data type of this array.
    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_to_py_str(self.inner.dtype())
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        let inner = self.inner.clone();
        Ok(allow(py, move || inner.len())? as usize)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let inner = self.inner.clone();
        py.allow_threads(move || format!("{}", inner))
    }

    fn __str__(&self, py: Python<'_>) -> String {
        let inner = self.inner.clone();
        py.allow_threads(move || format!("{}", inner))
    }

    /// First n values as a Python list.
    #[pyo3(signature = (n=10))]
    fn head(&self, n: usize, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let vals = allow(py, move || inner.head(n))?;
        let list = PyList::new(py, vals.iter().map(|v| flextype_to_py(py, v)))?;
        Ok(list.into_any().unbind())
    }

    /// Last n values as a Python list.
    #[pyo3(signature = (n=10))]
    fn tail(&self, n: usize, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let vals = allow(py, move || inner.tail(n))?;
        let list = PyList::new(py, vals.iter().map(|v| flextype_to_py(py, v)))?;
        Ok(list.into_any().unbind())
    }

    /// All values as a Python list.
    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let vals = allow(py, move || inner.to_vec())?;
        let list = PyList::new(py, vals.iter().map(|v| flextype_to_py(py, v)))?;
        Ok(list.into_any().unbind())
    }

    // ── Arithmetic operators ────────────────────────────────────────

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.binop(other, BinOp::Add)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.binop(other, BinOp::Add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.binop(other, BinOp::Sub)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.reverse_scalar_op(other, BinOp::Sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.binop(other, BinOp::Mul)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.binop(other, BinOp::Mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.binop(other, BinOp::Div)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.reverse_scalar_op(other, BinOp::Div)
    }

    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.binop(other, BinOp::Rem)
    }

    fn __rmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        self.reverse_scalar_op(other, BinOp::Rem)
    }

    // ── Comparison operators ────────────────────────────────────────

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: pyo3::basic::CompareOp) -> PyResult<PySArray> {
        if let Ok(rhs) = other.extract::<PyRef<PySArray>>() {
            let result = match op {
                pyo3::basic::CompareOp::Eq => self.inner.eq(&rhs.inner),
                pyo3::basic::CompareOp::Ne => self.inner.ne(&rhs.inner),
                pyo3::basic::CompareOp::Lt => self.inner.lt(&rhs.inner),
                pyo3::basic::CompareOp::Le => self.inner.le(&rhs.inner),
                pyo3::basic::CompareOp::Gt => self.inner.gt(&rhs.inner),
                pyo3::basic::CompareOp::Ge => self.inner.ge(&rhs.inner),
            }
            .into_pyresult()?;
            return Ok(PySArray::new(result));
        }
        let scalar = py_to_flextype(other)?;
        let result = match op {
            pyo3::basic::CompareOp::Eq => self.inner.eq_scalar(scalar),
            pyo3::basic::CompareOp::Ne => self.inner.ne_scalar(scalar),
            pyo3::basic::CompareOp::Lt => self.inner.lt_scalar(scalar),
            pyo3::basic::CompareOp::Le => self.inner.le_scalar(scalar),
            pyo3::basic::CompareOp::Gt => self.inner.gt_scalar(scalar),
            pyo3::basic::CompareOp::Ge => self.inner.ge_scalar(scalar),
        };
        Ok(PySArray::new(result))
    }

    // ── Logical operators ───────────────────────────────────────────

    fn __and__(&self, other: &PySArray) -> PyResult<PySArray> {
        let result = self.inner.and(&other.inner).into_pyresult()?;
        Ok(PySArray::new(result))
    }

    fn __or__(&self, other: &PySArray) -> PyResult<PySArray> {
        let result = self.inner.or(&other.inner).into_pyresult()?;
        Ok(PySArray::new(result))
    }

    // ── Reductions ──────────────────────────────────────────────────

    fn sum(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let val = allow(py, move || inner.sum())?;
        Ok(flextype_to_py(py, &val))
    }

    fn min(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let val = allow(py, move || inner.min_val())?;
        Ok(flextype_to_py(py, &val))
    }

    fn max(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let val = allow(py, move || inner.max_val())?;
        Ok(flextype_to_py(py, &val))
    }

    fn mean(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let val = allow(py, move || inner.mean())?;
        Ok(flextype_to_py(py, &val))
    }

    #[pyo3(signature = (ddof=1))]
    fn std(&self, ddof: u8, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let val = allow(py, move || inner.std_dev(ddof))?;
        Ok(flextype_to_py(py, &val))
    }

    #[pyo3(signature = (ddof=1))]
    fn var(&self, ddof: u8, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let val = allow(py, move || inner.variance(ddof))?;
        Ok(flextype_to_py(py, &val))
    }

    fn any(&self, py: Python<'_>) -> PyResult<bool> {
        let inner = self.inner.clone();
        allow(py, move || inner.any())
    }

    fn all(&self, py: Python<'_>) -> PyResult<bool> {
        let inner = self.inner.clone();
        allow(py, move || inner.all())
    }

    fn nnz(&self, py: Python<'_>) -> PyResult<u64> {
        let inner = self.inner.clone();
        allow(py, move || inner.nnz())
    }

    fn num_missing(&self, py: Python<'_>) -> PyResult<u64> {
        let inner = self.inner.clone();
        allow(py, move || inner.num_missing())
    }

    // ── Missing values ──────────────────────────────────────────────

    fn countna(&self, py: Python<'_>) -> PyResult<u64> {
        let inner = self.inner.clone();
        allow(py, move || inner.countna())
    }

    fn dropna(&self) -> PySArray {
        PySArray::new(self.inner.dropna())
    }

    fn fillna(&self, value: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        let val = py_to_flextype(value)?;
        Ok(PySArray::new(self.inner.fillna(val)))
    }

    fn is_na(&self) -> PySArray {
        PySArray::new(self.inner.is_na())
    }

    // ── Type/transform ──────────────────────────────────────────────

    #[pyo3(signature = (dtype, undefined_on_failure=true))]
    fn astype(&self, dtype: &str, undefined_on_failure: bool) -> PyResult<PySArray> {
        let dt = py_str_to_dtype(dtype)?;
        Ok(PySArray::new(self.inner.astype(dt, undefined_on_failure)))
    }

    fn clip(&self, lower: &Bound<'_, PyAny>, upper: &Bound<'_, PyAny>) -> PyResult<PySArray> {
        let lo = py_to_flextype(lower)?;
        let hi = py_to_flextype(upper)?;
        Ok(PySArray::new(self.inner.clip(lo, hi)))
    }

    fn unique(&self, py: Python<'_>) -> PyResult<PySArray> {
        let inner = self.inner.clone();
        let result = allow(py, move || inner.unique())?;
        Ok(PySArray::new(result))
    }

    #[pyo3(signature = (ascending=true))]
    fn sort(&self, ascending: bool, py: Python<'_>) -> PyResult<PySArray> {
        let inner = self.inner.clone();
        let result = allow(py, move || inner.sort(ascending))?;
        Ok(PySArray::new(result))
    }

    fn append(&self, other: &PySArray) -> PyResult<PySArray> {
        let result = self.inner.append(&other.inner).into_pyresult()?;
        Ok(PySArray::new(result))
    }

    #[pyo3(signature = (fraction, seed=None))]
    fn sample(&self, fraction: f64, seed: Option<u64>, py: Python<'_>) -> PyResult<PySArray> {
        let inner = self.inner.clone();
        let result = allow(py, move || inner.sample(fraction, seed))?;
        Ok(PySArray::new(result))
    }

    // ── Approximate ─────────────────────────────────────────────────

    fn approx_count_distinct(&self, py: Python<'_>) -> PyResult<u64> {
        let inner = self.inner.clone();
        allow(py, move || inner.approx_count_distinct())
    }

    fn frequent_items(&self, k: usize, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let items = allow(py, move || inner.frequent_items(k))?;
        let list = PyList::new(
            py,
            items.iter().map(|(val, count)| {
                let py_val = flextype_to_py(py, val);
                let py_count = count.into_pyobject(py).unwrap().into_any().unbind();
                (py_val, py_count)
            }),
        )?;
        Ok(list.into_any().unbind())
    }

    // ── String operations ───────────────────────────────────────────

    fn contains(&self, substring: &str) -> PySArray {
        PySArray::new(self.inner.contains(substring))
    }

    #[pyo3(signature = (to_lower=true))]
    fn count_bag_of_words(&self, to_lower: bool) -> PySArray {
        PySArray::new(self.inner.count_bag_of_words(to_lower))
    }

    #[pyo3(signature = (n=2, to_lower=true))]
    fn count_ngrams(&self, n: usize, to_lower: bool) -> PySArray {
        PySArray::new(self.inner.count_ngrams(n, to_lower))
    }

    #[pyo3(signature = (n=2, to_lower=true))]
    fn count_character_ngrams(&self, n: usize, to_lower: bool) -> PySArray {
        PySArray::new(self.inner.count_character_ngrams(n, to_lower))
    }

    // ── Dict operations ─────────────────────────────────────────────

    fn dict_keys(&self) -> PySArray {
        PySArray::new(self.inner.dict_keys())
    }

    fn dict_values(&self) -> PySArray {
        PySArray::new(self.inner.dict_values())
    }

    fn dict_trim_by_keys(&self, keys: &Bound<'_, PyList>, exclude: bool) -> PyResult<PySArray> {
        let flex_keys: Vec<FlexType> = keys
            .iter()
            .map(|k| py_to_flextype(&k))
            .collect::<PyResult<_>>()?;
        Ok(PySArray::new(self.inner.dict_trim_by_keys(flex_keys, exclude)))
    }

    fn dict_trim_by_values(
        &self,
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
    ) -> PyResult<PySArray> {
        let lo = py_to_flextype(lower)?;
        let hi = py_to_flextype(upper)?;
        Ok(PySArray::new(self.inner.dict_trim_by_values(lo, hi)))
    }

    fn dict_has_any_keys(&self, keys: &Bound<'_, PyList>) -> PyResult<PySArray> {
        let flex_keys: Vec<FlexType> = keys
            .iter()
            .map(|k| py_to_flextype(&k))
            .collect::<PyResult<_>>()?;
        Ok(PySArray::new(self.inner.dict_has_any_keys(flex_keys)))
    }

    fn dict_has_all_keys(&self, keys: &Bound<'_, PyList>) -> PyResult<PySArray> {
        let flex_keys: Vec<FlexType> = keys
            .iter()
            .map(|k| py_to_flextype(&k))
            .collect::<PyResult<_>>()?;
        Ok(PySArray::new(self.inner.dict_has_all_keys(flex_keys)))
    }

    // ── Container operations ────────────────────────────────────────

    fn item_length(&self) -> PySArray {
        PySArray::new(self.inner.item_length())
    }

    #[pyo3(signature = (start, end=None))]
    fn vector_slice(&self, start: usize, end: Option<usize>) -> PySArray {
        PySArray::new(self.inner.vector_slice(start, end))
    }

    // ── Rolling window ──────────────────────────────────────────────

    #[pyo3(signature = (before, after, min_observations=0))]
    fn rolling_sum(&self, before: usize, after: usize, min_observations: usize, py: Python<'_>) -> PyResult<PySArray> {
        let inner = self.inner.clone();
        let result = allow(py, move || inner.rolling_sum(before, after, min_observations))?;
        Ok(PySArray::new(result))
    }

    #[pyo3(signature = (before, after, min_observations=0))]
    fn rolling_mean(&self, before: usize, after: usize, min_observations: usize, py: Python<'_>) -> PyResult<PySArray> {
        let inner = self.inner.clone();
        let result = allow(py, move || inner.rolling_mean(before, after, min_observations))?;
        Ok(PySArray::new(result))
    }

    #[pyo3(signature = (before, after, min_observations=0))]
    fn rolling_min(&self, before: usize, after: usize, min_observations: usize, py: Python<'_>) -> PyResult<PySArray> {
        let inner = self.inner.clone();
        let result = allow(py, move || inner.rolling_min(before, after, min_observations))?;
        Ok(PySArray::new(result))
    }

    #[pyo3(signature = (before, after, min_observations=0))]
    fn rolling_max(&self, before: usize, after: usize, min_observations: usize, py: Python<'_>) -> PyResult<PySArray> {
        let inner = self.inner.clone();
        let result = allow(py, move || inner.rolling_max(before, after, min_observations))?;
        Ok(PySArray::new(result))
    }

    // ── Apply / Filter (Python lambda support) ──────────────────────

    #[pyo3(signature = (func, dtype=None))]
    fn apply(&self, func: PyObject, dtype: Option<&str>) -> PyResult<PySArray> {
        let output_type = match dtype {
            Some(s) => py_str_to_dtype(s)?,
            None => self.inner.dtype(),
        };
        let closure: Arc<dyn Fn(&FlexType) -> FlexType + Send + Sync> =
            Arc::new(move |v: &FlexType| {
                Python::with_gil(|py| {
                    let py_val = flextype_to_py(py, v);
                    match func.call1(py, (py_val,)) {
                        Ok(r) => py_to_flextype(r.bind(py)).unwrap_or(FlexType::Undefined),
                        Err(_) => FlexType::Undefined,
                    }
                })
            });
        Ok(PySArray::new(self.inner.apply(closure, output_type)))
    }

    #[pyo3(name = "filter")]
    fn py_filter(&self, func: PyObject) -> PyResult<PySArray> {
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
        Ok(PySArray::new(self.inner.filter(pred)))
    }

    fn logical_filter(&self, mask: &PySArray) -> PyResult<PySArray> {
        let result = self.inner.logical_filter(&mask.inner).into_pyresult()?;
        Ok(PySArray::new(result))
    }

    // ── Iteration ───────────────────────────────────────────────────

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PySArrayIter> {
        let inner = slf.inner.clone();
        let vals = slf.py().allow_threads(move || inner.to_vec()).into_pyresult()?;
        Ok(PySArrayIter { values: vals, index: 0 })
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyValueError::new_err(
            "The truth value of an SArray is ambiguous. Use a.any() or a.all().",
        ))
    }

    fn __getitem__(&self, index: isize, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let len = allow(py, move || inner.len())? as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx < 0 || idx >= len {
            return Err(PyIndexError::new_err("SArray index out of range"));
        }
        let inner = self.inner.clone();
        let n = idx as usize + 1;
        let vals = allow(py, move || inner.head(n))?;
        match vals.last() {
            Some(v) => Ok(flextype_to_py(py, v)),
            None => Err(PyIndexError::new_err("SArray index out of range")),
        }
    }
}

/// Iterator for PySArray values.
#[pyclass]
pub struct PySArrayIter {
    values: Vec<FlexType>,
    index: usize,
}

#[pymethods]
impl PySArrayIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<PyObject> {
        if self.index < self.values.len() {
            let val = flextype_to_py(py, &self.values[self.index]);
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

// ── Helper types and methods ────────────────────────────────────────

enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

impl PySArray {
    fn binop(&self, other: &Bound<'_, PyAny>, op: BinOp) -> PyResult<PySArray> {
        if let Ok(rhs) = other.extract::<PyRef<PySArray>>() {
            let result = match op {
                BinOp::Add => self.inner.add(&rhs.inner),
                BinOp::Sub => self.inner.sub(&rhs.inner),
                BinOp::Mul => self.inner.mul(&rhs.inner),
                BinOp::Div => self.inner.div(&rhs.inner),
                BinOp::Rem => self.inner.rem(&rhs.inner),
            }
            .into_pyresult()?;
            return Ok(PySArray::new(result));
        }
        let scalar = py_to_flextype(other)?;
        let result = match op {
            BinOp::Add => self.inner.add_scalar(scalar),
            BinOp::Sub => self.inner.sub_scalar(scalar),
            BinOp::Mul => self.inner.mul_scalar(scalar),
            BinOp::Div => self.inner.div_scalar(scalar),
            BinOp::Rem => self.inner.rem_scalar(scalar),
        };
        Ok(PySArray::new(result))
    }

    /// Reverse scalar operations: scalar op self (e.g., 10 - sa)
    fn reverse_scalar_op(&self, other: &Bound<'_, PyAny>, op: BinOp) -> PyResult<PySArray> {
        let scalar = py_to_flextype(other)?;
        let closure: Arc<dyn Fn(&FlexType) -> FlexType + Send + Sync> = match op {
            BinOp::Sub => Arc::new(move |v: &FlexType| match (&scalar, v) {
                (FlexType::Integer(a), FlexType::Integer(b)) => FlexType::Integer(a - b),
                (FlexType::Float(a), FlexType::Float(b)) => FlexType::Float(a - b),
                (FlexType::Integer(a), FlexType::Float(b)) => FlexType::Float(*a as f64 - b),
                (FlexType::Float(a), FlexType::Integer(b)) => FlexType::Float(a - *b as f64),
                _ => FlexType::Undefined,
            }),
            BinOp::Div => Arc::new(move |v: &FlexType| match (&scalar, v) {
                (FlexType::Integer(a), FlexType::Integer(b)) if *b != 0 => {
                    FlexType::Float(*a as f64 / *b as f64)
                }
                (FlexType::Float(a), FlexType::Float(b)) if *b != 0.0 => FlexType::Float(a / b),
                (FlexType::Integer(a), FlexType::Float(b)) if *b != 0.0 => {
                    FlexType::Float(*a as f64 / b)
                }
                (FlexType::Float(a), FlexType::Integer(b)) if *b != 0 => {
                    FlexType::Float(a / *b as f64)
                }
                _ => FlexType::Undefined,
            }),
            BinOp::Rem => Arc::new(move |v: &FlexType| match (&scalar, v) {
                (FlexType::Integer(a), FlexType::Integer(b)) if *b != 0 => {
                    FlexType::Integer(a % b)
                }
                (FlexType::Float(a), FlexType::Float(b)) if *b != 0.0 => FlexType::Float(a % b),
                _ => FlexType::Undefined,
            }),
            // Add/Mul are commutative — shouldn't reach here, but handle gracefully
            BinOp::Add => return self.binop(other, BinOp::Add),
            BinOp::Mul => return self.binop(other, BinOp::Mul),
        };
        let output_type = match self.inner.dtype() {
            FlexTypeEnum::Integer | FlexTypeEnum::Float => self.inner.dtype(),
            _ => FlexTypeEnum::Float,
        };
        Ok(PySArray::new(self.inner.apply(closure, output_type)))
    }
}
