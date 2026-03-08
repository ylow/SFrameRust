//! Single-pass Sketch computation over an SArray.
//!
//! Implements a composite `SketchAggregator` that goes through the query
//! executor's parallel reduction path. Each rayon worker gets its own
//! clone, reduces locally, and partial results are merged — giving us
//! the same parallelism as sum/mean/etc. but computing *all* statistics
//! in a single pass.

use std::any::Any;
use std::io::{Read, Write};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sframe_core::SArray;
use sframe_query::algorithms::hyperloglog::HyperLogLog;
use sframe_query::algorithms::quantile_sketch::QuantileSketch;
use sframe_query::algorithms::space_saving::SpaceSaving;
use sframe_query::planner::Aggregator;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::conversion::flextype_to_py;
use crate::error::IntoPyResult;
use crate::py_sarray::PySArray;

// ── SketchAggregator ────────────────────────────────────────────────

/// Composite aggregator that computes all sketch statistics in one pass.
///
/// Implements the `Aggregator` trait so it plugs into `SArray::reduce_with`
/// and gets automatic parallel partitioning + merge.
#[derive(Clone)]
pub(crate) struct SketchAggregator {
    count: u64,
    num_undefined: u64,
    sum_int: i64,
    sum_float: f64,
    is_float: bool,
    has_numeric: bool,
    min_val: Option<FlexType>,
    max_val: Option<FlexType>,
    // Welford online variance
    welford_count: u64,
    welford_mean: f64,
    welford_m2: f64,
    // Approximate sketches
    hll: HyperLogLog,
    space_saving: SpaceSaving,
    quantile_sketch: QuantileSketch,
}

impl SketchAggregator {
    pub fn new() -> Self {
        SketchAggregator {
            count: 0,
            num_undefined: 0,
            sum_int: 0,
            sum_float: 0.0,
            is_float: false,
            has_numeric: false,
            min_val: None,
            max_val: None,
            welford_count: 0,
            welford_mean: 0.0,
            welford_m2: 0.0,
            hll: HyperLogLog::new(12),
            space_saving: SpaceSaving::new(512),
            quantile_sketch: QuantileSketch::new(0.005),
        }
    }

    fn ingest(&mut self, val: &FlexType) {
        self.count += 1;

        if matches!(val, FlexType::Undefined) {
            self.num_undefined += 1;
            return;
        }

        self.hll.insert(val);
        self.space_saving.insert(val);

        let fval = match val {
            FlexType::Integer(i) => {
                self.sum_int += i;
                self.has_numeric = true;
                *i as f64
            }
            FlexType::Float(f) => {
                self.sum_float += f;
                self.is_float = true;
                self.has_numeric = true;
                *f
            }
            _ => return,
        };

        // Min/max
        match &self.min_val {
            None => self.min_val = Some(val.clone()),
            Some(cur) if flex_lt(val, cur) => self.min_val = Some(val.clone()),
            _ => {}
        }
        match &self.max_val {
            None => self.max_val = Some(val.clone()),
            Some(cur) if flex_lt(cur, val) => self.max_val = Some(val.clone()),
            _ => {}
        }

        // Welford online variance
        self.welford_count += 1;
        let delta = fval - self.welford_mean;
        self.welford_mean += delta / self.welford_count as f64;
        let delta2 = fval - self.welford_mean;
        self.welford_m2 += delta * delta2;

        self.quantile_sketch.insert(val.clone());
    }

    /// Merge another SketchAggregator into this one.
    fn merge_sketch(&mut self, other: &SketchAggregator) {
        self.count += other.count;
        self.num_undefined += other.num_undefined;
        self.sum_int += other.sum_int;
        self.sum_float += other.sum_float;
        self.is_float |= other.is_float;
        self.has_numeric |= other.has_numeric;

        // Min/max
        if let Some(ref other_min) = other.min_val {
            match &self.min_val {
                None => self.min_val = Some(other_min.clone()),
                Some(cur) if flex_lt(other_min, cur) => {
                    self.min_val = Some(other_min.clone());
                }
                _ => {}
            }
        }
        if let Some(ref other_max) = other.max_val {
            match &self.max_val {
                None => self.max_val = Some(other_max.clone()),
                Some(cur) if flex_lt(cur, other_max) => {
                    self.max_val = Some(other_max.clone());
                }
                _ => {}
            }
        }

        // Welford parallel merge (Chan et al.)
        if other.welford_count > 0 {
            let n_a = self.welford_count as f64;
            let n_b = other.welford_count as f64;
            let n_ab = n_a + n_b;
            if n_ab > 0.0 {
                let delta = other.welford_mean - self.welford_mean;
                self.welford_m2 += other.welford_m2 + delta * delta * n_a * n_b / n_ab;
                self.welford_mean = (n_a * self.welford_mean + n_b * other.welford_mean) / n_ab;
            }
            self.welford_count += other.welford_count;
        }

        // Approximate sketches
        self.hll.merge(&other.hll);
        self.space_saving.merge(&other.space_saving);
        self.quantile_sketch.merge(&other.quantile_sketch);
    }

    pub fn variance(&self) -> f64 {
        if self.welford_count < 2 {
            return 0.0;
        }
        self.welford_m2 / (self.welford_count - 1) as f64
    }
}

fn flex_lt(a: &FlexType, b: &FlexType) -> bool {
    match (a, b) {
        (FlexType::Integer(x), FlexType::Integer(y)) => x < y,
        (FlexType::Float(x), FlexType::Float(y)) => x < y,
        (FlexType::Integer(x), FlexType::Float(y)) => (*x as f64) < *y,
        (FlexType::Float(x), FlexType::Integer(y)) => *x < (*y as f64),
        _ => false,
    }
}

impl Aggregator for SketchAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if let Some(val) = values.first() {
            self.ingest(val);
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(other) = other.as_any().downcast_ref::<SketchAggregator>() {
            self.merge_sketch(other);
        }
    }

    fn finalize(&mut self) -> FlexType {
        self.quantile_sketch.finish();
        // The Sketch aggregator doesn't reduce to a single FlexType;
        // we use a sentinel. The real data is extracted via from_aggregator.
        FlexType::Integer(0)
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Integer
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(SketchAggregator::new())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, _writer: &mut dyn Write) -> sframe_types::error::Result<()> {
        Err(sframe_types::error::SFrameError::Format("Sketch aggregator does not support serialization".into()))
    }

    fn load(&mut self, _reader: &mut dyn Read) -> sframe_types::error::Result<()> {
        Err(sframe_types::error::SFrameError::Format("Sketch aggregator does not support deserialization".into()))
    }
}

// ── compute_sketch (parallel via reduce_with) ───────────────────────

/// Compute a Sketch from an SArray using the parallel query executor.
///
/// Uses `SArray::reduce_aggregate` which partitions rows across rayon
/// workers, each running a local `SketchAggregator`, then merges all
/// partials — the same parallel reduction path as sum/mean/etc.
pub(crate) fn compute_sketch(sa: &SArray) -> sframe_types::error::Result<SketchAggregator> {
    let mut result = sa.reduce_aggregate(SketchAggregator::new())?;
    result.quantile_sketch.finish();
    Ok(result)
}

// ── Python-facing Sketch class ──────────────────────────────────────

/// Python-facing Sketch object holding pre-computed statistics.
#[pyclass(name = "Sketch")]
pub struct PySketch {
    data: SketchAggregator,
}

impl PySketch {
    pub(crate) fn from_data(data: SketchAggregator) -> Self {
        PySketch { data }
    }
}

#[pymethods]
impl PySketch {
    /// Construct a Sketch from an SArray.
    #[new]
    #[pyo3(signature = (array, background=false))]
    fn py_new(array: &PySArray, background: bool, py: Python<'_>) -> PyResult<Self> {
        if background {
            return Err(PyValueError::new_err(
                "background=True is not supported; Sketch is computed synchronously",
            ));
        }
        let inner = array.inner.clone();
        let data = py.detach(move || compute_sketch(&inner)).into_pyresult()?;
        Ok(PySketch { data })
    }

    fn size(&self) -> u64 {
        self.data.count
    }

    fn num_undefined(&self) -> u64 {
        self.data.num_undefined
    }

    fn min(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.data.min_val {
            Some(v) => flextype_to_py(py, v),
            None => py.None().into_pyobject(py).unwrap().unbind(),
        }
    }

    fn max(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.data.max_val {
            Some(v) => flextype_to_py(py, v),
            None => py.None().into_pyobject(py).unwrap().unbind(),
        }
    }

    fn sum(&self, py: Python<'_>) -> Py<PyAny> {
        if !self.data.has_numeric {
            return py.None().into_pyobject(py).unwrap().unbind();
        }
        if self.data.is_float {
            let total = self.data.sum_int as f64 + self.data.sum_float;
            flextype_to_py(py, &FlexType::Float(total))
        } else {
            flextype_to_py(py, &FlexType::Integer(self.data.sum_int))
        }
    }

    fn mean(&self) -> f64 {
        self.data.welford_mean
    }

    fn var(&self) -> f64 {
        self.data.variance()
    }

    fn std(&self) -> f64 {
        self.data.variance().sqrt()
    }

    fn num_unique(&self) -> u64 {
        self.data.hll.estimate()
    }

    fn frequent_items(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let items = self.data.space_saving.top_k(128);
        let dict = PyDict::new(py);
        for (val, count) in &items {
            let py_val = flextype_to_py(py, val);
            dict.set_item(py_val, count)?;
        }
        Ok(dict.into_any().unbind())
    }

    fn quantile(&self, quantile_val: f64, py: Python<'_>) -> Py<PyAny> {
        let val = self.data.quantile_sketch.query(quantile_val);
        flextype_to_py(py, &val)
    }

    fn frequency_count(&self, element: &Bound<'_, PyAny>) -> PyResult<u64> {
        let val = crate::conversion::py_to_flextype(element)?;
        let items = self.data.space_saving.top_k(self.data.space_saving.len());
        for (item, count) in &items {
            if *item == val {
                return Ok(*count);
            }
        }
        Ok(0)
    }

    fn __repr__(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "+--------------------+{:-^20}+----------+",
            ""
        ));
        lines.push(format!(
            "|{:<20}|{:^20}|{:^10}|",
            "        item", "value", "is exact"
        ));
        lines.push(format!(
            "+--------------------+{:-^20}+----------+",
            ""
        ));

        let rows: Vec<(&str, String, &str)> = vec![
            ("Length", format!("{}", self.data.count), "Yes"),
            ("Min", {
                match &self.data.min_val {
                    Some(v) => format!("{v}"),
                    None => "None".to_string(),
                }
            }, "Yes"),
            ("Max", {
                match &self.data.max_val {
                    Some(v) => format!("{v}"),
                    None => "None".to_string(),
                }
            }, "Yes"),
            ("Mean", format!("{}", self.data.welford_mean), "Yes"),
            ("Sum", {
                if self.data.is_float {
                    format!("{}", self.data.sum_int as f64 + self.data.sum_float)
                } else {
                    format!("{}", self.data.sum_int)
                }
            }, "Yes"),
            (
                "Variance",
                format!("{}", self.data.variance()),
                "Yes",
            ),
            (
                "Standard Deviation",
                format!("{}", self.data.variance().sqrt()),
                "Yes",
            ),
            (
                "# Missing Values",
                format!("{}", self.data.num_undefined),
                "Yes",
            ),
            (
                "# unique values",
                format!("{}", self.data.hll.estimate()),
                "No",
            ),
        ];

        for (label, value, exact) in &rows {
            lines.push(format!(
                "| {:<18} | {:>18} | {:^8} |",
                label, value, exact
            ));
        }
        lines.push(format!(
            "+--------------------+{:-^20}+----------+",
            ""
        ));

        // Frequent items
        let freq = self.data.space_saving.top_k(10);
        if freq.is_empty() {
            lines.push("\nMost frequent items:".to_string());
            lines.push(" -- All elements appear with less than 0.01% frequency --".to_string());
        } else {
            lines.push("\nMost frequent items:".to_string());
            let mut header = "| value   |".to_string();
            let mut counts = "| count   |".to_string();
            for (val, count) in &freq {
                let s = format!("{val}");
                let width = s.len().max(5);
                header.push_str(&format!(" {:^w$} |", s, w = width));
                counts.push_str(&format!(" {:^w$} |", count, w = width));
            }
            lines.push(header);
            lines.push(counts);
        }

        // Quantiles (only if numeric)
        if self.data.has_numeric {
            lines.push("\nQuantiles:".to_string());
            let quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0];
            let labels = ["0%", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "100%"];
            let mut header = String::new();
            let mut values = String::new();
            for (q, label) in quantiles.iter().zip(labels.iter()) {
                let val = self.data.quantile_sketch.query(*q);
                let s = format!("{val}");
                let width = s.len().max(label.len()).max(4);
                header.push_str(&format!("| {:^w$} ", label, w = width));
                values.push_str(&format!("| {:^w$} ", s, w = width));
            }
            header.push('|');
            values.push('|');
            lines.push(header);
            lines.push(values);
        }

        lines.join("\n")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}
