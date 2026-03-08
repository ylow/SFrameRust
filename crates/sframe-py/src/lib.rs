#![allow(non_snake_case)]

use pyo3::prelude::*;

mod conversion;
mod error;
mod py_config;
mod py_sarray;
mod py_sframe;
mod py_stream_writer;

use error::IntoPyResult;
use py_config::PyConfig;
use py_sarray::{PySArray, PySArrayIter};
use py_sframe::{PySFrame, PySFrameIter};
use py_stream_writer::PySFrameStreamWriter;

use pyo3::exceptions::PyValueError;
use sframe_query::algorithms::aggregators::{
    ConcatAggregator, CountAggregator, CountDistinctAggregator, MaxAggregator, MeanAggregator,
    MinAggregator, SelectOneAggregator, StdDevAggregator, SumAggregator, VarianceAggregator,
};
use sframe_query::planner::Aggregator;

/// Specification for an aggregation operation used in groupby.
#[pyclass(name = "AggSpec")]
pub struct PyAggSpec {
    pub(crate) column: String,
    pub(crate) op: String,
}

impl PyAggSpec {
    pub fn make_aggregator(&self) -> PyResult<Box<dyn Aggregator>> {
        match self.op.as_str() {
            "SUM" => Ok(Box::new(SumAggregator::new())),
            "MEAN" => Ok(Box::new(MeanAggregator::new())),
            "MIN" => Ok(Box::new(MinAggregator::new())),
            "MAX" => Ok(Box::new(MaxAggregator::new())),
            "COUNT" => Ok(Box::new(CountAggregator::new())),
            "VARIANCE" => Ok(Box::new(VarianceAggregator::sample())),
            "STD" => Ok(Box::new(StdDevAggregator::sample())),
            "COUNT_DISTINCT" => Ok(Box::new(CountDistinctAggregator::new())),
            "CONCAT" => Ok(Box::new(ConcatAggregator::new())),
            "SELECT_ONE" => Ok(Box::new(SelectOneAggregator::new())),
            _ => Err(PyValueError::new_err(format!(
                "Unknown aggregation: '{}'",
                self.op
            ))),
        }
    }
}

#[pymethods]
impl PyAggSpec {
    fn __repr__(&self) -> String {
        format!("aggregate.{}('{}')", self.op, self.column)
    }
}

// ── Aggregate module-level factory functions ────────────────────────

#[pyfunction]
#[allow(non_snake_case)]
fn SUM(col: &str) -> PyAggSpec {
    PyAggSpec {
        column: col.to_string(),
        op: "SUM".to_string(),
    }
}

#[pyfunction]
#[allow(non_snake_case)]
fn MEAN(col: &str) -> PyAggSpec {
    PyAggSpec {
        column: col.to_string(),
        op: "MEAN".to_string(),
    }
}

#[pyfunction]
#[allow(non_snake_case)]
fn MIN(col: &str) -> PyAggSpec {
    PyAggSpec {
        column: col.to_string(),
        op: "MIN".to_string(),
    }
}

#[pyfunction]
#[allow(non_snake_case)]
fn MAX(col: &str) -> PyAggSpec {
    PyAggSpec {
        column: col.to_string(),
        op: "MAX".to_string(),
    }
}

#[pyfunction]
#[pyo3(signature = (col=None))]
#[allow(non_snake_case)]
fn COUNT(col: Option<&str>) -> PyAggSpec {
    PyAggSpec {
        column: col.unwrap_or("").to_string(),
        op: "COUNT".to_string(),
    }
}

#[pyfunction]
#[allow(non_snake_case)]
fn VARIANCE(col: &str) -> PyAggSpec {
    PyAggSpec {
        column: col.to_string(),
        op: "VARIANCE".to_string(),
    }
}

#[pyfunction]
#[allow(non_snake_case)]
fn STD(col: &str) -> PyAggSpec {
    PyAggSpec {
        column: col.to_string(),
        op: "STD".to_string(),
    }
}

#[pyfunction]
#[allow(non_snake_case)]
fn COUNT_DISTINCT(col: &str) -> PyAggSpec {
    PyAggSpec {
        column: col.to_string(),
        op: "COUNT_DISTINCT".to_string(),
    }
}

#[pyfunction]
#[allow(non_snake_case)]
fn CONCAT(col: &str) -> PyAggSpec {
    PyAggSpec {
        column: col.to_string(),
        op: "CONCAT".to_string(),
    }
}

#[pyfunction]
#[allow(non_snake_case)]
fn SELECT_ONE(col: &str) -> PyAggSpec {
    PyAggSpec {
        column: col.to_string(),
        op: "SELECT_ONE".to_string(),
    }
}

// ── Top-level convenience ───────────────────────────────────────────

/// Load an SFrame from a path (convenience function).
#[pyfunction]
fn load(path: &str) -> PyResult<PySFrame> {
    let sf = sframe_core::SFrame::read(path).into_pyresult()?;
    Ok(PySFrame::new(sf))
}

// ── Module registration ─────────────────────────────────────────────

#[pymodule]
fn _sframe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySFrame>()?;
    m.add_class::<PySArray>()?;
    m.add_class::<PySFrameStreamWriter>()?;
    m.add_class::<PySArrayIter>()?;
    m.add_class::<PySFrameIter>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add("config", PyConfig)?;

    // aggregate submodule
    let aggregate = PyModule::new(m.py(), "aggregate")?;
    aggregate.add_class::<PyAggSpec>()?;
    aggregate.add_function(wrap_pyfunction!(SUM, &aggregate)?)?;
    aggregate.add_function(wrap_pyfunction!(MEAN, &aggregate)?)?;
    aggregate.add_function(wrap_pyfunction!(MIN, &aggregate)?)?;
    aggregate.add_function(wrap_pyfunction!(MAX, &aggregate)?)?;
    aggregate.add_function(wrap_pyfunction!(COUNT, &aggregate)?)?;
    aggregate.add_function(wrap_pyfunction!(VARIANCE, &aggregate)?)?;
    aggregate.add_function(wrap_pyfunction!(STD, &aggregate)?)?;
    aggregate.add_function(wrap_pyfunction!(COUNT_DISTINCT, &aggregate)?)?;
    aggregate.add_function(wrap_pyfunction!(CONCAT, &aggregate)?)?;
    aggregate.add_function(wrap_pyfunction!(SELECT_ONE, &aggregate)?)?;
    m.add_submodule(&aggregate)?;

    Ok(())
}
