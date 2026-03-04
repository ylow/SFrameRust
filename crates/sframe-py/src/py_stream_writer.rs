use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sframe_core::SFrameStreamWriter;
use sframe_query::batch::ColumnData;
use sframe_query::batch::SFrameRows;
use sframe_types::flex_type::FlexTypeEnum;

use crate::conversion::{py_str_to_dtype, py_to_flextype};
use crate::error::IntoPyResult;

/// Streaming SFrame writer for building SFrames incrementally.
#[pyclass(name = "SFrameStreamWriter", unsendable)]
pub struct PySFrameStreamWriter {
    inner: Option<SFrameStreamWriter>,
    dtypes: Vec<FlexTypeEnum>,
}

#[pymethods]
impl PySFrameStreamWriter {
    #[new]
    fn py_new(path: &str, column_names: Vec<String>, column_types: Vec<String>) -> PyResult<Self> {
        let dtypes: Vec<FlexTypeEnum> = column_types
            .iter()
            .map(|s| py_str_to_dtype(s))
            .collect::<PyResult<_>>()?;

        let name_refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
        let writer = SFrameStreamWriter::new(path, &name_refs, &dtypes).into_pyresult()?;
        Ok(PySFrameStreamWriter {
            inner: Some(writer),
            dtypes,
        })
    }

    /// Write a batch of data. Accepts a dict of {column_name: [values...]}.
    fn write_batch(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Writer already finished"))?;

        let mut columns: Vec<ColumnData> = Vec::new();
        let mut expected_len: Option<usize> = None;

        for (i, (_key, val)) in data.iter().enumerate() {
            let list = val.downcast::<PyList>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("Column values must be lists")
            })?;

            let dtype = self
                .dtypes
                .get(i)
                .copied()
                .unwrap_or(FlexTypeEnum::Undefined);
            let mut col = ColumnData::empty(dtype);

            for item in list.iter() {
                let flex = py_to_flextype(&item)?;
                col.push(&flex).into_pyresult()?;
            }

            let len = list.len();
            if let Some(prev) = expected_len {
                if len != prev {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "All columns must have the same length",
                    ));
                }
            }
            expected_len = Some(len);

            columns.push(col);
        }

        let batch = SFrameRows::new(columns).into_pyresult()?;
        writer.write_batch(&batch).into_pyresult()
    }

    fn set_metadata(&mut self, key: &str, value: &str) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Writer already finished"))?;
        writer.set_metadata(key, value);
        Ok(())
    }

    /// Finalize the SFrame, flushing all data and writing the footer.
    fn finish(&mut self) -> PyResult<()> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Writer already finished"))?;
        writer.finish().into_pyresult()
    }

    // Context manager protocol
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        if self.inner.is_some() {
            self.finish()?;
        }
        Ok(false) // don't suppress exceptions
    }
}
