use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::PyErr;
use sframe_types::error::SFrameError;

/// Convert an SFrameError into a PyErr.
pub fn to_py_err(e: SFrameError) -> PyErr {
    match &e {
        SFrameError::Io(_) => PyIOError::new_err(e.to_string()),
        SFrameError::Type(_) => PyTypeError::new_err(e.to_string()),
        SFrameError::Format(_) => PyValueError::new_err(e.to_string()),
    }
}

/// Extension trait for converting `Result<T, SFrameError>` to `PyResult<T>`.
pub trait IntoPyResult<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T>;
}

impl<T> IntoPyResult<T> for sframe_types::error::Result<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T> {
        self.map_err(to_py_err)
    }
}
