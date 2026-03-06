use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

/// Convert a FlexType to a Python object.
pub fn flextype_to_py(py: Python<'_>, val: &FlexType) -> PyObject {
    match val {
        FlexType::Integer(i) => i.into_pyobject(py).unwrap().into_any().unbind(),
        FlexType::Float(f) => f.into_pyobject(py).unwrap().into_any().unbind(),
        FlexType::String(s) => s.as_ref().into_pyobject(py).unwrap().into_any().unbind(),
        FlexType::Vector(v) => {
            let list = PyList::new(py, v.iter()).unwrap();
            list.into_any().unbind()
        }
        FlexType::List(items) => {
            let py_items: Vec<PyObject> = items.iter().map(|x| flextype_to_py(py, x)).collect();
            let list = PyList::new(py, &py_items).unwrap();
            list.into_any().unbind()
        }
        FlexType::Dict(pairs) => {
            let dict = PyDict::new(py);
            for (k, v) in pairs.iter() {
                let pk = flextype_to_py(py, k);
                let pv = flextype_to_py(py, v);
                dict.set_item(pk, pv).unwrap();
            }
            dict.into_any().unbind()
        }
        FlexType::DateTime(dt) => {
            // Return as a float (posix timestamp + microseconds)
            let ts = dt.posix_timestamp as f64 + dt.microsecond as f64 / 1_000_000.0;
            ts.into_pyobject(py).unwrap().into_any().unbind()
        }
        FlexType::Undefined => py.None(),
    }
}

/// Convert a Python object to a FlexType.
pub fn py_to_flextype(obj: &Bound<'_, PyAny>) -> PyResult<FlexType> {
    // None
    if obj.is_none() {
        return Ok(FlexType::Undefined);
    }
    // Bool must be checked before int (bool is a subclass of int in Python)
    if let Ok(b) = obj.downcast::<PyBool>() {
        return Ok(FlexType::Integer(if b.is_true() { 1 } else { 0 }));
    }
    // Int
    if let Ok(i) = obj.downcast::<PyInt>() {
        let val: i64 = i.extract()?;
        return Ok(FlexType::Integer(val));
    }
    // Float
    if let Ok(f) = obj.downcast::<PyFloat>() {
        let val: f64 = f.extract()?;
        return Ok(FlexType::Float(val));
    }
    // String
    if let Ok(s) = obj.downcast::<PyString>() {
        let val: String = s.extract()?;
        return Ok(FlexType::String(Arc::from(val.as_str())));
    }
    // Dict
    if let Ok(d) = obj.downcast::<PyDict>() {
        let mut pairs = Vec::new();
        for (k, v) in d.iter() {
            pairs.push((py_to_flextype(&k)?, py_to_flextype(&v)?));
        }
        return Ok(FlexType::Dict(Arc::from(pairs)));
    }
    // List
    if let Ok(l) = obj.downcast::<PyList>() {
        // Check if all elements are numeric -> Vector
        let items: Vec<Bound<'_, PyAny>> = l.iter().collect();
        let all_numeric = !items.is_empty()
            && items.iter().all(|item| {
                item.downcast::<PyFloat>().is_ok() || item.downcast::<PyInt>().is_ok()
            });
        if all_numeric {
            let floats: Vec<f64> = items
                .iter()
                .map(|item| item.extract::<f64>())
                .collect::<PyResult<_>>()?;
            return Ok(FlexType::Vector(Arc::from(floats)));
        }
        // Mixed list
        let flex_items: Vec<FlexType> = items
            .iter()
            .map(|item| py_to_flextype(item))
            .collect::<PyResult<_>>()?;
        return Ok(FlexType::List(Arc::from(flex_items)));
    }
    // Fallback: try to convert to string
    let s: String = obj.str()?.extract()?;
    Ok(FlexType::String(Arc::from(s.as_str())))
}

/// Convert a Python string to a FlexTypeEnum.
pub fn py_str_to_dtype(s: &str) -> PyResult<FlexTypeEnum> {
    match s {
        "int" | "integer" => Ok(FlexTypeEnum::Integer),
        "float" => Ok(FlexTypeEnum::Float),
        "str" | "string" => Ok(FlexTypeEnum::String),
        "vector" | "array" => Ok(FlexTypeEnum::Vector),
        "list" => Ok(FlexTypeEnum::List),
        "dict" => Ok(FlexTypeEnum::Dict),
        "datetime" => Ok(FlexTypeEnum::DateTime),
        "undefined" => Ok(FlexTypeEnum::Undefined),
        _ => Err(PyValueError::new_err(format!("Unknown dtype: '{s}'"))),
    }
}

/// Convert a FlexTypeEnum to a Python string.
pub fn dtype_to_py_str(dt: FlexTypeEnum) -> &'static str {
    match dt {
        FlexTypeEnum::Integer => "int",
        FlexTypeEnum::Float => "float",
        FlexTypeEnum::String => "str",
        FlexTypeEnum::Vector => "vector",
        FlexTypeEnum::List => "list",
        FlexTypeEnum::Dict => "dict",
        FlexTypeEnum::DateTime => "datetime",
        FlexTypeEnum::Undefined => "undefined",
    }
}
