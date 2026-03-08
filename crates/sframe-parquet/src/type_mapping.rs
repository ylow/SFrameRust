// Type mapping between Arrow/Parquet types and SFrame types.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Date64Array, Float16Array, Float32Array,
    Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, LargeListArray, LargeStringArray,
    ListArray, StringArray, StructArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::datatypes::{DataType, Field, TimeUnit};
use arrow::record_batch::RecordBatch;

use sframe_query::algorithms::json::flex_to_json;
use sframe_query::batch::{ColumnData, SFrameRows};
use sframe_query::nullable_vec::NullableVec;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexDateTime, FlexType, FlexTypeEnum};
use sframe_types::flex_wrappers::{FlexDict, FlexList, FlexString, FlexVec};

// ---------------------------------------------------------------------------
// Task 2: Arrow DataType -> FlexTypeEnum
// ---------------------------------------------------------------------------

/// Map an Arrow data type to the corresponding SFrame type enum.
pub fn arrow_type_to_sframe(dt: &DataType) -> Result<FlexTypeEnum> {
    match dt {
        // Integer types
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Boolean => Ok(FlexTypeEnum::Integer),

        // Float types
        DataType::Float16 | DataType::Float32 | DataType::Float64 => Ok(FlexTypeEnum::Float),

        // String types
        DataType::Utf8 | DataType::LargeUtf8 => Ok(FlexTypeEnum::String),

        // DateTime types
        DataType::Timestamp(_, _) | DataType::Date32 | DataType::Date64 => {
            Ok(FlexTypeEnum::DateTime)
        }

        // List types
        DataType::List(field) => {
            if matches!(field.data_type(), DataType::Float64) {
                Ok(FlexTypeEnum::Vector)
            } else {
                Ok(FlexTypeEnum::List)
            }
        }
        DataType::LargeList(_) => Ok(FlexTypeEnum::List),

        // Dict types
        DataType::Struct(_) | DataType::Map(_, _) => Ok(FlexTypeEnum::Dict),

        other => Err(SFrameError::Type(format!(
            "Unsupported Arrow type: {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Task 3: Arrow ArrayRef -> ColumnData
// ---------------------------------------------------------------------------

/// Convert an Arrow array to SFrame ColumnData according to the target SFrame type.
pub fn arrow_array_to_column(array: &dyn Array, sframe_type: FlexTypeEnum) -> Result<ColumnData> {
    match sframe_type {
        FlexTypeEnum::Integer => convert_to_integer(array),
        FlexTypeEnum::Float => convert_to_float(array),
        FlexTypeEnum::String => convert_to_string(array),
        FlexTypeEnum::DateTime => convert_to_datetime(array),
        FlexTypeEnum::Vector => convert_to_vector(array),
        FlexTypeEnum::List => convert_to_list(array),
        FlexTypeEnum::Dict => convert_to_dict(array),
        FlexTypeEnum::Undefined => Err(SFrameError::Type(
            "Cannot convert Arrow array to Undefined column".to_string(),
        )),
    }
}

/// Helper macro to convert a primitive Arrow array to NullableVec<i64>.
macro_rules! cast_int_array {
    ($array:expr, $arrow_type:ty) => {{
        let arr = $array
            .as_any()
            .downcast_ref::<$arrow_type>()
            .ok_or_else(|| SFrameError::Type(format!("Expected {} array", stringify!($arrow_type))))?;
        let mut out = NullableVec::with_capacity(arr.len());
        for i in 0..arr.len() {
            if arr.is_null(i) {
                out.push(None);
            } else {
                out.push(Some(arr.value(i) as i64));
            }
        }
        Ok(ColumnData::Integer(out))
    }};
}

fn convert_to_integer(array: &dyn Array) -> Result<ColumnData> {
    match array.data_type() {
        DataType::Int8 => cast_int_array!(array, Int8Array),
        DataType::Int16 => cast_int_array!(array, Int16Array),
        DataType::Int32 => cast_int_array!(array, Int32Array),
        DataType::Int64 => cast_int_array!(array, Int64Array),
        DataType::UInt8 => cast_int_array!(array, UInt8Array),
        DataType::UInt16 => cast_int_array!(array, UInt16Array),
        DataType::UInt32 => cast_int_array!(array, UInt32Array),
        DataType::UInt64 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| SFrameError::Type("Expected UInt64Array".to_string()))?;
            let mut out = NullableVec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    let v = arr.value(i);
                    if v > i64::MAX as u64 {
                        return Err(SFrameError::Type(format!(
                            "UInt64 value {v} exceeds i64::MAX"
                        )));
                    }
                    out.push(Some(v as i64));
                }
            }
            Ok(ColumnData::Integer(out))
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| SFrameError::Type("Expected BooleanArray".to_string()))?;
            let mut out = NullableVec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(if arr.value(i) { 1 } else { 0 }));
                }
            }
            Ok(ColumnData::Integer(out))
        }
        other => Err(SFrameError::Type(format!(
            "Cannot convert {other:?} to Integer"
        ))),
    }
}

fn convert_to_float(array: &dyn Array) -> Result<ColumnData> {
    match array.data_type() {
        DataType::Float16 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float16Array>()
                .ok_or_else(|| SFrameError::Type("Expected Float16Array".to_string()))?;
            let mut out = NullableVec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i).to_f64()));
                }
            }
            Ok(ColumnData::Float(out))
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| SFrameError::Type("Expected Float32Array".to_string()))?;
            let mut out = NullableVec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i) as f64));
                }
            }
            Ok(ColumnData::Float(out))
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| SFrameError::Type("Expected Float64Array".to_string()))?;
            let mut out = NullableVec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i)));
                }
            }
            Ok(ColumnData::Float(out))
        }
        other => Err(SFrameError::Type(format!(
            "Cannot convert {other:?} to Float"
        ))),
    }
}

fn convert_to_string(array: &dyn Array) -> Result<ColumnData> {
    if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
        let mut out = NullableVec::with_capacity(arr.len());
        for i in 0..arr.len() {
            if arr.is_null(i) {
                out.push(None);
            } else {
                out.push(Some(FlexString::from(arr.value(i))));
            }
        }
        Ok(ColumnData::String(out))
    } else if let Some(arr) = array.as_any().downcast_ref::<LargeStringArray>() {
        let mut out = NullableVec::with_capacity(arr.len());
        for i in 0..arr.len() {
            if arr.is_null(i) {
                out.push(None);
            } else {
                out.push(Some(FlexString::from(arr.value(i))));
            }
        }
        Ok(ColumnData::String(out))
    } else {
        Err(SFrameError::Type(format!(
            "Cannot convert {:?} to String",
            array.data_type()
        )))
    }
}

/// Convert timestamp/date Arrow arrays to FlexDateTime column.
///
/// FlexDateTime stores:
/// - posix_timestamp: i64 (seconds since epoch)
/// - tz_offset_quarter_hours: i8
/// - microsecond: u32 (sub-second microseconds)
fn convert_to_datetime(array: &dyn Array) -> Result<ColumnData> {
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Second, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampSecondArray>()
                .ok_or_else(|| SFrameError::Type("Expected TimestampSecondArray".to_string()))?;
            let out = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let secs = arr.value(i);
                        Some(FlexDateTime {
                            posix_timestamp: secs,
                            tz_offset_quarter_hours: 0,
                            microsecond: 0,
                        })
                    }
                })
                .collect();
            Ok(ColumnData::DateTime(out))
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .ok_or_else(|| {
                    SFrameError::Type("Expected TimestampMillisecondArray".to_string())
                })?;
            let out = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let ms = arr.value(i);
                        let secs = ms.div_euclid(1_000);
                        let remainder_us = (ms.rem_euclid(1_000) * 1_000) as u32;
                        Some(FlexDateTime {
                            posix_timestamp: secs,
                            tz_offset_quarter_hours: 0,
                            microsecond: remainder_us,
                        })
                    }
                })
                .collect();
            Ok(ColumnData::DateTime(out))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| {
                    SFrameError::Type("Expected TimestampMicrosecondArray".to_string())
                })?;
            let out = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let us = arr.value(i);
                        let secs = us.div_euclid(1_000_000);
                        let remainder_us = us.rem_euclid(1_000_000) as u32;
                        Some(FlexDateTime {
                            posix_timestamp: secs,
                            tz_offset_quarter_hours: 0,
                            microsecond: remainder_us,
                        })
                    }
                })
                .collect();
            Ok(ColumnData::DateTime(out))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .ok_or_else(|| {
                    SFrameError::Type("Expected TimestampNanosecondArray".to_string())
                })?;
            let out = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let ns = arr.value(i);
                        let us = ns / 1_000; // truncate nanoseconds to microseconds
                        let secs = us.div_euclid(1_000_000);
                        let remainder_us = us.rem_euclid(1_000_000) as u32;
                        Some(FlexDateTime {
                            posix_timestamp: secs,
                            tz_offset_quarter_hours: 0,
                            microsecond: remainder_us,
                        })
                    }
                })
                .collect();
            Ok(ColumnData::DateTime(out))
        }
        DataType::Date32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| SFrameError::Type("Expected Date32Array".to_string()))?;
            let out = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let days = arr.value(i) as i64;
                        Some(FlexDateTime {
                            posix_timestamp: days * 86400,
                            tz_offset_quarter_hours: 0,
                            microsecond: 0,
                        })
                    }
                })
                .collect();
            Ok(ColumnData::DateTime(out))
        }
        DataType::Date64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Date64Array>()
                .ok_or_else(|| SFrameError::Type("Expected Date64Array".to_string()))?;
            let out = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        let ms = arr.value(i);
                        let secs = ms.div_euclid(1_000);
                        let remainder_us = (ms.rem_euclid(1_000) * 1_000) as u32;
                        Some(FlexDateTime {
                            posix_timestamp: secs,
                            tz_offset_quarter_hours: 0,
                            microsecond: remainder_us,
                        })
                    }
                })
                .collect();
            Ok(ColumnData::DateTime(out))
        }
        other => Err(SFrameError::Type(format!(
            "Cannot convert {other:?} to DateTime"
        ))),
    }
}

fn convert_to_vector(array: &dyn Array) -> Result<ColumnData> {
    let list_arr = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| SFrameError::Type("Expected ListArray for Vector conversion".to_string()))?;

    let values = list_arr
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            SFrameError::Type("Expected Float64 inner array for Vector conversion".to_string())
        })?;

    let mut out = NullableVec::with_capacity(list_arr.len());
    for i in 0..list_arr.len() {
        if list_arr.is_null(i) {
            out.push(None);
        } else {
            let start = list_arr.value_offsets()[i] as usize;
            let end = list_arr.value_offsets()[i + 1] as usize;
            let slice: Vec<f64> = (start..end).map(|j| values.value(j)).collect();
            out.push(Some(FlexVec::from(slice)));
        }
    }
    Ok(ColumnData::Vector(out))
}

fn convert_to_list(array: &dyn Array) -> Result<ColumnData> {
    if let Some(list_arr) = array.as_any().downcast_ref::<ListArray>() {
        convert_list_array_to_list(list_arr)
    } else if let Some(list_arr) = array.as_any().downcast_ref::<LargeListArray>() {
        convert_large_list_array_to_list(list_arr)
    } else {
        Err(SFrameError::Type(format!(
            "Expected ListArray or LargeListArray, got {:?}",
            array.data_type()
        )))
    }
}

fn convert_list_array_to_list(list_arr: &ListArray) -> Result<ColumnData> {
    let inner_type = list_arr.value_type();
    let inner_sframe_type = arrow_type_to_sframe(&inner_type)?;
    let values = list_arr.values();

    let mut out = NullableVec::with_capacity(list_arr.len());
    for i in 0..list_arr.len() {
        if list_arr.is_null(i) {
            out.push(None);
        } else {
            let start = list_arr.value_offsets()[i] as usize;
            let end = list_arr.value_offsets()[i + 1] as usize;
            let len = end - start;
            let slice = values.slice(start, len);
            let inner_col = arrow_array_to_column(slice.as_ref(), inner_sframe_type)?;
            let flex_vec = inner_col.to_flex_vec();
            out.push(Some(FlexList::from(flex_vec)));
        }
    }
    Ok(ColumnData::List(out))
}

fn convert_large_list_array_to_list(list_arr: &LargeListArray) -> Result<ColumnData> {
    let inner_type = list_arr.value_type();
    let inner_sframe_type = arrow_type_to_sframe(&inner_type)?;
    let values = list_arr.values();

    let mut out = NullableVec::with_capacity(list_arr.len());
    for i in 0..list_arr.len() {
        if list_arr.is_null(i) {
            out.push(None);
        } else {
            let start = list_arr.value_offsets()[i] as usize;
            let end = list_arr.value_offsets()[i + 1] as usize;
            let len = end - start;
            let slice = values.slice(start, len);
            let inner_col = arrow_array_to_column(slice.as_ref(), inner_sframe_type)?;
            let flex_vec = inner_col.to_flex_vec();
            out.push(Some(FlexList::from(flex_vec)));
        }
    }
    Ok(ColumnData::List(out))
}

fn convert_to_dict(array: &dyn Array) -> Result<ColumnData> {
    let struct_arr = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| SFrameError::Type("Expected StructArray for Dict conversion".to_string()))?;

    let fields = struct_arr.fields();
    let num_fields = fields.len();

    // For each field, convert the column to ColumnData so we can extract values
    let mut field_cols: Vec<(String, ColumnData)> = Vec::with_capacity(num_fields);
    for (idx, field) in fields.iter().enumerate() {
        let field_sframe_type = arrow_type_to_sframe(field.data_type())?;
        let col_array = struct_arr.column(idx);
        let col = arrow_array_to_column(col_array.as_ref(), field_sframe_type)?;
        field_cols.push((field.name().clone(), col));
    }

    let mut out = NullableVec::with_capacity(struct_arr.len());
    for i in 0..struct_arr.len() {
        if struct_arr.is_null(i) {
            out.push(None);
        } else {
            let entries: Vec<(FlexType, FlexType)> = field_cols
                .iter()
                .map(|(name, col)| {
                    let key = FlexType::String(FlexString::from(name.as_str()));
                    let val = col.get(i);
                    (key, val)
                })
                .collect();
            out.push(Some(FlexDict::from(entries)));
        }
    }
    Ok(ColumnData::Dict(out))
}

// ---------------------------------------------------------------------------
// Task 4: FlexTypeEnum -> Arrow DataType and ColumnData -> ArrayRef
// ---------------------------------------------------------------------------

/// Map an SFrame type enum to the corresponding Arrow data type.
pub fn sframe_type_to_arrow(ft: FlexTypeEnum) -> DataType {
    match ft {
        FlexTypeEnum::Integer => DataType::Int64,
        FlexTypeEnum::Float => DataType::Float64,
        FlexTypeEnum::String => DataType::Utf8,
        FlexTypeEnum::DateTime => DataType::Timestamp(TimeUnit::Microsecond, None),
        FlexTypeEnum::Vector => {
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true)))
        }
        // List, Dict, Undefined -> JSON-serialized strings
        FlexTypeEnum::List | FlexTypeEnum::Dict | FlexTypeEnum::Undefined => DataType::Utf8,
    }
}

/// Convert SFrame ColumnData to an Arrow ArrayRef.
pub fn column_to_arrow_array(col: &ColumnData) -> Result<ArrayRef> {
    match col {
        ColumnData::Integer(v) => {
            let arr = Int64Array::from(v.to_option_vec());
            Ok(Arc::new(arr) as ArrayRef)
        }
        ColumnData::Float(v) => {
            let arr = Float64Array::from(v.to_option_vec());
            Ok(Arc::new(arr) as ArrayRef)
        }
        ColumnData::String(v) => {
            let arr = StringArray::from(
                v.iter()
                    .map(|opt| opt.map(|s| s.as_ref()))
                    .collect::<Vec<Option<&str>>>(),
            );
            Ok(Arc::new(arr) as ArrayRef)
        }
        ColumnData::DateTime(v) => {
            let values: Vec<Option<i64>> = v
                .iter()
                .map(|opt| {
                    opt.map(|dt| {
                        dt.posix_timestamp * 1_000_000 + dt.microsecond as i64
                    })
                })
                .collect();
            let arr = TimestampMicrosecondArray::from(values);
            Ok(Arc::new(arr) as ArrayRef)
        }
        ColumnData::Vector(v) => {
            // Build a ListArray of Float64 values
            let mut offsets: Vec<i32> = Vec::with_capacity(v.len() + 1);
            let mut values: Vec<Option<f64>> = Vec::new();
            let mut nulls: Vec<bool> = Vec::with_capacity(v.len());

            offsets.push(0);
            for opt_vec in v.iter() {
                match opt_vec {
                    Some(vec) => {
                        for &val in vec.iter() {
                            values.push(Some(val));
                        }
                        offsets.push(values.len() as i32);
                        nulls.push(true);
                    }
                    None => {
                        offsets.push(*offsets.last().unwrap());
                        nulls.push(false);
                    }
                }
            }

            let value_array = Float64Array::from(values);
            let field = Arc::new(Field::new("item", DataType::Float64, true));
            let null_buffer =
                arrow::buffer::NullBuffer::from(arrow::buffer::BooleanBuffer::from(nulls));
            let offset_buffer = arrow::buffer::OffsetBuffer::new(
                arrow::buffer::ScalarBuffer::<i32>::from(offsets),
            );

            let list_array = ListArray::try_new(
                field,
                offset_buffer,
                Arc::new(value_array),
                Some(null_buffer),
            )
            .map_err(|e| SFrameError::Format(format!("Failed to create ListArray: {e}")))?;

            Ok(Arc::new(list_array) as ArrayRef)
        }
        ColumnData::List(v) => {
            // Serialize each list value to JSON string
            let strings: Vec<Option<String>> = v
                .iter()
                .map(|opt| {
                    opt.map(|items| {
                        let flex = FlexType::List(items.clone());
                        let json_val = flex_to_json(&flex);
                        serde_json::to_string(&json_val).unwrap_or_else(|_| "null".to_string())
                    })
                })
                .collect();
            let arr = StringArray::from(
                strings
                    .iter()
                    .map(|opt| opt.as_deref())
                    .collect::<Vec<Option<&str>>>(),
            );
            Ok(Arc::new(arr) as ArrayRef)
        }
        ColumnData::Dict(v) => {
            // Serialize each dict value to JSON string
            let strings: Vec<Option<String>> = v
                .iter()
                .map(|opt| {
                    opt.map(|entries| {
                        let flex = FlexType::Dict(entries.clone());
                        let json_val = flex_to_json(&flex);
                        serde_json::to_string(&json_val).unwrap_or_else(|_| "null".to_string())
                    })
                })
                .collect();
            let arr = StringArray::from(
                strings
                    .iter()
                    .map(|opt| opt.as_deref())
                    .collect::<Vec<Option<&str>>>(),
            );
            Ok(Arc::new(arr) as ArrayRef)
        }
        ColumnData::Flexible(v) => {
            // Serialize each value to JSON string; Undefined -> null
            let strings: Vec<String> = v
                .iter()
                .map(|val| {
                    let json_val = flex_to_json(val);
                    serde_json::to_string(&json_val).unwrap_or_else(|_| "null".to_string())
                })
                .collect();
            let arr = StringArray::from(
                strings
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<&str>>(),
            );
            Ok(Arc::new(arr) as ArrayRef)
        }
    }
}

// ---------------------------------------------------------------------------
// Task 5: RecordBatch <-> SFrameRows batch conversion
// ---------------------------------------------------------------------------

/// Convert an Arrow RecordBatch to SFrameRows.
///
/// Each column in the RecordBatch is mapped to an SFrame type and converted.
pub fn record_batch_to_sframe_rows(batch: &RecordBatch) -> Result<SFrameRows> {
    let schema = batch.schema();
    let mut columns = Vec::with_capacity(batch.num_columns());

    for (i, field) in schema.fields().iter().enumerate() {
        let sframe_type = arrow_type_to_sframe(field.data_type())?;
        let col = arrow_array_to_column(batch.column(i).as_ref(), sframe_type)?;
        columns.push(col);
    }

    SFrameRows::new(columns)
}

/// Convert SFrameRows to an Arrow RecordBatch.
///
/// Column names and types are used to build the Arrow schema.
pub fn sframe_rows_to_record_batch(
    rows: &SFrameRows,
    column_names: &[String],
    column_types: &[FlexTypeEnum],
) -> Result<RecordBatch> {
    if column_names.len() != rows.num_columns() {
        return Err(SFrameError::Format(format!(
            "column_names length {} != num_columns {}",
            column_names.len(),
            rows.num_columns()
        )));
    }
    if column_types.len() != rows.num_columns() {
        return Err(SFrameError::Format(format!(
            "column_types length {} != num_columns {}",
            column_types.len(),
            rows.num_columns()
        )));
    }

    let mut fields = Vec::with_capacity(rows.num_columns());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(rows.num_columns());

    for i in 0..rows.num_columns() {
        let arrow_dt = sframe_type_to_arrow(column_types[i]);
        fields.push(Field::new(&column_names[i], arrow_dt, true));
        let arr = column_to_arrow_array(rows.column(i))?;
        arrays.push(arr);
    }

    let schema = Arc::new(arrow::datatypes::Schema::new(fields));
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| SFrameError::Format(format!("Failed to create RecordBatch: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Fields, Schema};

    // ===== Task 2 tests: arrow_type_to_sframe =====

    #[test]
    fn test_arrow_int_types_to_sframe() {
        assert_eq!(
            arrow_type_to_sframe(&DataType::Int8).unwrap(),
            FlexTypeEnum::Integer
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Int16).unwrap(),
            FlexTypeEnum::Integer
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Int32).unwrap(),
            FlexTypeEnum::Integer
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Int64).unwrap(),
            FlexTypeEnum::Integer
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::UInt8).unwrap(),
            FlexTypeEnum::Integer
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::UInt16).unwrap(),
            FlexTypeEnum::Integer
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::UInt32).unwrap(),
            FlexTypeEnum::Integer
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::UInt64).unwrap(),
            FlexTypeEnum::Integer
        );
    }

    #[test]
    fn test_arrow_boolean_to_sframe() {
        assert_eq!(
            arrow_type_to_sframe(&DataType::Boolean).unwrap(),
            FlexTypeEnum::Integer
        );
    }

    #[test]
    fn test_arrow_float_types_to_sframe() {
        assert_eq!(
            arrow_type_to_sframe(&DataType::Float16).unwrap(),
            FlexTypeEnum::Float
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Float32).unwrap(),
            FlexTypeEnum::Float
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Float64).unwrap(),
            FlexTypeEnum::Float
        );
    }

    #[test]
    fn test_arrow_string_types_to_sframe() {
        assert_eq!(
            arrow_type_to_sframe(&DataType::Utf8).unwrap(),
            FlexTypeEnum::String
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::LargeUtf8).unwrap(),
            FlexTypeEnum::String
        );
    }

    #[test]
    fn test_arrow_timestamp_types_to_sframe() {
        assert_eq!(
            arrow_type_to_sframe(&DataType::Timestamp(TimeUnit::Second, None)).unwrap(),
            FlexTypeEnum::DateTime
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Timestamp(
                TimeUnit::Millisecond,
                Some("UTC".into())
            ))
            .unwrap(),
            FlexTypeEnum::DateTime
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Timestamp(TimeUnit::Microsecond, None)).unwrap(),
            FlexTypeEnum::DateTime
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Timestamp(TimeUnit::Nanosecond, None)).unwrap(),
            FlexTypeEnum::DateTime
        );
    }

    #[test]
    fn test_arrow_date_types_to_sframe() {
        assert_eq!(
            arrow_type_to_sframe(&DataType::Date32).unwrap(),
            FlexTypeEnum::DateTime
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Date64).unwrap(),
            FlexTypeEnum::DateTime
        );
    }

    #[test]
    fn test_arrow_list_float64_to_sframe_vector() {
        let dt = DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
        assert_eq!(
            arrow_type_to_sframe(&dt).unwrap(),
            FlexTypeEnum::Vector
        );
    }

    #[test]
    fn test_arrow_list_int32_to_sframe_list() {
        let dt = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        assert_eq!(
            arrow_type_to_sframe(&dt).unwrap(),
            FlexTypeEnum::List
        );
    }

    #[test]
    fn test_arrow_large_list_to_sframe_list() {
        let dt = DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true)));
        assert_eq!(
            arrow_type_to_sframe(&dt).unwrap(),
            FlexTypeEnum::List
        );
    }

    #[test]
    fn test_arrow_struct_to_sframe_dict() {
        let dt = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        assert_eq!(
            arrow_type_to_sframe(&dt).unwrap(),
            FlexTypeEnum::Dict
        );
    }

    #[test]
    fn test_arrow_unsupported_type_error() {
        assert!(arrow_type_to_sframe(&DataType::Binary).is_err());
        assert!(arrow_type_to_sframe(&DataType::Duration(TimeUnit::Second)).is_err());
    }

    // ===== Task 3 tests: arrow_array_to_column =====

    #[test]
    fn test_convert_int8_array() {
        let arr = Int8Array::from(vec![Some(1i8), None, Some(-3i8)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Integer).unwrap();
        match &col {
            ColumnData::Integer(v) => {
                assert_eq!(v, &vec![Some(1i64), None, Some(-3i64)]);
            }
            _ => panic!("Expected Integer column"),
        }
    }

    #[test]
    fn test_convert_int64_array() {
        let arr = Int64Array::from(vec![Some(100i64), None, Some(i64::MAX)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Integer).unwrap();
        match &col {
            ColumnData::Integer(v) => {
                assert_eq!(v, &vec![Some(100i64), None, Some(i64::MAX)]);
            }
            _ => panic!("Expected Integer column"),
        }
    }

    #[test]
    fn test_convert_uint64_overflow() {
        let arr = UInt64Array::from(vec![Some(u64::MAX)]);
        let result = arrow_array_to_column(&arr, FlexTypeEnum::Integer);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeds i64::MAX"), "Error was: {err}");
    }

    #[test]
    fn test_convert_uint64_valid() {
        let arr = UInt64Array::from(vec![Some(42u64), None, Some(0u64)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Integer).unwrap();
        match &col {
            ColumnData::Integer(v) => {
                assert_eq!(v, &vec![Some(42i64), None, Some(0i64)]);
            }
            _ => panic!("Expected Integer column"),
        }
    }

    #[test]
    fn test_convert_boolean_array() {
        let arr = BooleanArray::from(vec![Some(true), Some(false), None]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Integer).unwrap();
        match &col {
            ColumnData::Integer(v) => {
                assert_eq!(v, &vec![Some(1i64), Some(0i64), None]);
            }
            _ => panic!("Expected Integer column"),
        }
    }

    #[test]
    fn test_convert_float32_array() {
        let arr = Float32Array::from(vec![Some(1.5f32), None, Some(3.14f32)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Float).unwrap();
        match &col {
            ColumnData::Float(v) => {
                assert_eq!(v.len(), 3);
                assert!((*v.get(0).unwrap() - 1.5).abs() < 1e-6);
                assert!(v.get(1).is_none());
                assert!((*v.get(2).unwrap() - 3.14).abs() < 1e-4);
            }
            _ => panic!("Expected Float column"),
        }
    }

    #[test]
    fn test_convert_float64_array() {
        let arr = Float64Array::from(vec![Some(1.5), None, Some(f64::NAN)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Float).unwrap();
        match &col {
            ColumnData::Float(v) => {
                assert_eq!(v.get(0), Some(&1.5));
                assert!(v.get(1).is_none());
                assert!(v.get(2).unwrap().is_nan());
            }
            _ => panic!("Expected Float column"),
        }
    }

    #[test]
    fn test_convert_string_array() {
        let arr = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::String).unwrap();
        match &col {
            ColumnData::String(v) => {
                assert_eq!(v.get(0).map(|s| s.as_ref()), Some("hello"));
                assert!(v.get(1).is_none());
                assert_eq!(v.get(2).map(|s| s.as_ref()), Some("world"));
            }
            _ => panic!("Expected String column"),
        }
    }

    #[test]
    fn test_convert_large_string_array() {
        let arr = LargeStringArray::from(vec![Some("large"), None]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::String).unwrap();
        match &col {
            ColumnData::String(v) => {
                assert_eq!(v.get(0).map(|s| s.as_ref()), Some("large"));
                assert!(v.get(1).is_none());
            }
            _ => panic!("Expected String column"),
        }
    }

    #[test]
    fn test_convert_timestamp_microsecond() {
        // 1_500_000_123 microseconds = 1500 seconds + 123 microseconds
        let arr = TimestampMicrosecondArray::from(vec![Some(1_500_000_123i64), None]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::DateTime).unwrap();
        match &col {
            ColumnData::DateTime(v) => {
                let dt = v.get(0).unwrap();
                assert_eq!(dt.posix_timestamp, 1500);
                assert_eq!(dt.microsecond, 123);
                assert_eq!(dt.tz_offset_quarter_hours, 0);
                assert!(v.get(1).is_none());
            }
            _ => panic!("Expected DateTime column"),
        }
    }

    #[test]
    fn test_convert_timestamp_second() {
        let arr = TimestampSecondArray::from(vec![Some(1609459200i64)]); // 2021-01-01 00:00:00 UTC
        let col = arrow_array_to_column(&arr, FlexTypeEnum::DateTime).unwrap();
        match &col {
            ColumnData::DateTime(v) => {
                let dt = v.get(0).unwrap();
                assert_eq!(dt.posix_timestamp, 1609459200);
                assert_eq!(dt.microsecond, 0);
            }
            _ => panic!("Expected DateTime column"),
        }
    }

    #[test]
    fn test_convert_timestamp_millisecond() {
        // 1500123 ms = 1500 seconds + 123 ms = 1500 seconds + 123000 microseconds
        let arr = TimestampMillisecondArray::from(vec![Some(1_500_123i64)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::DateTime).unwrap();
        match &col {
            ColumnData::DateTime(v) => {
                let dt = v.get(0).unwrap();
                assert_eq!(dt.posix_timestamp, 1500);
                assert_eq!(dt.microsecond, 123_000);
            }
            _ => panic!("Expected DateTime column"),
        }
    }

    #[test]
    fn test_convert_timestamp_nanosecond() {
        // 1_500_000_123_456 ns -> 1_500_000_123 us -> 1500 seconds + 123 us
        // (456 ns truncated)
        let arr = TimestampNanosecondArray::from(vec![Some(1_500_000_123_456i64)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::DateTime).unwrap();
        match &col {
            ColumnData::DateTime(v) => {
                let dt = v.get(0).unwrap();
                assert_eq!(dt.posix_timestamp, 1500);
                assert_eq!(dt.microsecond, 123);
            }
            _ => panic!("Expected DateTime column"),
        }
    }

    #[test]
    fn test_convert_date32() {
        // Day 1 since epoch = 86400 seconds
        let arr = Date32Array::from(vec![Some(1i32), None, Some(0i32)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::DateTime).unwrap();
        match &col {
            ColumnData::DateTime(v) => {
                let dt = v.get(0).unwrap();
                assert_eq!(dt.posix_timestamp, 86400);
                assert_eq!(dt.microsecond, 0);
                assert!(v.get(1).is_none());
                let dt0 = v.get(2).unwrap();
                assert_eq!(dt0.posix_timestamp, 0);
            }
            _ => panic!("Expected DateTime column"),
        }
    }

    #[test]
    fn test_convert_date64() {
        // 86400123 ms = 86400 seconds + 123 ms = 86400 seconds + 123000 us
        let arr = Date64Array::from(vec![Some(86_400_123i64)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::DateTime).unwrap();
        match &col {
            ColumnData::DateTime(v) => {
                let dt = v.get(0).unwrap();
                assert_eq!(dt.posix_timestamp, 86400);
                assert_eq!(dt.microsecond, 123_000);
            }
            _ => panic!("Expected DateTime column"),
        }
    }

    #[test]
    fn test_convert_negative_timestamp() {
        // Negative timestamp: -1_500_123 microseconds
        // div_euclid(-1_500_123, 1_000_000) = -2
        // rem_euclid(-1_500_123, 1_000_000) = 499_877
        let arr = TimestampMicrosecondArray::from(vec![Some(-1_500_123i64)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::DateTime).unwrap();
        match &col {
            ColumnData::DateTime(v) => {
                let dt = v.get(0).unwrap();
                assert_eq!(dt.posix_timestamp, -2);
                assert_eq!(dt.microsecond, 499_877);
                // Verify roundtrip: -2 * 1_000_000 + 499_877 = -1_500_123
                assert_eq!(
                    dt.posix_timestamp * 1_000_000 + dt.microsecond as i64,
                    -1_500_123
                );
            }
            _ => panic!("Expected DateTime column"),
        }
    }

    #[test]
    fn test_convert_vector_list_array() {
        // Build a ListArray with Float64 values: [[1.0, 2.0], null, [3.0]]
        let values = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let offsets = OffsetBuffer::new(vec![0i32, 2, 2, 3].into());
        let field = Arc::new(Field::new("item", DataType::Float64, true));
        let null_buffer = arrow::buffer::NullBuffer::from(vec![true, false, true]);
        let list_arr =
            ListArray::try_new(field, offsets, Arc::new(values), Some(null_buffer)).unwrap();

        let col = arrow_array_to_column(&list_arr, FlexTypeEnum::Vector).unwrap();
        match &col {
            ColumnData::Vector(v) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v.get(0).unwrap().as_ref(), &[1.0, 2.0]);
                assert!(v.get(1).is_none());
                assert_eq!(v.get(2).unwrap().as_ref(), &[3.0]);
            }
            _ => panic!("Expected Vector column"),
        }
    }

    #[test]
    fn test_convert_list_of_int() {
        // Build a ListArray with Int32 values: [[10, 20], null, [30]]
        let values = Int32Array::from(vec![10, 20, 30]);
        let offsets = OffsetBuffer::new(vec![0i32, 2, 2, 3].into());
        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let null_buffer = arrow::buffer::NullBuffer::from(vec![true, false, true]);
        let list_arr =
            ListArray::try_new(field, offsets, Arc::new(values), Some(null_buffer)).unwrap();

        let col = arrow_array_to_column(&list_arr, FlexTypeEnum::List).unwrap();
        match &col {
            ColumnData::List(v) => {
                assert_eq!(v.len(), 3);
                let first = v.get(0).unwrap();
                assert_eq!(first.len(), 2);
                assert_eq!(first[0], FlexType::Integer(10));
                assert_eq!(first[1], FlexType::Integer(20));
                assert!(v.get(1).is_none());
                let third = v.get(2).unwrap();
                assert_eq!(third.len(), 1);
                assert_eq!(third[0], FlexType::Integer(30));
            }
            _ => panic!("Expected List column"),
        }
    }

    #[test]
    fn test_convert_struct_to_dict() {
        let int_arr = Int64Array::from(vec![Some(1i64), Some(2i64)]);
        let str_arr = StringArray::from(vec![Some("a"), Some("b")]);
        let fields = vec![
            Arc::new(Field::new("x", DataType::Int64, true)),
            Arc::new(Field::new("y", DataType::Utf8, true)),
        ];
        let struct_arr =
            StructArray::try_new(fields.into(), vec![Arc::new(int_arr), Arc::new(str_arr)], None)
                .unwrap();

        let col = arrow_array_to_column(&struct_arr, FlexTypeEnum::Dict).unwrap();
        match &col {
            ColumnData::Dict(v) => {
                assert_eq!(v.len(), 2);
                let first = v.get(0).unwrap();
                assert_eq!(first.len(), 2);
                assert_eq!(first[0].0, FlexType::String(FlexString::from("x")));
                assert_eq!(first[0].1, FlexType::Integer(1));
                assert_eq!(first[1].0, FlexType::String(FlexString::from("y")));
                assert_eq!(first[1].1, FlexType::String(FlexString::from("a")));
            }
            _ => panic!("Expected Dict column"),
        }
    }

    // ===== Task 4 tests: sframe_type_to_arrow and column_to_arrow_array =====

    #[test]
    fn test_sframe_type_to_arrow_mapping() {
        assert_eq!(sframe_type_to_arrow(FlexTypeEnum::Integer), DataType::Int64);
        assert_eq!(
            sframe_type_to_arrow(FlexTypeEnum::Float),
            DataType::Float64
        );
        assert_eq!(sframe_type_to_arrow(FlexTypeEnum::String), DataType::Utf8);
        assert_eq!(
            sframe_type_to_arrow(FlexTypeEnum::DateTime),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            sframe_type_to_arrow(FlexTypeEnum::Vector),
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true)))
        );
        // List, Dict, Undefined all map to Utf8 (JSON)
        assert_eq!(sframe_type_to_arrow(FlexTypeEnum::List), DataType::Utf8);
        assert_eq!(sframe_type_to_arrow(FlexTypeEnum::Dict), DataType::Utf8);
        assert_eq!(
            sframe_type_to_arrow(FlexTypeEnum::Undefined),
            DataType::Utf8
        );
    }

    #[test]
    fn test_column_to_arrow_integer() {
        let col = ColumnData::Integer(vec![Some(1), None, Some(42)].into());
        let arr = column_to_arrow_array(&col).unwrap();
        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.len(), 3);
        assert_eq!(int_arr.value(0), 1);
        assert!(int_arr.is_null(1));
        assert_eq!(int_arr.value(2), 42);
    }

    #[test]
    fn test_column_to_arrow_float() {
        let col = ColumnData::Float(vec![Some(1.5), None, Some(3.14)].into());
        let arr = column_to_arrow_array(&col).unwrap();
        let f_arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(f_arr.len(), 3);
        assert!((f_arr.value(0) - 1.5).abs() < 1e-10);
        assert!(f_arr.is_null(1));
        assert!((f_arr.value(2) - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_column_to_arrow_string() {
        let col = ColumnData::String(vec![Some(FlexString::from("hello")), None, Some(FlexString::from(""))].into());
        let arr = column_to_arrow_array(&col).unwrap();
        let s_arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(s_arr.value(0), "hello");
        assert!(s_arr.is_null(1));
        assert_eq!(s_arr.value(2), "");
    }

    #[test]
    fn test_column_to_arrow_datetime() {
        let col = ColumnData::DateTime(vec![
            Some(FlexDateTime {
                posix_timestamp: 1500,
                tz_offset_quarter_hours: 0,
                microsecond: 123,
            }),
            None,
        ].into());
        let arr = column_to_arrow_array(&col).unwrap();
        let ts_arr = arr
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(ts_arr.value(0), 1_500_000_123);
        assert!(ts_arr.is_null(1));
    }

    #[test]
    fn test_column_to_arrow_vector() {
        let col = ColumnData::Vector(vec![
            Some(FlexVec::from(vec![1.0, 2.0])),
            None,
            Some(FlexVec::from(vec![3.0])),
        ].into());
        let arr = column_to_arrow_array(&col).unwrap();
        let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_arr.len(), 3);

        // First element: [1.0, 2.0]
        assert!(!list_arr.is_null(0));
        let first = list_arr.value(0);
        let f64_arr = first.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(f64_arr.len(), 2);
        assert_eq!(f64_arr.value(0), 1.0);
        assert_eq!(f64_arr.value(1), 2.0);

        // Second element: null
        assert!(list_arr.is_null(1));

        // Third element: [3.0]
        assert!(!list_arr.is_null(2));
        let third = list_arr.value(2);
        let f64_arr = third.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(f64_arr.len(), 1);
        assert_eq!(f64_arr.value(0), 3.0);
    }

    #[test]
    fn test_column_to_arrow_list_json() {
        let col = ColumnData::List(vec![
            Some(FlexList::from(
                vec![FlexType::Integer(1), FlexType::Integer(2)],
            )),
            None,
        ].into());
        let arr = column_to_arrow_array(&col).unwrap();
        let s_arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(s_arr.value(0), "[1,2]");
        assert!(s_arr.is_null(1));
    }

    #[test]
    fn test_column_to_arrow_dict_json() {
        let col = ColumnData::Dict(vec![Some(FlexDict::from(
            vec![(
                FlexType::String(FlexString::from("key")),
                FlexType::Integer(42),
            )],
        ))].into());
        let arr = column_to_arrow_array(&col).unwrap();
        let s_arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(s_arr.value(0), "{\"key\":42}");
    }

    #[test]
    fn test_column_to_arrow_flexible() {
        let col = ColumnData::Flexible(vec![
            FlexType::Integer(42),
            FlexType::Undefined,
            FlexType::String(FlexString::from("hello")),
        ]);
        let arr = column_to_arrow_array(&col).unwrap();
        let s_arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(s_arr.value(0), "42");
        assert_eq!(s_arr.value(1), "null");
        assert_eq!(s_arr.value(2), "\"hello\"");
    }

    // ===== Task 5 tests: RecordBatch <-> SFrameRows =====

    #[test]
    fn test_record_batch_to_sframe_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("val", DataType::Float64, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2)])),
                Arc::new(Float64Array::from(vec![Some(1.5), None])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .unwrap();

        let rows = record_batch_to_sframe_rows(&batch).unwrap();
        assert_eq!(rows.num_rows(), 2);
        assert_eq!(rows.num_columns(), 3);
        assert_eq!(
            rows.dtypes(),
            vec![FlexTypeEnum::Integer, FlexTypeEnum::Float, FlexTypeEnum::String]
        );
        assert_eq!(rows.column(0).get(0), FlexType::Integer(1));
        assert_eq!(rows.column(1).get(0), FlexType::Float(1.5));
        assert_eq!(rows.column(1).get(1), FlexType::Undefined);
        assert_eq!(rows.column(2).get(1), FlexType::String(FlexString::from("b")));
    }

    #[test]
    fn test_sframe_rows_to_record_batch() {
        let rows = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(10), Some(20)].into()),
            ColumnData::String(vec![Some(FlexString::from("x")), None].into()),
        ])
        .unwrap();
        let names = vec!["id".to_string(), "label".to_string()];
        let types = vec![FlexTypeEnum::Integer, FlexTypeEnum::String];

        let batch = sframe_rows_to_record_batch(&rows, &names, &types).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.schema().field(0).name(), "id");
        assert_eq!(*batch.schema().field(0).data_type(), DataType::Int64);
        assert_eq!(batch.schema().field(1).name(), "label");
        assert_eq!(*batch.schema().field(1).data_type(), DataType::Utf8);

        let int_arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(int_arr.value(0), 10);
        assert_eq!(int_arr.value(1), 20);

        let str_arr = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(str_arr.value(0), "x");
        assert!(str_arr.is_null(1));
    }

    #[test]
    fn test_roundtrip_record_batch_sframe_rows() {
        // Build Arrow RecordBatch -> SFrameRows -> RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Float64, true),
        ]));
        let orig_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])),
                Arc::new(Float64Array::from(vec![Some(1.5), Some(2.5), None])),
            ],
        )
        .unwrap();

        let sframe_rows = record_batch_to_sframe_rows(&orig_batch).unwrap();
        let names = vec!["a".to_string(), "b".to_string()];
        let types = vec![FlexTypeEnum::Integer, FlexTypeEnum::Float];
        let roundtrip_batch =
            sframe_rows_to_record_batch(&sframe_rows, &names, &types).unwrap();

        assert_eq!(roundtrip_batch.num_rows(), 3);

        let int_arr = roundtrip_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(int_arr.value(0), 1);
        assert!(int_arr.is_null(1));
        assert_eq!(int_arr.value(2), 3);

        let f_arr = roundtrip_batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(f_arr.value(0), 1.5);
        assert_eq!(f_arr.value(1), 2.5);
        assert!(f_arr.is_null(2));
    }

    #[test]
    fn test_sframe_rows_to_record_batch_mismatched_names_length() {
        let rows = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1)].into()),
        ])
        .unwrap();
        let result = sframe_rows_to_record_batch(
            &rows,
            &["a".to_string(), "b".to_string()],
            &[FlexTypeEnum::Integer],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_sframe_rows_to_record_batch_mismatched_types_length() {
        let rows = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1)].into()),
        ])
        .unwrap();
        let result = sframe_rows_to_record_batch(
            &rows,
            &["a".to_string()],
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_record_batch_roundtrip() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(Vec::<Option<i64>>::new()))],
        )
        .unwrap();

        let sframe_rows = record_batch_to_sframe_rows(&batch).unwrap();
        assert_eq!(sframe_rows.num_rows(), 0);
        assert_eq!(sframe_rows.num_columns(), 1);

        let roundtrip = sframe_rows_to_record_batch(
            &sframe_rows,
            &["x".to_string()],
            &[FlexTypeEnum::Integer],
        )
        .unwrap();
        assert_eq!(roundtrip.num_rows(), 0);
    }

    #[test]
    fn test_datetime_roundtrip() {
        // Create SFrame DateTime -> Arrow Timestamp -> SFrame DateTime
        let orig = ColumnData::DateTime(vec![
            Some(FlexDateTime {
                posix_timestamp: 1609459200,
                tz_offset_quarter_hours: 0,
                microsecond: 500_000,
            }),
            None,
            Some(FlexDateTime {
                posix_timestamp: 0,
                tz_offset_quarter_hours: 0,
                microsecond: 0,
            }),
        ].into());

        let arrow_arr = column_to_arrow_array(&orig).unwrap();
        let back = arrow_array_to_column(arrow_arr.as_ref(), FlexTypeEnum::DateTime).unwrap();

        match (&orig, &back) {
            (ColumnData::DateTime(a), ColumnData::DateTime(b)) => {
                assert_eq!(a.len(), b.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    match (x, y) {
                        (Some(ax), Some(bx)) => {
                            assert_eq!(ax.posix_timestamp, bx.posix_timestamp);
                            assert_eq!(ax.microsecond, bx.microsecond);
                        }
                        (None, None) => {}
                        _ => panic!("Null mismatch"),
                    }
                }
            }
            _ => panic!("Expected DateTime columns"),
        }
    }

    #[test]
    fn test_vector_roundtrip() {
        let orig = ColumnData::Vector(vec![
            Some(FlexVec::from(vec![1.0, 2.0, 3.0])),
            None,
            Some(FlexVec::from(Vec::<f64>::new())),
        ].into());

        let arrow_arr = column_to_arrow_array(&orig).unwrap();
        let back = arrow_array_to_column(arrow_arr.as_ref(), FlexTypeEnum::Vector).unwrap();

        match (&orig, &back) {
            (ColumnData::Vector(a), ColumnData::Vector(b)) => {
                assert_eq!(a.len(), b.len());
                assert_eq!(a.get(0).unwrap().as_ref(), b.get(0).unwrap().as_ref());
                assert!(a.get(1).is_none() && b.get(1).is_none());
                assert_eq!(a.get(2).unwrap().as_ref(), b.get(2).unwrap().as_ref());
            }
            _ => panic!("Expected Vector columns"),
        }
    }

    #[test]
    fn test_all_int_types_via_record_batch() {
        // Test that we can handle a RecordBatch with various int types
        let schema = Arc::new(Schema::new(vec![
            Field::new("i8", DataType::Int8, true),
            Field::new("i16", DataType::Int16, true),
            Field::new("i32", DataType::Int32, true),
            Field::new("u8", DataType::UInt8, true),
            Field::new("u16", DataType::UInt16, true),
            Field::new("u32", DataType::UInt32, true),
            Field::new("bool", DataType::Boolean, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int8Array::from(vec![Some(1i8)])),
                Arc::new(Int16Array::from(vec![Some(2i16)])),
                Arc::new(Int32Array::from(vec![Some(3i32)])),
                Arc::new(UInt8Array::from(vec![Some(4u8)])),
                Arc::new(UInt16Array::from(vec![Some(5u16)])),
                Arc::new(UInt32Array::from(vec![Some(6u32)])),
                Arc::new(BooleanArray::from(vec![Some(true)])),
            ],
        )
        .unwrap();

        let rows = record_batch_to_sframe_rows(&batch).unwrap();
        assert_eq!(rows.num_rows(), 1);
        assert_eq!(rows.column(0).get(0), FlexType::Integer(1));
        assert_eq!(rows.column(1).get(0), FlexType::Integer(2));
        assert_eq!(rows.column(2).get(0), FlexType::Integer(3));
        assert_eq!(rows.column(3).get(0), FlexType::Integer(4));
        assert_eq!(rows.column(4).get(0), FlexType::Integer(5));
        assert_eq!(rows.column(5).get(0), FlexType::Integer(6));
        assert_eq!(rows.column(6).get(0), FlexType::Integer(1));
    }

    #[test]
    fn test_map_type_to_sframe() {
        let dt = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Int64, true),
                ])),
                false,
            )),
            false,
        );
        assert_eq!(arrow_type_to_sframe(&dt).unwrap(), FlexTypeEnum::Dict);
    }
}
