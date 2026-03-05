//! Batch representation for the query engine.
//!
//! `SFrameRows` is a columnar batch of rows using typed column vectors.
//! This avoids the per-value tag overhead of `Vec<Vec<FlexType>>`.

use std::sync::Arc;

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexDateTime, FlexType, FlexTypeEnum};

/// A batch of rows stored in columnar format with typed vectors.
#[derive(Debug, Clone)]
pub struct SFrameRows {
    columns: Vec<ColumnData>,
    num_rows: usize,
}

/// Typed column vector. `None` represents UNDEFINED/NULL.
#[derive(Debug, Clone)]
pub enum ColumnData {
    Integer(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
    String(Vec<Option<Arc<str>>>),
    Vector(Vec<Option<Arc<[f64]>>>),
    List(Vec<Option<Arc<[FlexType]>>>),
    Dict(Vec<Option<Arc<[(FlexType, FlexType)]>>>),
    DateTime(Vec<Option<FlexDateTime>>),
    /// Mixed-type column for UNDEFINED/per-value-parsed data.
    /// Uses `FlexType::Undefined` for null instead of `Option` wrapping.
    Flexible(Vec<FlexType>),
}

/// Dispatch on ColumnData variant, binding the inner Vec to `$vec`.
/// Use for operations where the logic is the same across all variants.
macro_rules! with_column_data {
    ($col:expr, $vec:ident => $body:expr) => {
        match $col {
            ColumnData::Integer($vec) => $body,
            ColumnData::Float($vec) => $body,
            ColumnData::String($vec) => $body,
            ColumnData::Vector($vec) => $body,
            ColumnData::List($vec) => $body,
            ColumnData::Dict($vec) => $body,
            ColumnData::DateTime($vec) => $body,
            ColumnData::Flexible($vec) => $body,
        }
    };
}

/// Dispatch on two ColumnData values of the same variant.
/// Returns Err on type mismatch.
macro_rules! with_column_data_pair {
    ($a:expr, $b:expr, $va:ident, $vb:ident => $body:expr) => {
        match ($a, $b) {
            (ColumnData::Integer($va), ColumnData::Integer($vb)) => $body,
            (ColumnData::Float($va), ColumnData::Float($vb)) => $body,
            (ColumnData::String($va), ColumnData::String($vb)) => $body,
            (ColumnData::Vector($va), ColumnData::Vector($vb)) => $body,
            (ColumnData::List($va), ColumnData::List($vb)) => $body,
            (ColumnData::Dict($va), ColumnData::Dict($vb)) => $body,
            (ColumnData::DateTime($va), ColumnData::DateTime($vb)) => $body,
            (ColumnData::Flexible($va), ColumnData::Flexible($vb)) => $body,
            (a, b) => {
                return Err(SFrameError::Type(format!(
                    "Column type mismatch: {:?} vs {:?}",
                    a.dtype(),
                    b.dtype()
                )));
            }
        }
    };
}

impl ColumnData {
    /// Create an empty column of the given type.
    pub fn empty(dtype: FlexTypeEnum) -> Self {
        match dtype {
            FlexTypeEnum::Integer => ColumnData::Integer(Vec::new()),
            FlexTypeEnum::Float => ColumnData::Float(Vec::new()),
            FlexTypeEnum::String => ColumnData::String(Vec::new()),
            FlexTypeEnum::Vector => ColumnData::Vector(Vec::new()),
            FlexTypeEnum::List => ColumnData::List(Vec::new()),
            FlexTypeEnum::Dict => ColumnData::Dict(Vec::new()),
            FlexTypeEnum::DateTime => ColumnData::DateTime(Vec::new()),
            FlexTypeEnum::Undefined => ColumnData::Flexible(Vec::new()),
        }
    }

    /// Number of elements in this column.
    pub fn len(&self) -> usize {
        with_column_data!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The type of this column.
    pub fn dtype(&self) -> FlexTypeEnum {
        match self {
            ColumnData::Integer(_) => FlexTypeEnum::Integer,
            ColumnData::Float(_) => FlexTypeEnum::Float,
            ColumnData::String(_) => FlexTypeEnum::String,
            ColumnData::Vector(_) => FlexTypeEnum::Vector,
            ColumnData::List(_) => FlexTypeEnum::List,
            ColumnData::Dict(_) => FlexTypeEnum::Dict,
            ColumnData::DateTime(_) => FlexTypeEnum::DateTime,
            ColumnData::Flexible(_) => FlexTypeEnum::Undefined,
        }
    }

    /// Push a FlexType value into this column.
    pub fn push(&mut self, value: &FlexType) -> Result<()> {
        match (self, value) {
            (ColumnData::Integer(v), FlexType::Integer(i)) => v.push(Some(*i)),
            (ColumnData::Integer(v), FlexType::Undefined) => v.push(None),
            (ColumnData::Float(v), FlexType::Float(f)) => v.push(Some(*f)),
            (ColumnData::Float(v), FlexType::Undefined) => v.push(None),
            (ColumnData::String(v), FlexType::String(s)) => v.push(Some(s.clone())),
            (ColumnData::String(v), FlexType::Undefined) => v.push(None),
            (ColumnData::Vector(v), FlexType::Vector(vec)) => v.push(Some(vec.clone())),
            (ColumnData::Vector(v), FlexType::Undefined) => v.push(None),
            (ColumnData::List(v), FlexType::List(l)) => v.push(Some(l.clone())),
            (ColumnData::List(v), FlexType::Undefined) => v.push(None),
            (ColumnData::Dict(v), FlexType::Dict(d)) => v.push(Some(d.clone())),
            (ColumnData::Dict(v), FlexType::Undefined) => v.push(None),
            (ColumnData::DateTime(v), FlexType::DateTime(dt)) => v.push(Some(dt.clone())),
            (ColumnData::DateTime(v), FlexType::Undefined) => v.push(None),
            (ColumnData::Flexible(v), val) => v.push(val.clone()),
            (col, val) => {
                return Err(SFrameError::Type(format!(
                    "Cannot push {:?} into {:?} column",
                    val.type_enum(),
                    col.dtype()
                )));
            }
        }
        Ok(())
    }

    /// Get a value at the given index as a FlexType.
    pub fn get(&self, index: usize) -> FlexType {
        match self {
            ColumnData::Integer(v) => match &v[index] {
                Some(i) => FlexType::Integer(*i),
                None => FlexType::Undefined,
            },
            ColumnData::Float(v) => match &v[index] {
                Some(f) => FlexType::Float(*f),
                None => FlexType::Undefined,
            },
            ColumnData::String(v) => match &v[index] {
                Some(s) => FlexType::String(s.clone()),
                None => FlexType::Undefined,
            },
            ColumnData::Vector(v) => match &v[index] {
                Some(vec) => FlexType::Vector(vec.clone()),
                None => FlexType::Undefined,
            },
            ColumnData::List(v) => match &v[index] {
                Some(l) => FlexType::List(l.clone()),
                None => FlexType::Undefined,
            },
            ColumnData::Dict(v) => match &v[index] {
                Some(d) => FlexType::Dict(d.clone()),
                None => FlexType::Undefined,
            },
            ColumnData::DateTime(v) => match &v[index] {
                Some(dt) => FlexType::DateTime(dt.clone()),
                None => FlexType::Undefined,
            },
            ColumnData::Flexible(v) => v[index].clone(),
        }
    }

    /// Extend this column with values from another column of the same type.
    pub fn extend(&mut self, other: &ColumnData) -> Result<()> {
        with_column_data_pair!(self, other, a, b => a.extend_from_slice(b));
        Ok(())
    }

    /// Bulk-convert a borrowed slice of FlexType values to typed ColumnData.
    ///
    /// Single variant dispatch at the column level instead of per-element.
    /// Values that don't match the expected type are treated as None/Undefined.
    pub fn from_flex_slice(data: &[FlexType], dtype: FlexTypeEnum) -> Self {
        macro_rules! convert_opt {
            ($variant:ident, $inner:ident) => {
                ColumnData::$variant(data.iter().map(|v| match v {
                    FlexType::$inner(x) => Some(x.clone()),
                    _ => None,
                }).collect())
            };
        }
        match dtype {
            FlexTypeEnum::Integer => convert_opt!(Integer, Integer),
            FlexTypeEnum::Float => convert_opt!(Float, Float),
            FlexTypeEnum::String => convert_opt!(String, String),
            FlexTypeEnum::Vector => convert_opt!(Vector, Vector),
            FlexTypeEnum::List => convert_opt!(List, List),
            FlexTypeEnum::Dict => convert_opt!(Dict, Dict),
            FlexTypeEnum::DateTime => convert_opt!(DateTime, DateTime),
            FlexTypeEnum::Undefined => ColumnData::Flexible(data.to_vec()),
        }
    }

    /// Bulk-convert by consuming a Vec<FlexType> (avoids Arc clones for refcounted types).
    pub fn from_flex_vec(data: Vec<FlexType>, dtype: FlexTypeEnum) -> Self {
        macro_rules! convert_opt {
            ($variant:ident, $inner:ident) => {
                ColumnData::$variant(data.into_iter().map(|v| match v {
                    FlexType::$inner(x) => Some(x),
                    _ => None,
                }).collect())
            };
        }
        match dtype {
            FlexTypeEnum::Integer => convert_opt!(Integer, Integer),
            FlexTypeEnum::Float => convert_opt!(Float, Float),
            FlexTypeEnum::String => convert_opt!(String, String),
            FlexTypeEnum::Vector => convert_opt!(Vector, Vector),
            FlexTypeEnum::List => convert_opt!(List, List),
            FlexTypeEnum::Dict => convert_opt!(Dict, Dict),
            FlexTypeEnum::DateTime => convert_opt!(DateTime, DateTime),
            FlexTypeEnum::Undefined => ColumnData::Flexible(data),
        }
    }

    /// Select elements by indices, staying in typed storage.
    ///
    /// Avoids the FlexType round-trip that `get()` + `push()` incurs.
    pub fn gather(&self, indices: &[usize]) -> Self {
        macro_rules! gather_typed {
            ($variant:ident, $v:expr) => {
                ColumnData::$variant(indices.iter().map(|&i| $v[i].clone()).collect())
            };
        }
        match self {
            ColumnData::Integer(v) => gather_typed!(Integer, v),
            ColumnData::Float(v) => gather_typed!(Float, v),
            ColumnData::String(v) => gather_typed!(String, v),
            ColumnData::Vector(v) => gather_typed!(Vector, v),
            ColumnData::List(v) => gather_typed!(List, v),
            ColumnData::Dict(v) => gather_typed!(Dict, v),
            ColumnData::DateTime(v) => gather_typed!(DateTime, v),
            ColumnData::Flexible(v) => gather_typed!(Flexible, v),
        }
    }

    /// Apply a function to each element, collecting results as FlexType values.
    ///
    /// Single variant dispatch, then tight loop — avoids per-element `get()` dispatch.
    pub fn map(&self, func: &dyn Fn(&FlexType) -> FlexType) -> Vec<FlexType> {
        macro_rules! map_opt {
            ($v:expr, $variant:ident) => {
                $v.iter().map(|val| {
                    let ft = match val {
                        Some(x) => FlexType::$variant(x.clone()),
                        None => FlexType::Undefined,
                    };
                    func(&ft)
                }).collect()
            };
        }
        match self {
            ColumnData::Integer(v) => map_opt!(v, Integer),
            ColumnData::Float(v) => map_opt!(v, Float),
            ColumnData::String(v) => map_opt!(v, String),
            ColumnData::Vector(v) => map_opt!(v, Vector),
            ColumnData::List(v) => map_opt!(v, List),
            ColumnData::Dict(v) => map_opt!(v, Dict),
            ColumnData::DateTime(v) => map_opt!(v, DateTime),
            ColumnData::Flexible(v) => v.iter().map(|val| func(val)).collect(),
        }
    }

    /// Evaluate a predicate on each element, returning indices of matches.
    ///
    /// Dispatches on the column variant once, then runs a tight loop over
    /// the typed vec — avoids per-element variant dispatch that `get()` incurs.
    pub fn filter_indices(&self, pred: &dyn Fn(&FlexType) -> bool) -> Vec<usize> {
        macro_rules! filter_opt {
            ($v:expr, $variant:ident) => {
                $v.iter().enumerate().filter_map(|(i, val)| {
                    let ft = match val {
                        Some(x) => FlexType::$variant(x.clone()),
                        None => FlexType::Undefined,
                    };
                    if pred(&ft) { Some(i) } else { None }
                }).collect()
            };
        }
        match self {
            ColumnData::Integer(v) => filter_opt!(v, Integer),
            ColumnData::Float(v) => filter_opt!(v, Float),
            ColumnData::String(v) => filter_opt!(v, String),
            ColumnData::Vector(v) => filter_opt!(v, Vector),
            ColumnData::List(v) => filter_opt!(v, List),
            ColumnData::Dict(v) => filter_opt!(v, Dict),
            ColumnData::DateTime(v) => filter_opt!(v, DateTime),
            ColumnData::Flexible(v) => {
                v.iter().enumerate().filter_map(|(i, val)| {
                    if pred(val) { Some(i) } else { None }
                }).collect()
            }
        }
    }

    /// Convert this column to a `Vec<FlexType>`.
    ///
    /// Used when feeding column data to `SegmentWriter::write_column_block()`.
    pub fn to_flex_vec(&self) -> Vec<FlexType> {
        macro_rules! convert_opt {
            ($v:expr, $variant:ident) => {
                $v.iter().map(|val| match val {
                    Some(x) => FlexType::$variant(x.clone()),
                    None => FlexType::Undefined,
                }).collect()
            };
        }
        match self {
            ColumnData::Integer(v) => convert_opt!(v, Integer),
            ColumnData::Float(v) => convert_opt!(v, Float),
            ColumnData::String(v) => convert_opt!(v, String),
            ColumnData::Vector(v) => convert_opt!(v, Vector),
            ColumnData::List(v) => convert_opt!(v, List),
            ColumnData::Dict(v) => convert_opt!(v, Dict),
            ColumnData::DateTime(v) => convert_opt!(v, DateTime),
            ColumnData::Flexible(v) => v.clone(),
        }
    }

    /// Return indices of "truthy" elements (non-zero, non-null).
    ///
    /// Operates directly on typed storage without constructing FlexType values.
    pub fn truthy_indices(&self) -> Vec<usize> {
        match self {
            ColumnData::Integer(v) => v.iter().enumerate()
                .filter_map(|(i, val)| match val {
                    Some(x) if *x != 0 => Some(i),
                    _ => None,
                }).collect(),
            ColumnData::Float(v) => v.iter().enumerate()
                .filter_map(|(i, val)| match val {
                    Some(x) if *x != 0.0 => Some(i),
                    _ => None,
                }).collect(),
            ColumnData::String(v) => v.iter().enumerate()
                .filter_map(|(i, val)| if val.is_some() { Some(i) } else { None }).collect(),
            ColumnData::Vector(v) => v.iter().enumerate()
                .filter_map(|(i, val)| if val.is_some() { Some(i) } else { None }).collect(),
            ColumnData::List(v) => v.iter().enumerate()
                .filter_map(|(i, val)| if val.is_some() { Some(i) } else { None }).collect(),
            ColumnData::Dict(v) => v.iter().enumerate()
                .filter_map(|(i, val)| if val.is_some() { Some(i) } else { None }).collect(),
            ColumnData::DateTime(v) => v.iter().enumerate()
                .filter_map(|(i, val)| if val.is_some() { Some(i) } else { None }).collect(),
            ColumnData::Flexible(v) => v.iter().enumerate()
                .filter_map(|(i, val)| match val {
                    FlexType::Integer(x) if *x != 0 => Some(i),
                    FlexType::Float(x) if *x != 0.0 => Some(i),
                    FlexType::Undefined | FlexType::Integer(_) | FlexType::Float(_) => None,
                    _ => Some(i),
                }).collect(),
        }
    }
}

impl SFrameRows {
    /// Create a new batch from typed columns. All columns must have the same length.
    pub fn new(columns: Vec<ColumnData>) -> Result<Self> {
        if columns.is_empty() {
            return Ok(SFrameRows {
                columns,
                num_rows: 0,
            });
        }
        let num_rows = columns[0].len();
        for (i, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(SFrameError::Format(format!(
                    "Column {} has {} rows, expected {}",
                    i,
                    col.len(),
                    num_rows
                )));
            }
        }
        Ok(SFrameRows { columns, num_rows })
    }

    /// Create an empty batch with the given schema.
    pub fn empty(dtypes: &[FlexTypeEnum]) -> Self {
        let columns = dtypes.iter().map(|&dt| ColumnData::empty(dt)).collect();
        SFrameRows {
            columns,
            num_rows: 0,
        }
    }

    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Column types.
    pub fn dtypes(&self) -> Vec<FlexTypeEnum> {
        self.columns.iter().map(|c| c.dtype()).collect()
    }

    /// Access a column by index.
    pub fn column(&self, index: usize) -> &ColumnData {
        &self.columns[index]
    }

    /// Access all columns.
    pub fn columns(&self) -> &[ColumnData] {
        &self.columns
    }

    /// Get a single row as a Vec<FlexType>.
    pub fn row(&self, index: usize) -> Vec<FlexType> {
        self.columns.iter().map(|col| col.get(index)).collect()
    }

    /// Convert from a Vec of FlexType rows (row-major) to columnar format.
    pub fn from_rows(rows: &[Vec<FlexType>], dtypes: &[FlexTypeEnum]) -> Result<Self> {
        let num_columns = dtypes.len();
        let mut columns: Vec<ColumnData> = dtypes.iter().map(|&dt| ColumnData::empty(dt)).collect();

        for (row_idx, row) in rows.iter().enumerate() {
            if row.len() != num_columns {
                return Err(SFrameError::Format(format!(
                    "Row {} has {} values, expected {}",
                    row_idx,
                    row.len(),
                    num_columns
                )));
            }
            for (col_idx, val) in row.iter().enumerate() {
                columns[col_idx].push(val)?;
            }
        }

        let num_rows = rows.len();
        Ok(SFrameRows { columns, num_rows })
    }

    /// Convert from column-major Vec<FlexType> arrays (as read from SFrame storage).
    pub fn from_column_vecs(
        column_data: Vec<Vec<FlexType>>,
        dtypes: &[FlexTypeEnum],
    ) -> Result<Self> {
        if column_data.len() != dtypes.len() {
            return Err(SFrameError::Format(format!(
                "Got {} columns, expected {}",
                column_data.len(),
                dtypes.len()
            )));
        }

        let num_rows = if column_data.is_empty() {
            0
        } else {
            column_data[0].len()
        };

        for (col_idx, data) in column_data.iter().enumerate() {
            if data.len() != num_rows {
                return Err(SFrameError::Format(format!(
                    "Column {} has {} rows, expected {}",
                    col_idx,
                    data.len(),
                    num_rows
                )));
            }
        }

        let columns: Vec<ColumnData> = column_data
            .into_iter()
            .zip(dtypes.iter())
            .map(|(data, &dtype)| ColumnData::from_flex_vec(data, dtype))
            .collect();

        Ok(SFrameRows { columns, num_rows })
    }

    /// Convert to row-major Vec<FlexType> format.
    pub fn to_rows(&self) -> Vec<Vec<FlexType>> {
        (0..self.num_rows).map(|i| self.row(i)).collect()
    }

    /// Convert to column-major Vec<FlexType> format.
    pub fn to_column_vecs(&self) -> Vec<Vec<FlexType>> {
        self.columns
            .iter()
            .map(|col| (0..self.num_rows).map(|i| col.get(i)).collect())
            .collect()
    }

    /// Select specific columns by index.
    pub fn select_columns(&self, indices: &[usize]) -> Result<Self> {
        let mut columns = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx >= self.columns.len() {
                return Err(SFrameError::Format(format!(
                    "Column index {} out of range ({})",
                    idx,
                    self.columns.len()
                )));
            }
            columns.push(self.columns[idx].clone());
        }
        Ok(SFrameRows {
            columns,
            num_rows: self.num_rows,
        })
    }

    /// Filter rows by a predicate on a specific column.
    pub fn filter_by_column(
        &self,
        column: usize,
        pred: &dyn Fn(&FlexType) -> bool,
    ) -> Result<Self> {
        if column >= self.columns.len() {
            return Err(SFrameError::Format(format!(
                "Column index {} out of range ({})",
                column,
                self.columns.len()
            )));
        }

        let keep_indices = self.columns[column].filter_indices(pred);

        let columns: Vec<ColumnData> = self.columns.iter()
            .map(|col| col.gather(&keep_indices))
            .collect();

        Ok(SFrameRows {
            num_rows: keep_indices.len(),
            columns,
        })
    }

    /// Select rows by indices (gather operation).
    pub fn take(&self, indices: &[usize]) -> Result<Self> {
        for &idx in indices {
            if idx >= self.num_rows {
                return Err(SFrameError::Format(format!(
                    "Row index {} out of range ({})",
                    idx, self.num_rows
                )));
            }
        }

        let columns: Vec<ColumnData> = self.columns.iter()
            .map(|col| col.gather(indices))
            .collect();

        Ok(SFrameRows {
            num_rows: indices.len(),
            columns,
        })
    }

    /// Append another batch (vertically). Columns must have matching types.
    pub fn append(&mut self, other: &SFrameRows) -> Result<()> {
        if self.columns.len() != other.columns.len() {
            return Err(SFrameError::Format(format!(
                "Column count mismatch: {} vs {}",
                self.columns.len(),
                other.columns.len()
            )));
        }
        for (i, (a, b)) in self.columns.iter_mut().zip(other.columns.iter()).enumerate() {
            if a.dtype() != b.dtype() {
                return Err(SFrameError::Type(format!(
                    "Column {} type mismatch: {:?} vs {:?}",
                    i,
                    a.dtype(),
                    b.dtype()
                )));
            }
            a.extend(b)?;
        }
        self.num_rows += other.num_rows;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_rows_and_back() {
        let rows = vec![
            vec![FlexType::Integer(1), FlexType::String("a".into())],
            vec![FlexType::Integer(2), FlexType::String("b".into())],
            vec![FlexType::Integer(3), FlexType::String("c".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];

        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);

        let back = batch.to_rows();
        assert_eq!(back, rows);
    }

    #[test]
    fn test_select_columns() {
        let rows = vec![
            vec![FlexType::Integer(1), FlexType::Float(1.5), FlexType::String("a".into())],
            vec![FlexType::Integer(2), FlexType::Float(2.5), FlexType::String("b".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::Float, FlexTypeEnum::String];

        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let projected = batch.select_columns(&[0, 2]).unwrap();

        assert_eq!(projected.num_columns(), 2);
        assert_eq!(projected.row(0), vec![FlexType::Integer(1), FlexType::String("a".into())]);
        assert_eq!(projected.row(1), vec![FlexType::Integer(2), FlexType::String("b".into())]);
    }

    #[test]
    fn test_filter() {
        let rows = vec![
            vec![FlexType::Integer(1), FlexType::String("keep".into())],
            vec![FlexType::Integer(2), FlexType::String("drop".into())],
            vec![FlexType::Integer(3), FlexType::String("keep".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let filtered = batch
            .filter_by_column(0, &|v| matches!(v, FlexType::Integer(i) if *i != 2))
            .unwrap();

        assert_eq!(filtered.num_rows(), 2);
        assert_eq!(filtered.row(0), vec![FlexType::Integer(1), FlexType::String("keep".into())]);
        assert_eq!(filtered.row(1), vec![FlexType::Integer(3), FlexType::String("keep".into())]);
    }

    #[test]
    fn test_append() {
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::Float];
        let rows1 = vec![vec![FlexType::Integer(1), FlexType::Float(1.0)]];
        let rows2 = vec![vec![FlexType::Integer(2), FlexType::Float(2.0)]];

        let mut batch1 = SFrameRows::from_rows(&rows1, &dtypes).unwrap();
        let batch2 = SFrameRows::from_rows(&rows2, &dtypes).unwrap();
        batch1.append(&batch2).unwrap();

        assert_eq!(batch1.num_rows(), 2);
        assert_eq!(batch1.row(0), vec![FlexType::Integer(1), FlexType::Float(1.0)]);
        assert_eq!(batch1.row(1), vec![FlexType::Integer(2), FlexType::Float(2.0)]);
    }

    #[test]
    fn test_undefined_values() {
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let rows = vec![
            vec![FlexType::Integer(1), FlexType::Undefined],
            vec![FlexType::Undefined, FlexType::String("hello".into())],
        ];

        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.row(0), vec![FlexType::Integer(1), FlexType::Undefined]);
        assert_eq!(batch.row(1), vec![FlexType::Undefined, FlexType::String("hello".into())]);
    }

    #[test]
    fn test_empty_batch() {
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let batch = SFrameRows::empty(&dtypes);
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.dtypes(), vec![FlexTypeEnum::Integer, FlexTypeEnum::String]);
    }

    #[test]
    fn test_flexible_column_mixed_types() {
        let dtypes = [FlexTypeEnum::Undefined];
        let rows = vec![
            vec![FlexType::Integer(42)],
            vec![FlexType::String("hello".into())],
            vec![FlexType::Float(3.14)],
            vec![FlexType::Undefined],
            vec![FlexType::Dict(Arc::from(
                vec![(FlexType::String("k".into()), FlexType::Integer(1))].as_slice(),
            ))],
        ];

        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.dtypes(), vec![FlexTypeEnum::Undefined]);
        assert_eq!(batch.column(0).get(0), FlexType::Integer(42));
        assert_eq!(batch.column(0).get(1), FlexType::String("hello".into()));
        assert_eq!(batch.column(0).get(2), FlexType::Float(3.14));
        assert_eq!(batch.column(0).get(3), FlexType::Undefined);
        match batch.column(0).get(4) {
            FlexType::Dict(d) => assert_eq!(d.len(), 1),
            other => panic!("Expected Dict, got {:?}", other),
        }
    }

    #[test]
    fn test_to_flex_vec_integer() {
        let col = ColumnData::Integer(vec![Some(1), None, Some(3)]);
        let result = col.to_flex_vec();
        assert_eq!(result, vec![
            FlexType::Integer(1),
            FlexType::Undefined,
            FlexType::Integer(3),
        ]);
    }

    #[test]
    fn test_to_flex_vec_float() {
        let col = ColumnData::Float(vec![Some(1.5), None]);
        let result = col.to_flex_vec();
        assert_eq!(result, vec![FlexType::Float(1.5), FlexType::Undefined]);
    }

    #[test]
    fn test_to_flex_vec_string() {
        let col = ColumnData::String(vec![Some("hello".into()), None]);
        let result = col.to_flex_vec();
        assert_eq!(result, vec![FlexType::String("hello".into()), FlexType::Undefined]);
    }

    #[test]
    fn test_to_flex_vec_flexible() {
        let col = ColumnData::Flexible(vec![FlexType::Integer(1), FlexType::String("x".into())]);
        let result = col.to_flex_vec();
        assert_eq!(result, vec![FlexType::Integer(1), FlexType::String("x".into())]);
    }
}
