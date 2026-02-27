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
            FlexTypeEnum::Undefined => ColumnData::Integer(Vec::new()),
        }
    }

    /// Number of elements in this column.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Integer(v) => v.len(),
            ColumnData::Float(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::Vector(v) => v.len(),
            ColumnData::List(v) => v.len(),
            ColumnData::Dict(v) => v.len(),
            ColumnData::DateTime(v) => v.len(),
        }
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
        }
    }

    /// Extend this column with values from another column of the same type.
    pub fn extend(&mut self, other: &ColumnData) -> Result<()> {
        match (self, other) {
            (ColumnData::Integer(a), ColumnData::Integer(b)) => a.extend_from_slice(b),
            (ColumnData::Float(a), ColumnData::Float(b)) => a.extend_from_slice(b),
            (ColumnData::String(a), ColumnData::String(b)) => a.extend_from_slice(b),
            (ColumnData::Vector(a), ColumnData::Vector(b)) => a.extend_from_slice(b),
            (ColumnData::List(a), ColumnData::List(b)) => a.extend_from_slice(b),
            (ColumnData::Dict(a), ColumnData::Dict(b)) => a.extend_from_slice(b),
            (ColumnData::DateTime(a), ColumnData::DateTime(b)) => a.extend_from_slice(b),
            (a, b) => {
                return Err(SFrameError::Type(format!(
                    "Cannot extend {:?} column with {:?} column",
                    a.dtype(),
                    b.dtype()
                )));
            }
        }
        Ok(())
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

        let mut columns = Vec::with_capacity(dtypes.len());
        for (col_idx, (data, &dtype)) in column_data.iter().zip(dtypes.iter()).enumerate() {
            if data.len() != num_rows {
                return Err(SFrameError::Format(format!(
                    "Column {} has {} rows, expected {}",
                    col_idx,
                    data.len(),
                    num_rows
                )));
            }
            let mut col = ColumnData::empty(dtype);
            for val in data {
                col.push(val)?;
            }
            columns.push(col);
        }

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

        // Collect indices of rows that pass the predicate
        let mut keep_indices: Vec<usize> = Vec::new();
        for i in 0..self.num_rows {
            let val = self.columns[column].get(i);
            if pred(&val) {
                keep_indices.push(i);
            }
        }

        self.take(&keep_indices)
    }

    /// Select rows by indices (gather operation).
    pub fn take(&self, indices: &[usize]) -> Result<Self> {
        let dtypes = self.dtypes();
        let mut new_columns: Vec<ColumnData> =
            dtypes.iter().map(|&dt| ColumnData::empty(dt)).collect();

        for &idx in indices {
            if idx >= self.num_rows {
                return Err(SFrameError::Format(format!(
                    "Row index {} out of range ({})",
                    idx, self.num_rows
                )));
            }
            for (col_idx, col) in self.columns.iter().enumerate() {
                let val = col.get(idx);
                new_columns[col_idx].push(&val)?;
            }
        }

        Ok(SFrameRows {
            num_rows: indices.len(),
            columns: new_columns,
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
}
