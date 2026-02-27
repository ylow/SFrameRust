//! SArray — lazy columnar array with deferred execution.
//!
//! An SArray can be:
//! - Materialized: backed by in-memory data
//! - Lazy: backed by a PlannerNode DAG that gets compiled and executed on demand

use std::sync::Arc;

use sframe_query::batch::{ColumnData, SFrameRows};
use sframe_query::execute::{compile, materialize_sync};
use sframe_query::planner::PlannerNode;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

/// A lazy columnar array. Operations build a plan DAG; execution happens on
/// `materialize()`, `head()`, `to_vec()`, or `Display`.
#[derive(Clone)]
pub struct SArray {
    /// The planner node representing this array's computation.
    plan: Arc<PlannerNode>,
    /// The data type of this column.
    dtype: FlexTypeEnum,
    /// Known length, if available (e.g., from a source).
    len: Option<u64>,
    /// Column index in the plan's output (for multi-column plans).
    column_index: usize,
}

impl SArray {
    /// Create an SArray from a vector of values.
    pub fn from_vec(values: Vec<FlexType>, dtype: FlexTypeEnum) -> Result<Self> {
        let len = values.len() as u64;
        let mut col = ColumnData::empty(dtype);
        for v in &values {
            col.push(v)?;
        }
        let batch = SFrameRows::new(vec![col])?;
        let plan = PlannerNode::materialized(batch);
        Ok(SArray {
            plan,
            dtype,
            len: Some(len),
            column_index: 0,
        })
    }

    /// Create an SArray backed by a planner node and column index.
    pub(crate) fn from_plan(
        plan: Arc<PlannerNode>,
        dtype: FlexTypeEnum,
        len: Option<u64>,
        column_index: usize,
    ) -> Self {
        SArray {
            plan,
            dtype,
            len,
            column_index,
        }
    }

    /// The data type of this array.
    pub fn dtype(&self) -> FlexTypeEnum {
        self.dtype
    }

    /// Known length (may require materialization if unknown).
    pub fn len(&self) -> Result<u64> {
        if let Some(l) = self.len {
            return Ok(l);
        }
        let data = self.materialize_column()?;
        Ok(data.len() as u64)
    }

    /// Materialize and return all values.
    pub fn to_vec(&self) -> Result<Vec<FlexType>> {
        let data = self.materialize_column()?;
        Ok((0..data.len()).map(|i| data.get(i)).collect())
    }

    /// Return the first n values.
    pub fn head(&self, n: usize) -> Result<Vec<FlexType>> {
        let data = self.materialize_column()?;
        let take = n.min(data.len());
        Ok((0..take).map(|i| data.get(i)).collect())
    }

    /// Apply a unary transform, producing a new SArray.
    pub fn apply(
        &self,
        func: Arc<dyn Fn(&FlexType) -> FlexType + Send + Sync>,
        output_type: FlexTypeEnum,
    ) -> SArray {
        let plan = PlannerNode::transform(
            self.plan.clone(),
            self.column_index,
            func,
            output_type,
        );
        // The new column is appended at the end
        let new_col_index = self.column_index + 1; // simplified — transform appends
        SArray {
            plan: PlannerNode::project(plan, vec![new_col_index]),
            dtype: output_type,
            len: self.len,
            column_index: 0,
        }
    }

    /// Filter elements by a predicate.
    pub fn filter(
        &self,
        pred: Arc<dyn Fn(&FlexType) -> bool + Send + Sync>,
    ) -> SArray {
        let plan = PlannerNode::filter(self.plan.clone(), self.column_index, pred);
        SArray {
            plan: PlannerNode::project(plan, vec![self.column_index]),
            dtype: self.dtype,
            len: None, // length unknown after filter
            column_index: 0,
        }
    }

    /// Access the underlying plan node.
    pub(crate) fn plan(&self) -> &Arc<PlannerNode> {
        &self.plan
    }

    /// Column index in the plan output.
    pub(crate) fn column_index(&self) -> usize {
        self.column_index
    }

    /// Materialize the column data.
    fn materialize_column(&self) -> Result<ColumnData> {
        let stream = compile(&self.plan)?;
        let batch = materialize_sync(stream)?;
        if self.column_index >= batch.num_columns() {
            return Err(SFrameError::Format(format!(
                "Column index {} out of range ({})",
                self.column_index,
                batch.num_columns()
            )));
        }
        Ok(batch.column(self.column_index).clone())
    }
}

impl std::fmt::Display for SArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let values = match self.head(10) {
            Ok(v) => v,
            Err(e) => return write!(f, "[SArray error: {}]", e),
        };
        let len = self.len.map(|l| l.to_string()).unwrap_or("?".to_string());

        writeln!(f, "dtype: {}", self.dtype)?;
        writeln!(f, "Rows: {}", len)?;
        writeln!(f, "[")?;
        for (i, v) in values.iter().enumerate() {
            if i >= 5 && values.len() > 5 {
                writeln!(f, "  ...")?;
                break;
            }
            writeln!(f, "  {},", v)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        assert_eq!(sa.dtype(), FlexTypeEnum::Integer);
        assert_eq!(sa.len().unwrap(), 3);
        assert_eq!(
            sa.to_vec().unwrap(),
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)]
        );
    }

    #[test]
    fn test_head() {
        let sa = SArray::from_vec(
            (0..100).map(|i| FlexType::Integer(i)).collect(),
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let head = sa.head(5).unwrap();
        assert_eq!(head.len(), 5);
        assert_eq!(head[0], FlexType::Integer(0));
        assert_eq!(head[4], FlexType::Integer(4));
    }

    #[test]
    fn test_apply() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let doubled = sa.apply(
            Arc::new(|v| match v {
                FlexType::Integer(i) => FlexType::Integer(i * 2),
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::Integer,
        );

        assert_eq!(
            doubled.to_vec().unwrap(),
            vec![FlexType::Integer(2), FlexType::Integer(4), FlexType::Integer(6)]
        );
    }

    #[test]
    fn test_filter() {
        let sa = SArray::from_vec(
            vec![
                FlexType::Integer(1),
                FlexType::Integer(2),
                FlexType::Integer(3),
                FlexType::Integer(4),
            ],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let filtered = sa.filter(Arc::new(|v| {
            matches!(v, FlexType::Integer(i) if *i > 2)
        }));

        assert_eq!(
            filtered.to_vec().unwrap(),
            vec![FlexType::Integer(3), FlexType::Integer(4)]
        );
    }

    #[test]
    fn test_display() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let s = format!("{}", sa);
        assert!(s.contains("dtype: integer"));
        assert!(s.contains("Rows: 2"));
    }
}
