//! SArray — lazy columnar array with deferred execution.
//!
//! An SArray can be:
//! - Materialized: backed by in-memory data
//! - Lazy: backed by a PlannerNode DAG that gets compiled and executed on demand

use std::sync::Arc;

use sframe_query::batch::{ColumnData, SFrameRows};
use sframe_query::execute::{compile, materialize_head_sync, materialize_sync};
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
    ///
    /// Only pulls enough batches from the stream to fill n rows, then stops.
    pub fn head(&self, n: usize) -> Result<Vec<FlexType>> {
        if n == 0 {
            return Ok(vec![]);
        }
        let stream = compile(&self.plan)?;
        let batch = materialize_head_sync(stream, n)?;
        if batch.num_rows() == 0 {
            return Ok(vec![]);
        }
        if self.column_index >= batch.num_columns() {
            return Err(SFrameError::Format(format!(
                "Column index {} out of range ({})",
                self.column_index,
                batch.num_columns()
            )));
        }
        let col = batch.column(self.column_index);
        Ok((0..col.len()).map(|i| col.get(i)).collect())
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

    // === Element-wise binary operations ===

    /// Element-wise addition: self + other.
    /// int+int→int, float+float→float, int+float→float, string+string→concat,
    /// vector+vector→element-wise.
    pub fn add(&self, other: &SArray) -> Result<SArray> {
        self.binary_op(other, Arc::new(|a, b| a.clone() + b.clone()), "add")
    }

    /// Element-wise subtraction: self - other.
    pub fn sub(&self, other: &SArray) -> Result<SArray> {
        self.binary_op(other, Arc::new(|a, b| a.clone() - b.clone()), "sub")
    }

    /// Element-wise multiplication: self * other.
    pub fn mul(&self, other: &SArray) -> Result<SArray> {
        self.binary_op(other, Arc::new(|a, b| a.clone() * b.clone()), "mul")
    }

    /// Element-wise division: self / other. Always returns Float.
    pub fn div(&self, other: &SArray) -> Result<SArray> {
        self.binary_op(other, Arc::new(|a, b| a.clone() / b.clone()), "div")
    }

    /// Element-wise remainder: self % other. Integer only.
    pub fn rem(&self, other: &SArray) -> Result<SArray> {
        self.binary_op(other, Arc::new(|a, b| a.clone() % b.clone()), "rem")
    }

    // === Scalar binary operations ===

    /// Add a scalar to each element.
    pub fn add_scalar(&self, scalar: FlexType) -> SArray {
        let s = scalar.clone();
        self.apply(
            Arc::new(move |v| v.clone() + s.clone()),
            result_type_for_add(self.dtype, scalar.type_enum()),
        )
    }

    /// Subtract a scalar from each element.
    pub fn sub_scalar(&self, scalar: FlexType) -> SArray {
        let s = scalar.clone();
        self.apply(
            Arc::new(move |v| v.clone() - s.clone()),
            result_type_for_arith(self.dtype, scalar.type_enum()),
        )
    }

    /// Multiply each element by a scalar.
    pub fn mul_scalar(&self, scalar: FlexType) -> SArray {
        let s = scalar.clone();
        self.apply(
            Arc::new(move |v| v.clone() * s.clone()),
            result_type_for_arith(self.dtype, scalar.type_enum()),
        )
    }

    /// Divide each element by a scalar. Always returns Float.
    pub fn div_scalar(&self, scalar: FlexType) -> SArray {
        let s = scalar.clone();
        self.apply(
            Arc::new(move |v| v.clone() / s.clone()),
            FlexTypeEnum::Float,
        )
    }

    // === Comparison operations (return Integer 0/1) ===

    /// Element-wise equality: returns Integer array (1 where equal, 0 where not).
    pub fn eq(&self, other: &SArray) -> Result<SArray> {
        self.comparison_op(other, Arc::new(|a, b| a == b))
    }

    /// Element-wise inequality.
    pub fn ne(&self, other: &SArray) -> Result<SArray> {
        self.comparison_op(other, Arc::new(|a, b| a != b))
    }

    /// Element-wise less-than.
    pub fn lt(&self, other: &SArray) -> Result<SArray> {
        self.comparison_op(other, Arc::new(|a, b| a < b))
    }

    /// Element-wise less-than-or-equal.
    pub fn le(&self, other: &SArray) -> Result<SArray> {
        self.comparison_op(other, Arc::new(|a, b| a <= b))
    }

    /// Element-wise greater-than.
    pub fn gt(&self, other: &SArray) -> Result<SArray> {
        self.comparison_op(other, Arc::new(|a, b| a > b))
    }

    /// Element-wise greater-than-or-equal.
    pub fn ge(&self, other: &SArray) -> Result<SArray> {
        self.comparison_op(other, Arc::new(|a, b| a >= b))
    }

    // === Logical operations ===

    /// Logical AND: treats non-zero integers as true.
    pub fn and(&self, other: &SArray) -> Result<SArray> {
        self.binary_op(
            other,
            Arc::new(|a, b| {
                let a_true = matches!(a, FlexType::Integer(i) if *i != 0);
                let b_true = matches!(b, FlexType::Integer(i) if *i != 0);
                FlexType::Integer(if a_true && b_true { 1 } else { 0 })
            }),
            "and",
        )
    }

    /// Logical OR: treats non-zero integers as true.
    pub fn or(&self, other: &SArray) -> Result<SArray> {
        self.binary_op(
            other,
            Arc::new(|a, b| {
                let a_true = matches!(a, FlexType::Integer(i) if *i != 0);
                let b_true = matches!(b, FlexType::Integer(i) if *i != 0);
                FlexType::Integer(if a_true || b_true { 1 } else { 0 })
            }),
            "or",
        )
    }

    // === Internal helpers ===

    /// Binary operation between two SArrays.
    /// Materializes the right-hand side and applies element-wise.
    fn binary_op(
        &self,
        other: &SArray,
        func: Arc<dyn Fn(&FlexType, &FlexType) -> FlexType + Send + Sync>,
        _op_name: &str,
    ) -> Result<SArray> {
        // Materialize the right-hand side
        let rhs_values = other.to_vec()?;
        let rhs = Arc::new(rhs_values);

        let output_type = infer_binary_output_type(self.dtype, other.dtype, &func);

        // Build a transform that zips with the materialized right side
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let f = func.clone();
        let rhs_clone = rhs.clone();
        let cnt = counter.clone();

        let result = self.apply(
            Arc::new(move |lhs_val| {
                let idx = cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if idx < rhs_clone.len() {
                    f(lhs_val, &rhs_clone[idx])
                } else {
                    FlexType::Undefined
                }
            }),
            output_type,
        );

        Ok(result)
    }

    /// Comparison operation returning Integer 0/1 array.
    fn comparison_op(
        &self,
        other: &SArray,
        pred: Arc<dyn Fn(&FlexType, &FlexType) -> bool + Send + Sync>,
    ) -> Result<SArray> {
        self.binary_op(
            other,
            Arc::new(move |a, b| {
                FlexType::Integer(if pred(a, b) { 1 } else { 0 })
            }),
            "cmp",
        )
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

/// Determine output type for addition.
fn result_type_for_add(left: FlexTypeEnum, right: FlexTypeEnum) -> FlexTypeEnum {
    match (left, right) {
        (FlexTypeEnum::Integer, FlexTypeEnum::Integer) => FlexTypeEnum::Integer,
        (FlexTypeEnum::Float, FlexTypeEnum::Float) => FlexTypeEnum::Float,
        (FlexTypeEnum::Integer, FlexTypeEnum::Float)
        | (FlexTypeEnum::Float, FlexTypeEnum::Integer) => FlexTypeEnum::Float,
        (FlexTypeEnum::String, FlexTypeEnum::String) => FlexTypeEnum::String,
        (FlexTypeEnum::Vector, FlexTypeEnum::Vector) => FlexTypeEnum::Vector,
        _ => FlexTypeEnum::Float, // fallback
    }
}

/// Determine output type for arithmetic (sub, mul).
fn result_type_for_arith(left: FlexTypeEnum, right: FlexTypeEnum) -> FlexTypeEnum {
    match (left, right) {
        (FlexTypeEnum::Integer, FlexTypeEnum::Integer) => FlexTypeEnum::Integer,
        (FlexTypeEnum::Float, FlexTypeEnum::Float) => FlexTypeEnum::Float,
        (FlexTypeEnum::Integer, FlexTypeEnum::Float)
        | (FlexTypeEnum::Float, FlexTypeEnum::Integer) => FlexTypeEnum::Float,
        (FlexTypeEnum::Vector, FlexTypeEnum::Vector) => FlexTypeEnum::Vector,
        (FlexTypeEnum::Vector, _) | (_, FlexTypeEnum::Vector) => FlexTypeEnum::Vector,
        _ => FlexTypeEnum::Float,
    }
}

/// Infer the output type of a binary operation by probing with sample values.
fn infer_binary_output_type(
    left_dtype: FlexTypeEnum,
    right_dtype: FlexTypeEnum,
    _func: &Arc<dyn Fn(&FlexType, &FlexType) -> FlexType + Send + Sync>,
) -> FlexTypeEnum {
    result_type_for_arith(left_dtype, right_dtype)
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

    // === Element-wise operation tests ===

    #[test]
    fn test_add_arrays() {
        let a = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();
        let b = SArray::from_vec(
            vec![FlexType::Integer(10), FlexType::Integer(20), FlexType::Integer(30)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let c = a.add(&b).unwrap();
        assert_eq!(
            c.to_vec().unwrap(),
            vec![FlexType::Integer(11), FlexType::Integer(22), FlexType::Integer(33)]
        );
    }

    #[test]
    fn test_sub_arrays() {
        let a = SArray::from_vec(
            vec![FlexType::Integer(10), FlexType::Integer(20)],
            FlexTypeEnum::Integer,
        )
        .unwrap();
        let b = SArray::from_vec(
            vec![FlexType::Integer(3), FlexType::Integer(7)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();
        assert_eq!(
            c.to_vec().unwrap(),
            vec![FlexType::Integer(7), FlexType::Integer(13)]
        );
    }

    #[test]
    fn test_mul_scalar() {
        let a = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let c = a.mul_scalar(FlexType::Integer(5));
        assert_eq!(
            c.to_vec().unwrap(),
            vec![FlexType::Integer(5), FlexType::Integer(10), FlexType::Integer(15)]
        );
    }

    #[test]
    fn test_div_scalar() {
        let a = SArray::from_vec(
            vec![FlexType::Integer(10), FlexType::Integer(20)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let c = a.div_scalar(FlexType::Integer(2));
        let vals = c.to_vec().unwrap();
        match &vals[0] {
            FlexType::Float(v) => assert!((v - 5.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_eq_comparison() {
        let a = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();
        let b = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(99), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let c = a.eq(&b).unwrap();
        assert_eq!(
            c.to_vec().unwrap(),
            vec![FlexType::Integer(1), FlexType::Integer(0), FlexType::Integer(1)]
        );
    }

    #[test]
    fn test_lt_comparison() {
        let a = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(5), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();
        let b = SArray::from_vec(
            vec![FlexType::Integer(2), FlexType::Integer(2), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let c = a.lt(&b).unwrap();
        assert_eq!(
            c.to_vec().unwrap(),
            vec![FlexType::Integer(1), FlexType::Integer(0), FlexType::Integer(0)]
        );
    }

    #[test]
    fn test_logical_and() {
        let a = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(0), FlexType::Integer(1)],
            FlexTypeEnum::Integer,
        )
        .unwrap();
        let b = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(1), FlexType::Integer(0)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let c = a.and(&b).unwrap();
        assert_eq!(
            c.to_vec().unwrap(),
            vec![FlexType::Integer(1), FlexType::Integer(0), FlexType::Integer(0)]
        );
    }

    #[test]
    fn test_logical_or() {
        let a = SArray::from_vec(
            vec![FlexType::Integer(0), FlexType::Integer(0), FlexType::Integer(1)],
            FlexTypeEnum::Integer,
        )
        .unwrap();
        let b = SArray::from_vec(
            vec![FlexType::Integer(0), FlexType::Integer(1), FlexType::Integer(0)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let c = a.or(&b).unwrap();
        assert_eq!(
            c.to_vec().unwrap(),
            vec![FlexType::Integer(0), FlexType::Integer(1), FlexType::Integer(1)]
        );
    }

    #[test]
    fn test_add_scalar() {
        let a = SArray::from_vec(
            vec![FlexType::Float(1.0), FlexType::Float(2.0)],
            FlexTypeEnum::Float,
        )
        .unwrap();

        let c = a.add_scalar(FlexType::Float(10.0));
        let vals = c.to_vec().unwrap();
        match &vals[0] {
            FlexType::Float(v) => assert!((v - 11.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }
}
