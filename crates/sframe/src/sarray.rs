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

    // === Phase 10.1: Core operations ===

    /// Return the last n values.
    pub fn tail(&self, n: usize) -> Result<Vec<FlexType>> {
        let all = self.to_vec()?;
        let start = all.len().saturating_sub(n);
        Ok(all[start..].to_vec())
    }

    /// Sort the array.
    pub fn sort(&self, ascending: bool) -> Result<SArray> {
        let mut values = self.to_vec()?;
        values.sort_by(|a, b| {
            let cmp = a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
            if ascending { cmp } else { cmp.reverse() }
        });
        SArray::from_vec(values, self.dtype)
    }

    /// Deduplicated values (preserves first occurrence order).
    pub fn unique(&self) -> Result<SArray> {
        let values = self.to_vec()?;
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for v in values {
            let key = format!("{:?}", v);
            if seen.insert(key) {
                result.push(v);
            }
        }
        SArray::from_vec(result, self.dtype)
    }

    /// Concatenate with another SArray.
    pub fn append(&self, other: &SArray) -> Result<SArray> {
        let plan = PlannerNode::append(self.plan.clone(), other.plan.clone());
        let new_len = match (self.len, other.len) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        Ok(SArray {
            plan: PlannerNode::project(plan, vec![self.column_index]),
            dtype: self.dtype,
            len: new_len,
            column_index: 0,
        })
    }

    /// Random sample of the array (fraction between 0.0 and 1.0).
    pub fn sample(&self, fraction: f64, seed: Option<u64>) -> Result<SArray> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let values = self.to_vec()?;
        let mut result = Vec::new();
        let seed = seed.unwrap_or(42);

        for (i, v) in values.into_iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            (seed, i as u64).hash(&mut hasher);
            let hash = hasher.finish();
            let threshold = (fraction * u64::MAX as f64) as u64;
            if hash < threshold {
                result.push(v);
            }
        }

        SArray::from_vec(result, self.dtype)
    }

    // === Phase 10.2: Missing value handling ===

    /// Count Undefined values.
    pub fn countna(&self) -> Result<u64> {
        let values = self.to_vec()?;
        Ok(values.iter().filter(|v| matches!(v, FlexType::Undefined)).count() as u64)
    }

    /// Remove Undefined values.
    pub fn dropna(&self) -> SArray {
        self.filter(Arc::new(|v| !matches!(v, FlexType::Undefined)))
    }

    /// Replace Undefined with a fill value.
    pub fn fillna(&self, value: FlexType) -> SArray {
        let v = value.clone();
        self.apply(
            Arc::new(move |x| {
                if matches!(x, FlexType::Undefined) {
                    v.clone()
                } else {
                    x.clone()
                }
            }),
            self.dtype,
        )
    }

    /// Returns Integer array: 1 where Undefined, 0 otherwise.
    pub fn is_na(&self) -> SArray {
        self.apply(
            Arc::new(|v| {
                FlexType::Integer(if matches!(v, FlexType::Undefined) { 1 } else { 0 })
            }),
            FlexTypeEnum::Integer,
        )
    }

    // === Phase 10.3: Numeric operations ===

    /// Clamp values to [lower, upper].
    pub fn clip(&self, lower: FlexType, upper: FlexType) -> SArray {
        let lo = lower.clone();
        let hi = upper.clone();
        self.apply(
            Arc::new(move |v| {
                if v < &lo {
                    lo.clone()
                } else if v > &hi {
                    hi.clone()
                } else {
                    v.clone()
                }
            }),
            self.dtype,
        )
    }

    // === Phase 10.4: Reduction operations ===

    /// Sum of all values.
    pub fn sum(&self) -> Result<FlexType> {
        let values = self.to_vec()?;
        let mut result = FlexType::Integer(0);
        for v in values {
            if !matches!(v, FlexType::Undefined) {
                result = result + v;
            }
        }
        Ok(result)
    }

    /// Minimum value.
    pub fn min_val(&self) -> Result<FlexType> {
        let values = self.to_vec()?;
        let mut result: Option<FlexType> = None;
        for v in values {
            if matches!(v, FlexType::Undefined) {
                continue;
            }
            result = Some(match result {
                None => v,
                Some(cur) => {
                    if v < cur { v } else { cur }
                }
            });
        }
        Ok(result.unwrap_or(FlexType::Undefined))
    }

    /// Maximum value.
    pub fn max_val(&self) -> Result<FlexType> {
        let values = self.to_vec()?;
        let mut result: Option<FlexType> = None;
        for v in values {
            if matches!(v, FlexType::Undefined) {
                continue;
            }
            result = Some(match result {
                None => v,
                Some(cur) => {
                    if v > cur { v } else { cur }
                }
            });
        }
        Ok(result.unwrap_or(FlexType::Undefined))
    }

    /// Mean of numeric values.
    pub fn mean(&self) -> Result<FlexType> {
        let values = self.to_vec()?;
        let mut sum = 0.0;
        let mut count = 0u64;
        for v in &values {
            match v {
                FlexType::Integer(i) => {
                    sum += *i as f64;
                    count += 1;
                }
                FlexType::Float(f) => {
                    sum += f;
                    count += 1;
                }
                _ => {}
            }
        }
        if count == 0 {
            Ok(FlexType::Undefined)
        } else {
            Ok(FlexType::Float(sum / count as f64))
        }
    }

    /// Standard deviation (sample, ddof=1).
    pub fn std_dev(&self, ddof: u8) -> Result<FlexType> {
        match self.variance(ddof)? {
            FlexType::Float(v) => Ok(FlexType::Float(v.sqrt())),
            other => Ok(other),
        }
    }

    /// Variance (sample by default, ddof=1).
    pub fn variance(&self, ddof: u8) -> Result<FlexType> {
        let values = self.to_vec()?;
        let mut count = 0u64;
        let mut mean = 0.0;
        let mut m2 = 0.0;

        for v in &values {
            let x = match v {
                FlexType::Integer(i) => *i as f64,
                FlexType::Float(f) => *f,
                _ => continue,
            };
            count += 1;
            let delta = x - mean;
            mean += delta / count as f64;
            let delta2 = x - mean;
            m2 += delta * delta2;
        }

        if count <= ddof as u64 {
            Ok(FlexType::Undefined)
        } else {
            Ok(FlexType::Float(m2 / (count - ddof as u64) as f64))
        }
    }

    /// True if any element is non-zero (for integer arrays).
    pub fn any(&self) -> Result<bool> {
        let values = self.to_vec()?;
        Ok(values.iter().any(|v| matches!(v, FlexType::Integer(i) if *i != 0)))
    }

    /// True if all elements are non-zero (for integer arrays).
    pub fn all(&self) -> Result<bool> {
        let values = self.to_vec()?;
        Ok(values
            .iter()
            .filter(|v| !matches!(v, FlexType::Undefined))
            .all(|v| matches!(v, FlexType::Integer(i) if *i != 0)))
    }

    /// Count non-zero elements.
    pub fn nnz(&self) -> Result<u64> {
        let values = self.to_vec()?;
        Ok(values
            .iter()
            .filter(|v| match v {
                FlexType::Integer(i) => *i != 0,
                FlexType::Float(f) => *f != 0.0,
                _ => false,
            })
            .count() as u64)
    }

    /// Count Undefined/missing values (alias for countna).
    pub fn num_missing(&self) -> Result<u64> {
        self.countna()
    }

    // === Phase 10.5: String Operations ===

    /// Count word frequencies in each string element.
    ///
    /// Returns a Dict-typed SArray where each element maps word → count.
    pub fn count_bag_of_words(&self, to_lower: bool) -> SArray {
        self.apply(
            Arc::new(move |v| {
                let s = match v {
                    FlexType::String(s) => s.to_string(),
                    _ => return FlexType::Undefined,
                };
                let text = if to_lower { s.to_lowercase() } else { s };
                let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
                for word in text.split_whitespace() {
                    // Strip leading/trailing non-alphanumeric characters
                    let trimmed: String = word.chars()
                        .skip_while(|c| !c.is_alphanumeric())
                        .collect::<String>();
                    let trimmed: String = trimmed.chars().rev()
                        .skip_while(|c| !c.is_alphanumeric())
                        .collect::<String>().chars().rev().collect();
                    if !trimmed.is_empty() {
                        *counts.entry(trimmed).or_insert(0) += 1;
                    }
                }
                let dict: Vec<(FlexType, FlexType)> = counts
                    .into_iter()
                    .map(|(k, v)| (FlexType::String(k.into()), FlexType::Integer(v as i64)))
                    .collect();
                FlexType::Dict(Arc::from(dict))
            }),
            FlexTypeEnum::Dict,
        )
    }

    /// Count character n-gram frequencies in each string element.
    pub fn count_character_ngrams(&self, n: usize, to_lower: bool) -> SArray {
        self.apply(
            Arc::new(move |v| {
                let s = match v {
                    FlexType::String(s) => s.to_string(),
                    _ => return FlexType::Undefined,
                };
                let text = if to_lower { s.to_lowercase() } else { s };
                let chars: Vec<char> = text.chars().collect();
                let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
                if chars.len() >= n {
                    for window in chars.windows(n) {
                        let ngram: String = window.iter().collect();
                        *counts.entry(ngram).or_insert(0) += 1;
                    }
                }
                let dict: Vec<(FlexType, FlexType)> = counts
                    .into_iter()
                    .map(|(k, v)| (FlexType::String(k.into()), FlexType::Integer(v as i64)))
                    .collect();
                FlexType::Dict(Arc::from(dict))
            }),
            FlexTypeEnum::Dict,
        )
    }

    /// Count word n-gram frequencies in each string element.
    pub fn count_ngrams(&self, n: usize, to_lower: bool) -> SArray {
        self.apply(
            Arc::new(move |v| {
                let s = match v {
                    FlexType::String(s) => s.to_string(),
                    _ => return FlexType::Undefined,
                };
                let text = if to_lower { s.to_lowercase() } else { s };
                let words: Vec<&str> = text.split_whitespace().collect();
                let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
                if words.len() >= n {
                    for window in words.windows(n) {
                        let ngram = window.join(" ");
                        *counts.entry(ngram).or_insert(0) += 1;
                    }
                }
                let dict: Vec<(FlexType, FlexType)> = counts
                    .into_iter()
                    .map(|(k, v)| (FlexType::String(k.into()), FlexType::Integer(v as i64)))
                    .collect();
                FlexType::Dict(Arc::from(dict))
            }),
            FlexTypeEnum::Dict,
        )
    }

    /// Check if each string contains a substring.
    /// Returns an Integer SArray (1 if contains, 0 otherwise).
    pub fn contains(&self, substring: &str) -> SArray {
        let sub = substring.to_string();
        self.apply(
            Arc::new(move |v| {
                match v {
                    FlexType::String(s) => {
                        FlexType::Integer(if s.contains(&*sub) { 1 } else { 0 })
                    }
                    _ => FlexType::Integer(0),
                }
            }),
            FlexTypeEnum::Integer,
        )
    }

    // === Phase 10.6: Dict Operations ===

    /// Extract keys from each Dict element as a List.
    pub fn dict_keys(&self) -> SArray {
        self.apply(
            Arc::new(|v| match v {
                FlexType::Dict(d) => {
                    let keys: Vec<FlexType> = d.iter().map(|(k, _)| k.clone()).collect();
                    FlexType::List(Arc::from(keys))
                }
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::List,
        )
    }

    /// Extract values from each Dict element as a List.
    pub fn dict_values(&self) -> SArray {
        self.apply(
            Arc::new(|v| match v {
                FlexType::Dict(d) => {
                    let vals: Vec<FlexType> = d.iter().map(|(_, v)| v.clone()).collect();
                    FlexType::List(Arc::from(vals))
                }
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::List,
        )
    }

    /// Filter dict entries to only include specified keys.
    /// If `exclude` is true, removes the specified keys instead.
    pub fn dict_trim_by_keys(&self, keys: Vec<FlexType>, exclude: bool) -> SArray {
        let keys = Arc::new(keys);
        self.apply(
            Arc::new(move |v| match v {
                FlexType::Dict(d) => {
                    let filtered: Vec<(FlexType, FlexType)> = d
                        .iter()
                        .filter(|(k, _)| {
                            let has_key = keys.contains(k);
                            if exclude { !has_key } else { has_key }
                        })
                        .cloned()
                        .collect();
                    FlexType::Dict(Arc::from(filtered))
                }
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::Dict,
        )
    }

    /// Filter dict entries by value range.
    pub fn dict_trim_by_values(&self, lower: FlexType, upper: FlexType) -> SArray {
        let lo = lower.clone();
        let hi = upper.clone();
        self.apply(
            Arc::new(move |v| match v {
                FlexType::Dict(d) => {
                    let filtered: Vec<(FlexType, FlexType)> = d
                        .iter()
                        .filter(|(_, v)| v >= &lo && v <= &hi)
                        .cloned()
                        .collect();
                    FlexType::Dict(Arc::from(filtered))
                }
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::Dict,
        )
    }

    /// Check if each Dict contains any of the specified keys.
    pub fn dict_has_any_keys(&self, keys: Vec<FlexType>) -> SArray {
        let keys = Arc::new(keys);
        self.apply(
            Arc::new(move |v| match v {
                FlexType::Dict(d) => {
                    let has = d.iter().any(|(k, _)| keys.contains(k));
                    FlexType::Integer(if has { 1 } else { 0 })
                }
                _ => FlexType::Integer(0),
            }),
            FlexTypeEnum::Integer,
        )
    }

    /// Check if each Dict contains all of the specified keys.
    pub fn dict_has_all_keys(&self, keys: Vec<FlexType>) -> SArray {
        let keys = Arc::new(keys);
        self.apply(
            Arc::new(move |v| match v {
                FlexType::Dict(d) => {
                    let dict_keys: Vec<&FlexType> = d.iter().map(|(k, _)| k).collect();
                    let has_all = keys.iter().all(|k| dict_keys.contains(&k));
                    FlexType::Integer(if has_all { 1 } else { 0 })
                }
                _ => FlexType::Integer(0),
            }),
            FlexTypeEnum::Integer,
        )
    }

    // === Phase 10.7: Structured Data Operations ===

    /// Length of each element (string len, vector/list/dict size).
    pub fn item_length(&self) -> SArray {
        self.apply(
            Arc::new(|v| match v {
                FlexType::String(s) => FlexType::Integer(s.len() as i64),
                FlexType::Vector(v) => FlexType::Integer(v.len() as i64),
                FlexType::List(l) => FlexType::Integer(l.len() as i64),
                FlexType::Dict(d) => FlexType::Integer(d.len() as i64),
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::Integer,
        )
    }

    /// Slice vector subarrays.
    pub fn vector_slice(&self, start: usize, end: Option<usize>) -> SArray {
        self.apply(
            Arc::new(move |v| match v {
                FlexType::Vector(vec) => {
                    let s = start.min(vec.len());
                    let e = end.unwrap_or(vec.len()).min(vec.len());
                    if s >= e {
                        FlexType::Vector(Arc::from(Vec::<f64>::new()))
                    } else {
                        FlexType::Vector(Arc::from(vec[s..e].to_vec()))
                    }
                }
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::Vector,
        )
    }

    // === Phase 10.8: Rolling / Windowed Aggregations ===

    /// Rolling sum with a window of [before, after] around each element.
    pub fn rolling_sum(&self, before: usize, after: usize, min_observations: usize) -> Result<SArray> {
        self.rolling_agg(before, after, min_observations, |vals| {
            vals.iter().cloned().reduce(|a, b| a + b).unwrap_or(FlexType::Integer(0))
        })
    }

    /// Rolling mean.
    pub fn rolling_mean(&self, before: usize, after: usize, min_observations: usize) -> Result<SArray> {
        self.rolling_agg(before, after, min_observations, |vals| {
            let count = vals.len() as f64;
            if count == 0.0 {
                return FlexType::Undefined;
            }
            let sum: f64 = vals.iter().map(|v| match v {
                FlexType::Integer(i) => *i as f64,
                FlexType::Float(f) => *f,
                _ => 0.0,
            }).sum();
            FlexType::Float(sum / count)
        })
    }

    /// Rolling min.
    pub fn rolling_min(&self, before: usize, after: usize, min_observations: usize) -> Result<SArray> {
        self.rolling_agg(before, after, min_observations, |vals| {
            vals.iter().cloned().reduce(|a, b| if a < b { a } else { b })
                .unwrap_or(FlexType::Undefined)
        })
    }

    /// Rolling max.
    pub fn rolling_max(&self, before: usize, after: usize, min_observations: usize) -> Result<SArray> {
        self.rolling_agg(before, after, min_observations, |vals| {
            vals.iter().cloned().reduce(|a, b| if a > b { a } else { b })
                .unwrap_or(FlexType::Undefined)
        })
    }

    /// Generic rolling aggregation helper.
    fn rolling_agg(
        &self,
        before: usize,
        after: usize,
        min_observations: usize,
        agg: impl Fn(&[FlexType]) -> FlexType,
    ) -> Result<SArray> {
        let values = self.to_vec()?;
        let n = values.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let start = if i >= before { i - before } else { 0 };
            let end = (i + after + 1).min(n);
            let window: Vec<FlexType> = values[start..end]
                .iter()
                .filter(|v| !matches!(v, FlexType::Undefined))
                .cloned()
                .collect();
            if window.len() >= min_observations {
                result.push(agg(&window));
            } else {
                result.push(FlexType::Undefined);
            }
        }

        SArray::from_vec(result, self.dtype)
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

    // === Phase 10 operation tests ===

    #[test]
    fn test_tail() {
        let sa = SArray::from_vec(
            (0..10).map(|i| FlexType::Integer(i)).collect(),
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let t = sa.tail(3).unwrap();
        assert_eq!(t, vec![FlexType::Integer(7), FlexType::Integer(8), FlexType::Integer(9)]);
    }

    #[test]
    fn test_sort_array() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(3), FlexType::Integer(1), FlexType::Integer(4), FlexType::Integer(1), FlexType::Integer(5)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let sorted = sa.sort(true).unwrap();
        assert_eq!(
            sorted.to_vec().unwrap(),
            vec![FlexType::Integer(1), FlexType::Integer(1), FlexType::Integer(3), FlexType::Integer(4), FlexType::Integer(5)]
        );
    }

    #[test]
    fn test_unique() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(1), FlexType::Integer(3), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let u = sa.unique().unwrap();
        assert_eq!(
            u.to_vec().unwrap(),
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)]
        );
    }

    #[test]
    fn test_dropna() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Undefined, FlexType::Integer(3), FlexType::Undefined],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let cleaned = sa.dropna();
        assert_eq!(
            cleaned.to_vec().unwrap(),
            vec![FlexType::Integer(1), FlexType::Integer(3)]
        );
    }

    #[test]
    fn test_fillna() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Undefined, FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let filled = sa.fillna(FlexType::Integer(0));
        assert_eq!(
            filled.to_vec().unwrap(),
            vec![FlexType::Integer(1), FlexType::Integer(0), FlexType::Integer(3)]
        );
    }

    #[test]
    fn test_is_na() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Undefined, FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let na_mask = sa.is_na();
        assert_eq!(
            na_mask.to_vec().unwrap(),
            vec![FlexType::Integer(0), FlexType::Integer(1), FlexType::Integer(0)]
        );
    }

    #[test]
    fn test_countna() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Undefined, FlexType::Undefined],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        assert_eq!(sa.countna().unwrap(), 2);
    }

    #[test]
    fn test_sum_reduction() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        assert_eq!(sa.sum().unwrap(), FlexType::Integer(6));
    }

    #[test]
    fn test_min_max_reduction() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(3), FlexType::Integer(1), FlexType::Integer(5)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        assert_eq!(sa.min_val().unwrap(), FlexType::Integer(1));
        assert_eq!(sa.max_val().unwrap(), FlexType::Integer(5));
    }

    #[test]
    fn test_mean_reduction() {
        let sa = SArray::from_vec(
            vec![FlexType::Float(2.0), FlexType::Float(4.0), FlexType::Float(6.0)],
            FlexTypeEnum::Float,
        )
        .unwrap();

        match sa.mean().unwrap() {
            FlexType::Float(v) => assert!((v - 4.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_variance_reduction() {
        let sa = SArray::from_vec(
            vec![FlexType::Float(2.0), FlexType::Float(4.0), FlexType::Float(4.0), FlexType::Float(4.0), FlexType::Float(5.0), FlexType::Float(5.0), FlexType::Float(7.0), FlexType::Float(9.0)],
            FlexTypeEnum::Float,
        )
        .unwrap();

        // Population variance = 4.0
        match sa.variance(0).unwrap() {
            FlexType::Float(v) => assert!((v - 4.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_any_all() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(0), FlexType::Integer(1), FlexType::Integer(0)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        assert!(sa.any().unwrap());
        assert!(!sa.all().unwrap());

        let sa2 = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(1)],
            FlexTypeEnum::Integer,
        )
        .unwrap();
        assert!(sa2.all().unwrap());
    }

    #[test]
    fn test_nnz() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(0), FlexType::Integer(1), FlexType::Integer(0), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        assert_eq!(sa.nnz().unwrap(), 2);
    }

    #[test]
    fn test_clip() {
        let sa = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(5), FlexType::Integer(10)],
            FlexTypeEnum::Integer,
        )
        .unwrap();

        let clipped = sa.clip(FlexType::Integer(3), FlexType::Integer(7));
        assert_eq!(
            clipped.to_vec().unwrap(),
            vec![FlexType::Integer(3), FlexType::Integer(5), FlexType::Integer(7)]
        );
    }

    // === Phase 10.5-10.8 Tests ===

    #[test]
    fn test_count_bag_of_words() {
        let sa = SArray::from_vec(
            vec![
                FlexType::String("hello world hello".into()),
                FlexType::String("foo bar".into()),
            ],
            FlexTypeEnum::String,
        ).unwrap();

        let bow = sa.count_bag_of_words(true);
        let vals = bow.to_vec().unwrap();
        // First element: {"hello": 2, "world": 1}
        if let FlexType::Dict(d) = &vals[0] {
            let hello_count = d.iter().find(|(k, _)| k == &FlexType::String("hello".into())).map(|(_, v)| v.clone());
            assert_eq!(hello_count, Some(FlexType::Integer(2)));
        } else {
            panic!("Expected Dict");
        }
    }

    #[test]
    fn test_count_character_ngrams() {
        let sa = SArray::from_vec(
            vec![FlexType::String("abcabc".into())],
            FlexTypeEnum::String,
        ).unwrap();

        let ngrams = sa.count_character_ngrams(2, false);
        let vals = ngrams.to_vec().unwrap();
        if let FlexType::Dict(d) = &vals[0] {
            // "ab" appears 2 times, "bc" appears 2 times, "ca" appears 1 time
            let ab_count = d.iter().find(|(k, _)| k == &FlexType::String("ab".into())).map(|(_, v)| v.clone());
            assert_eq!(ab_count, Some(FlexType::Integer(2)));
        } else {
            panic!("Expected Dict");
        }
    }

    #[test]
    fn test_count_ngrams() {
        let sa = SArray::from_vec(
            vec![FlexType::String("the cat sat on the mat".into())],
            FlexTypeEnum::String,
        ).unwrap();

        let ngrams = sa.count_ngrams(2, true);
        let vals = ngrams.to_vec().unwrap();
        if let FlexType::Dict(d) = &vals[0] {
            // "the cat" appears 1 time, "the mat" appears 1 time
            assert!(d.len() >= 4); // 5 bigrams from 6 words
        } else {
            panic!("Expected Dict");
        }
    }

    #[test]
    fn test_contains() {
        let sa = SArray::from_vec(
            vec![
                FlexType::String("hello world".into()),
                FlexType::String("goodbye".into()),
                FlexType::String("hello there".into()),
            ],
            FlexTypeEnum::String,
        ).unwrap();

        let result = sa.contains("hello");
        assert_eq!(
            result.to_vec().unwrap(),
            vec![FlexType::Integer(1), FlexType::Integer(0), FlexType::Integer(1)]
        );
    }

    #[test]
    fn test_dict_keys_values() {
        let d1: Vec<(FlexType, FlexType)> = vec![
            (FlexType::String("a".into()), FlexType::Integer(1)),
            (FlexType::String("b".into()), FlexType::Integer(2)),
        ];
        let sa = SArray::from_vec(
            vec![FlexType::Dict(Arc::from(d1))],
            FlexTypeEnum::Dict,
        ).unwrap();

        let keys = sa.dict_keys().to_vec().unwrap();
        if let FlexType::List(l) = &keys[0] {
            assert_eq!(l.len(), 2);
            assert!(l.contains(&FlexType::String("a".into())));
        } else { panic!("Expected List"); }

        let vals = sa.dict_values().to_vec().unwrap();
        if let FlexType::List(l) = &vals[0] {
            assert_eq!(l.len(), 2);
            assert!(l.contains(&FlexType::Integer(1)));
        } else { panic!("Expected List"); }
    }

    #[test]
    fn test_dict_trim_by_keys() {
        let d1: Vec<(FlexType, FlexType)> = vec![
            (FlexType::String("a".into()), FlexType::Integer(1)),
            (FlexType::String("b".into()), FlexType::Integer(2)),
            (FlexType::String("c".into()), FlexType::Integer(3)),
        ];
        let sa = SArray::from_vec(
            vec![FlexType::Dict(Arc::from(d1))],
            FlexTypeEnum::Dict,
        ).unwrap();

        // Include only "a" and "c"
        let trimmed = sa.dict_trim_by_keys(
            vec![FlexType::String("a".into()), FlexType::String("c".into())],
            false,
        );
        if let FlexType::Dict(d) = &trimmed.to_vec().unwrap()[0] {
            assert_eq!(d.len(), 2);
        } else { panic!("Expected Dict"); }

        // Exclude "b"
        let trimmed = sa.dict_trim_by_keys(
            vec![FlexType::String("b".into())],
            true,
        );
        if let FlexType::Dict(d) = &trimmed.to_vec().unwrap()[0] {
            assert_eq!(d.len(), 2); // "a" and "c"
        } else { panic!("Expected Dict"); }
    }

    #[test]
    fn test_dict_has_keys() {
        let d1: Vec<(FlexType, FlexType)> = vec![
            (FlexType::String("a".into()), FlexType::Integer(1)),
            (FlexType::String("b".into()), FlexType::Integer(2)),
        ];
        let sa = SArray::from_vec(
            vec![FlexType::Dict(Arc::from(d1))],
            FlexTypeEnum::Dict,
        ).unwrap();

        let has_any = sa.dict_has_any_keys(vec![FlexType::String("a".into()), FlexType::String("z".into())]);
        assert_eq!(has_any.to_vec().unwrap(), vec![FlexType::Integer(1)]);

        let has_all = sa.dict_has_all_keys(vec![FlexType::String("a".into()), FlexType::String("z".into())]);
        assert_eq!(has_all.to_vec().unwrap(), vec![FlexType::Integer(0)]);

        let has_all2 = sa.dict_has_all_keys(vec![FlexType::String("a".into()), FlexType::String("b".into())]);
        assert_eq!(has_all2.to_vec().unwrap(), vec![FlexType::Integer(1)]);
    }

    #[test]
    fn test_item_length() {
        // Test with strings
        let sa_str = SArray::from_vec(
            vec![FlexType::String("hello".into()), FlexType::String("hi".into())],
            FlexTypeEnum::String,
        ).unwrap();
        let lengths = sa_str.item_length().to_vec().unwrap();
        assert_eq!(lengths[0], FlexType::Integer(5));
        assert_eq!(lengths[1], FlexType::Integer(2));

        // Test with vectors
        let sa_vec = SArray::from_vec(
            vec![FlexType::Vector(Arc::from(vec![1.0, 2.0, 3.0]))],
            FlexTypeEnum::Vector,
        ).unwrap();
        let lengths = sa_vec.item_length().to_vec().unwrap();
        assert_eq!(lengths[0], FlexType::Integer(3));
    }

    #[test]
    fn test_vector_slice() {
        let sa = SArray::from_vec(
            vec![
                FlexType::Vector(Arc::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
                FlexType::Vector(Arc::from(vec![10.0, 20.0])),
            ],
            FlexTypeEnum::Vector,
        ).unwrap();

        let sliced = sa.vector_slice(1, Some(3));
        let vals = sliced.to_vec().unwrap();
        assert_eq!(vals[0], FlexType::Vector(Arc::from(vec![2.0, 3.0])));
        assert_eq!(vals[1], FlexType::Vector(Arc::from(vec![20.0])));
    }

    #[test]
    fn test_rolling_sum() {
        let sa = SArray::from_vec(
            vec![
                FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3),
                FlexType::Integer(4), FlexType::Integer(5),
            ],
            FlexTypeEnum::Integer,
        ).unwrap();

        // Window: 1 before, 1 after (size 3)
        let rolled = sa.rolling_sum(1, 1, 1).unwrap();
        let vals = rolled.to_vec().unwrap();
        assert_eq!(vals[0], FlexType::Integer(3));  // 1+2
        assert_eq!(vals[1], FlexType::Integer(6));  // 1+2+3
        assert_eq!(vals[2], FlexType::Integer(9));  // 2+3+4
        assert_eq!(vals[3], FlexType::Integer(12)); // 3+4+5
        assert_eq!(vals[4], FlexType::Integer(9));  // 4+5
    }

    #[test]
    fn test_rolling_mean() {
        let sa = SArray::from_vec(
            vec![
                FlexType::Float(1.0), FlexType::Float(2.0), FlexType::Float(3.0),
                FlexType::Float(4.0), FlexType::Float(5.0),
            ],
            FlexTypeEnum::Float,
        ).unwrap();

        let rolled = sa.rolling_mean(1, 1, 1).unwrap();
        let vals = rolled.to_vec().unwrap();
        if let FlexType::Float(f) = vals[2] {
            assert!((f - 3.0).abs() < 1e-9); // mean of [2, 3, 4]
        } else { panic!("Expected Float"); }
    }

    #[test]
    fn test_rolling_min_max() {
        let sa = SArray::from_vec(
            vec![
                FlexType::Integer(5), FlexType::Integer(1), FlexType::Integer(3),
                FlexType::Integer(2), FlexType::Integer(4),
            ],
            FlexTypeEnum::Integer,
        ).unwrap();

        let rolled_min = sa.rolling_min(1, 1, 1).unwrap();
        let min_vals = rolled_min.to_vec().unwrap();
        assert_eq!(min_vals[1], FlexType::Integer(1)); // min of [5, 1, 3]

        let rolled_max = sa.rolling_max(1, 1, 1).unwrap();
        let max_vals = rolled_max.to_vec().unwrap();
        assert_eq!(max_vals[1], FlexType::Integer(5)); // max of [5, 1, 3]
    }
}
