//! Logical query planner.
//!
//! DAG of PlannerNodes representing query operations. Nodes are Arc-shared
//! so the same subexpression can appear in multiple places.

use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::SFrameRows;

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);

/// Classification of an operator's output rate relative to its input(s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorRate {
    /// Generates data (no inputs). SFrameSource, MaterializedSource, Range.
    Source,
    /// 1:1 row mapping — output rate equals input rate.
    /// Project, Transform, BinaryTransform, GeneralizedTransform, Append, Union.
    Linear,
    /// Output rate differs from input rate (e.g., fewer rows).
    /// Filter, LogicalFilter, Reduce.
    SubLinear,
}

/// A node in the logical query plan DAG.
pub struct PlannerNode {
    pub op: LogicalOp,
    pub inputs: Vec<Arc<PlannerNode>>,
    /// Unique identifier assigned at construction time.
    id: u64,
    /// Deterministic hash of op + params + input content hashes.
    content_hash: u64,
}

/// Logical operations in the query plan.
pub enum LogicalOp {
    /// Read an SFrame from disk (or from cache://).
    SFrameSource {
        path: String,
        column_names: Vec<String>,
        column_types: Vec<FlexTypeEnum>,
        num_rows: u64,
        /// First row to read (inclusive). Default 0.
        begin_row: u64,
        /// Last row to read (exclusive). Default num_rows.
        end_row: u64,
        /// Keeps the backing store alive (e.g. AnonymousStore for cache:// paths).
        _keep_alive: Option<Arc<dyn Send + Sync>>,
    },

    /// Select specific columns by index.
    Project {
        column_indices: Vec<usize>,
    },

    /// Filter rows using a predicate on a specific column.
    Filter {
        column: usize,
        predicate: Arc<dyn Fn(&FlexType) -> bool + Send + Sync>,
    },

    /// Apply a function to one column, producing a new column.
    Transform {
        input_column: usize,
        func: Arc<dyn Fn(&FlexType) -> FlexType + Send + Sync>,
        output_type: FlexTypeEnum,
    },

    /// Apply a function to two columns, producing a new column.
    BinaryTransform {
        left_column: usize,
        right_column: usize,
        func: Arc<dyn Fn(&FlexType, &FlexType) -> FlexType + Send + Sync>,
        output_type: FlexTypeEnum,
    },

    /// Apply a function to the entire row, producing one or more new columns.
    GeneralizedTransform {
        func: Arc<dyn Fn(&[FlexType]) -> Vec<FlexType> + Send + Sync>,
        output_types: Vec<FlexTypeEnum>,
    },

    /// Concatenate two inputs vertically.
    Append,

    /// Generate a range of integers.
    Range {
        start: i64,
        step: i64,
        count: u64,
    },

    /// Reduce all rows using an aggregator, producing a single row.
    Reduce {
        aggregator: Arc<dyn Aggregator>,
    },

    /// Union of multiple inputs (same as Append but for >2 inputs).
    Union,

    /// Logical filter: two inputs of the same row count. A row from
    /// input 0 (data) is emitted only when the corresponding row in
    /// input 1 (mask, single column) is truthy (non-zero).
    LogicalFilter,

    /// An in-memory batch of rows (materialized data).
    MaterializedSource {
        data: Arc<SFrameRows>,
    },
}

impl LogicalOp {
    /// Clone a LogicalOp (needed because it contains Arc closures, not plain Clone).
    pub fn clone_op(&self) -> LogicalOp {
        match self {
            LogicalOp::SFrameSource { path, column_names, column_types, num_rows, begin_row, end_row, _keep_alive } => {
                LogicalOp::SFrameSource {
                    path: path.clone(),
                    column_names: column_names.clone(),
                    column_types: column_types.clone(),
                    num_rows: *num_rows,
                    begin_row: *begin_row,
                    end_row: *end_row,
                    _keep_alive: _keep_alive.clone(),
                }
            }
            LogicalOp::Project { column_indices } => LogicalOp::Project { column_indices: column_indices.clone() },
            LogicalOp::Filter { column, predicate } => LogicalOp::Filter { column: *column, predicate: predicate.clone() },
            LogicalOp::Transform { input_column, func, output_type } => {
                LogicalOp::Transform { input_column: *input_column, func: func.clone(), output_type: *output_type }
            }
            LogicalOp::BinaryTransform { left_column, right_column, func, output_type } => {
                LogicalOp::BinaryTransform {
                    left_column: *left_column, right_column: *right_column,
                    func: func.clone(), output_type: *output_type,
                }
            }
            LogicalOp::GeneralizedTransform { func, output_types } => {
                LogicalOp::GeneralizedTransform { func: func.clone(), output_types: output_types.clone() }
            }
            LogicalOp::LogicalFilter => LogicalOp::LogicalFilter,
            LogicalOp::Append => LogicalOp::Append,
            LogicalOp::Range { start, step, count } => LogicalOp::Range { start: *start, step: *step, count: *count },
            LogicalOp::Reduce { aggregator } => LogicalOp::Reduce { aggregator: aggregator.clone() },
            LogicalOp::Union => LogicalOp::Union,
            LogicalOp::MaterializedSource { data } => LogicalOp::MaterializedSource { data: data.clone() },
        }
    }

    /// Return this operator's rate classification.
    pub fn rate(&self) -> OperatorRate {
        match self {
            LogicalOp::SFrameSource { .. } => OperatorRate::Source,
            LogicalOp::MaterializedSource { .. } => OperatorRate::Source,
            LogicalOp::Range { .. } => OperatorRate::Source,
            LogicalOp::Project { .. } => OperatorRate::Linear,
            LogicalOp::Transform { .. } => OperatorRate::Linear,
            LogicalOp::BinaryTransform { .. } => OperatorRate::Linear,
            LogicalOp::GeneralizedTransform { .. } => OperatorRate::Linear,
            LogicalOp::Append => OperatorRate::Linear,
            LogicalOp::Union => OperatorRate::Linear,
            LogicalOp::Filter { .. } => OperatorRate::SubLinear,
            LogicalOp::LogicalFilter => OperatorRate::SubLinear,
            LogicalOp::Reduce { .. } => OperatorRate::SubLinear,
        }
    }

    /// Whether this multi-input operator consumes inputs in lockstep
    /// (same batch boundaries required) vs sequentially.
    pub fn is_lockstep(&self) -> bool {
        matches!(self, LogicalOp::LogicalFilter)
    }
}

/// Trait for aggregation operations.
pub trait Aggregator: Send + Sync {
    /// Add a row's values to the aggregation state.
    fn add(&mut self, values: &[FlexType]);

    /// Merge another aggregator's state into this one.
    fn merge(&mut self, other: &dyn Aggregator);

    /// Produce the final aggregated value.
    fn finalize(&mut self) -> FlexType;

    /// The output type given input types.
    fn output_type(&self, input_types: &[FlexTypeEnum]) -> FlexTypeEnum;

    /// Clone this aggregator (needed since we can't use Clone with dyn).
    fn box_clone(&self) -> Box<dyn Aggregator>;

    /// Downcast support for merge operations.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Serialize aggregator state to a writer (for spill-to-disk).
    fn save(&self, writer: &mut dyn Write) -> Result<()>;

    /// Deserialize aggregator state from a reader (for spill-to-disk).
    fn load(&mut self, reader: &mut dyn Read) -> Result<()>;

    /// Pre-spill hook called before serialization. Default is no-op.
    fn partial_finalize(&mut self) {}
}

impl PlannerNode {
    /// Create a PlannerNode with a unique id and deterministic content hash.
    pub(crate) fn new(op: LogicalOp, inputs: Vec<Arc<PlannerNode>>) -> Self {
        let id = NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed);
        let content_hash = Self::compute_hash(&op, &inputs, id);
        PlannerNode { op, inputs, id, content_hash }
    }

    /// Return this node's unique identifier.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Return this node's deterministic content hash.
    pub fn content_hash(&self) -> u64 {
        self.content_hash
    }

    /// Compute a deterministic hash from the operator, its parameters, and
    /// the content hashes of its inputs.
    ///
    /// For operators containing opaque closures or aggregators (which cannot
    /// be compared), the node's unique `id` is included in the hash instead,
    /// ensuring distinct nodes produce distinct hashes.
    fn compute_hash(op: &LogicalOp, inputs: &[Arc<PlannerNode>], id: u64) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut h = DefaultHasher::new();
        // Hash the discriminant to distinguish operator types
        std::mem::discriminant(op).hash(&mut h);

        match op {
            LogicalOp::SFrameSource {
                path, column_names, column_types, begin_row, end_row, ..
            } => {
                path.hash(&mut h);
                column_names.hash(&mut h);
                column_types.hash(&mut h);
                begin_row.hash(&mut h);
                end_row.hash(&mut h);
            }
            LogicalOp::MaterializedSource { .. } => {
                // Data is not cheaply hashable; use unique id
                id.hash(&mut h);
            }
            LogicalOp::Range { start, step, count } => {
                start.hash(&mut h);
                step.hash(&mut h);
                count.hash(&mut h);
            }
            LogicalOp::Project { column_indices } => {
                column_indices.hash(&mut h);
                for input in inputs {
                    input.content_hash.hash(&mut h);
                }
            }
            LogicalOp::Transform { input_column, output_type, .. } => {
                input_column.hash(&mut h);
                output_type.hash(&mut h);
                // Closure is opaque; use unique id
                id.hash(&mut h);
                for input in inputs {
                    input.content_hash.hash(&mut h);
                }
            }
            LogicalOp::BinaryTransform { left_column, right_column, output_type, .. } => {
                left_column.hash(&mut h);
                right_column.hash(&mut h);
                output_type.hash(&mut h);
                // Closure is opaque; use unique id
                id.hash(&mut h);
                for input in inputs {
                    input.content_hash.hash(&mut h);
                }
            }
            LogicalOp::GeneralizedTransform { output_types, .. } => {
                output_types.hash(&mut h);
                // Closure is opaque; use unique id
                id.hash(&mut h);
                for input in inputs {
                    input.content_hash.hash(&mut h);
                }
            }
            LogicalOp::Filter { column, .. } => {
                column.hash(&mut h);
                // Predicate is opaque; use unique id
                id.hash(&mut h);
                for input in inputs {
                    input.content_hash.hash(&mut h);
                }
            }
            LogicalOp::LogicalFilter => {
                for input in inputs {
                    input.content_hash.hash(&mut h);
                }
            }
            LogicalOp::Append => {
                for input in inputs {
                    input.content_hash.hash(&mut h);
                }
            }
            LogicalOp::Union => {
                for input in inputs {
                    input.content_hash.hash(&mut h);
                }
            }
            LogicalOp::Reduce { .. } => {
                // Aggregator is opaque; use unique id
                id.hash(&mut h);
                for input in inputs {
                    input.content_hash.hash(&mut h);
                }
            }
        }

        h.finish()
    }

    /// Create a source node that reads an SFrame from disk.
    pub fn sframe_source(
        path: &str,
        column_names: Vec<String>,
        column_types: Vec<FlexTypeEnum>,
        num_rows: u64,
    ) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::SFrameSource {
                path: path.to_string(),
                column_names,
                column_types,
                num_rows,
                begin_row: 0,
                end_row: num_rows,
                _keep_alive: None,
            },
            vec![],
        ))
    }

    /// Create a source node backed by a cache:// path with a keep-alive guard.
    pub fn sframe_source_cached(
        path: &str,
        column_names: Vec<String>,
        column_types: Vec<FlexTypeEnum>,
        num_rows: u64,
        keep_alive: Arc<dyn Send + Sync>,
    ) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::SFrameSource {
                path: path.to_string(),
                column_names,
                column_types,
                num_rows,
                begin_row: 0,
                end_row: num_rows,
                _keep_alive: Some(keep_alive),
            },
            vec![],
        ))
    }

    /// Create a materialized source from in-memory data.
    pub fn materialized(data: SFrameRows) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::MaterializedSource {
                data: Arc::new(data),
            },
            vec![],
        ))
    }

    /// Project specific columns from the input.
    pub fn project(input: Arc<PlannerNode>, column_indices: Vec<usize>) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::Project { column_indices },
            vec![input],
        ))
    }

    /// Filter rows from the input.
    pub fn filter(
        input: Arc<PlannerNode>,
        column: usize,
        predicate: Arc<dyn Fn(&FlexType) -> bool + Send + Sync>,
    ) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::Filter { column, predicate },
            vec![input],
        ))
    }

    /// Transform a column.
    pub fn transform(
        input: Arc<PlannerNode>,
        input_column: usize,
        func: Arc<dyn Fn(&FlexType) -> FlexType + Send + Sync>,
        output_type: FlexTypeEnum,
    ) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::Transform {
                input_column,
                func,
                output_type,
            },
            vec![input],
        ))
    }

    /// Binary transform on two columns.
    pub fn binary_transform(
        input: Arc<PlannerNode>,
        left_column: usize,
        right_column: usize,
        func: Arc<dyn Fn(&FlexType, &FlexType) -> FlexType + Send + Sync>,
        output_type: FlexTypeEnum,
    ) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::BinaryTransform {
                left_column,
                right_column,
                func,
                output_type,
            },
            vec![input],
        ))
    }

    /// Generalized transform producing multiple output columns.
    pub fn generalized_transform(
        input: Arc<PlannerNode>,
        func: Arc<dyn Fn(&[FlexType]) -> Vec<FlexType> + Send + Sync>,
        output_types: Vec<FlexTypeEnum>,
    ) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::GeneralizedTransform {
                func,
                output_types,
            },
            vec![input],
        ))
    }

    /// Append two inputs vertically.
    pub fn append(left: Arc<PlannerNode>, right: Arc<PlannerNode>) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::Append,
            vec![left, right],
        ))
    }

    /// Generate a range of integers.
    pub fn range(start: i64, step: i64, count: u64) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::Range { start, step, count },
            vec![],
        ))
    }

    /// Reduce the input using an aggregator.
    pub fn reduce(input: Arc<PlannerNode>, aggregator: Arc<dyn Aggregator>) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::Reduce { aggregator },
            vec![input],
        ))
    }

    /// Logical filter: emit rows from `data` where `mask` is truthy.
    ///
    /// `mask` must produce a single-column stream of the same row count as `data`.
    pub fn logical_filter(data: Arc<PlannerNode>, mask: Arc<PlannerNode>) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::LogicalFilter,
            vec![data, mask],
        ))
    }

    /// Union of multiple inputs.
    pub fn union(inputs: Vec<Arc<PlannerNode>>) -> Arc<Self> {
        Arc::new(PlannerNode::new(
            LogicalOp::Union,
            inputs,
        ))
    }

    /// Compute the output row count if statically known.
    ///
    /// Returns `Some(n)` for source nodes and linear operators whose
    /// input lengths are known. Returns `None` for sub-linear operators
    /// (Filter, LogicalFilter) whose output size depends on data.
    /// Reduce always returns `Some(1)`.
    pub fn length(&self) -> Option<u64> {
        match &self.op {
            LogicalOp::SFrameSource { begin_row, end_row, .. } => {
                Some(end_row - begin_row)
            }
            LogicalOp::MaterializedSource { data } => {
                Some(data.num_rows() as u64)
            }
            LogicalOp::Range { count, .. } => Some(*count),
            LogicalOp::Project { .. }
            | LogicalOp::Transform { .. }
            | LogicalOp::BinaryTransform { .. }
            | LogicalOp::GeneralizedTransform { .. } => {
                self.inputs[0].length()
            }
            LogicalOp::Append => {
                let a = self.inputs[0].length()?;
                let b = self.inputs[1].length()?;
                Some(a + b)
            }
            LogicalOp::Union => {
                let mut total: u64 = 0;
                for input in &self.inputs {
                    total += input.length()?;
                }
                Some(total)
            }
            LogicalOp::Filter { .. } | LogicalOp::LogicalFilter => None,
            LogicalOp::Reduce { .. } => Some(1),
        }
    }

    /// Return a Graphviz DOT representation of the query plan DAG.
    ///
    /// Shared subexpressions (same Arc) appear as a single node with
    /// multiple incoming edges, making reuse visible.
    pub fn explain(self: &Arc<Self>) -> String {
        use std::collections::HashSet;
        use std::fmt::Write;

        let mut buf = String::new();
        let mut visited = HashSet::new();
        writeln!(buf, "digraph plan {{").unwrap();
        writeln!(buf, "    rankdir=BT;").unwrap();
        Self::explain_dot(&mut buf, self, &mut visited);
        writeln!(buf, "}}").unwrap();
        buf
    }

    fn explain_dot(buf: &mut String, node: &Arc<Self>, visited: &mut std::collections::HashSet<usize>) {
        use std::fmt::Write;

        let id = Arc::as_ptr(node) as usize;
        if !visited.insert(id) {
            return; // already emitted
        }

        // Emit node
        let label = node.op_label();
        // Escape quotes for DOT
        let label = label.replace('\\', "\\\\").replace('"', "\\\"");
        writeln!(buf, "    n{} [label=\"{}\"];", id, label).unwrap();

        // Emit edges and recurse
        for input in &node.inputs {
            let child_id = Arc::as_ptr(input) as usize;
            writeln!(buf, "    n{} -> n{};", child_id, id).unwrap();
            Self::explain_dot(buf, input, visited);
        }
    }

    fn op_label(&self) -> String {
        match &self.op {
            LogicalOp::SFrameSource { path, column_names, column_types, num_rows, begin_row, end_row, .. } => {
                let cols: Vec<String> = column_names
                    .iter()
                    .zip(column_types.iter())
                    .map(|(n, t)| format!("{}:{:?}", n, t))
                    .collect();
                if *begin_row == 0 && *end_row == *num_rows {
                    format!("SFrameSource({}, {} rows, [{}])", path, num_rows, cols.join(", "))
                } else {
                    format!("SFrameSource({}, rows {}-{} of {}, [{}])", path, begin_row, end_row, num_rows, cols.join(", "))
                }
            }
            LogicalOp::Project { column_indices } => {
                format!("Project({:?})", column_indices)
            }
            LogicalOp::Filter { column, .. } => {
                format!("Filter(col={})", column)
            }
            LogicalOp::Transform { input_column, output_type, .. } => {
                format!("Transform(col={}, out={:?})", input_column, output_type)
            }
            LogicalOp::BinaryTransform { left_column, right_column, output_type, .. } => {
                format!("BinaryTransform(cols=[{}, {}], out={:?})", left_column, right_column, output_type)
            }
            LogicalOp::GeneralizedTransform { output_types, .. } => {
                format!("GeneralizedTransform(out={:?})", output_types)
            }
            LogicalOp::LogicalFilter => "LogicalFilter".to_string(),
            LogicalOp::Append => "Append".to_string(),
            LogicalOp::Range { start, step, count } => {
                format!("Range(start={}, step={}, count={})", start, step, count)
            }
            LogicalOp::Reduce { .. } => "Reduce".to_string(),
            LogicalOp::Union => "Union".to_string(),
            LogicalOp::MaterializedSource { data } => {
                format!("MaterializedSource({} rows, {} cols)", data.num_rows(), data.num_columns())
            }
        }
    }
}

/// Clone a plan DAG, rewriting all SFrameSource nodes to read
/// `[begin_row, end_row)` instead of their original range.
///
/// Preserves shared subexpressions: if two inputs in the original plan
/// are the same `Arc`, the cloned plan will share one cloned node.
pub fn clone_plan_with_row_range(
    plan: &Arc<PlannerNode>,
    begin_row: u64,
    end_row: u64,
) -> Arc<PlannerNode> {
    use std::collections::HashMap;

    fn walk(
        node: &Arc<PlannerNode>,
        begin_row: u64,
        end_row: u64,
        memo: &mut HashMap<usize, Arc<PlannerNode>>,
    ) -> Arc<PlannerNode> {
        let id = Arc::as_ptr(node) as usize;
        if let Some(existing) = memo.get(&id) {
            return existing.clone();
        }

        let new_inputs: Vec<Arc<PlannerNode>> = node
            .inputs
            .iter()
            .map(|input| walk(input, begin_row, end_row, memo))
            .collect();

        let new_op = match &node.op {
            LogicalOp::SFrameSource {
                path,
                column_names,
                column_types,
                num_rows,
                _keep_alive,
                ..
            } => LogicalOp::SFrameSource {
                path: path.clone(),
                column_names: column_names.clone(),
                column_types: column_types.clone(),
                num_rows: *num_rows,
                begin_row,
                end_row,
                _keep_alive: _keep_alive.clone(),
            },
            other => other.clone_op(),
        };

        let result = Arc::new(PlannerNode::new(new_op, new_inputs));
        memo.insert(id, result.clone());
        result
    }

    let mut memo = HashMap::new();
    walk(plan, begin_row, end_row, &mut memo)
}

/// Slice a plan to output only rows `[begin, end)`.
///
/// Works by recursively rewriting source nodes. The plan must be
/// fully linear (no Filter, LogicalFilter, or Reduce). Returns an
/// error if the plan contains sub-linear operators or if the range
/// is out of bounds.
pub fn slice_plan(
    plan: &Arc<PlannerNode>,
    begin: u64,
    end: u64,
) -> Result<Arc<PlannerNode>> {
    if begin > end {
        return Err(SFrameError::Format(format!(
            "Invalid slice range: begin ({}) > end ({})",
            begin, end
        )));
    }
    if begin == end {
        let dtypes = plan_output_types(plan);
        return Ok(PlannerNode::materialized(SFrameRows::empty(&dtypes)));
    }
    validate_sliceable(plan)?;
    let len = plan.length().ok_or_else(|| {
        SFrameError::Format("Cannot slice plan with unknown length".to_string())
    })?;
    if end > len {
        return Err(SFrameError::Format(format!(
            "Slice end ({}) exceeds plan length ({})",
            end, len
        )));
    }
    slice_recursive(plan, begin, end)
}

fn validate_sliceable(plan: &Arc<PlannerNode>) -> Result<()> {
    if plan.op.rate() == OperatorRate::SubLinear {
        return Err(SFrameError::Format(
            "Cannot slice a plan containing sub-linear operators (Filter, LogicalFilter, Reduce)".to_string()
        ));
    }
    for input in &plan.inputs {
        validate_sliceable(input)?;
    }
    Ok(())
}

fn slice_recursive(
    plan: &Arc<PlannerNode>,
    begin: u64,
    end: u64,
) -> Result<Arc<PlannerNode>> {
    match &plan.op {
        LogicalOp::SFrameSource {
            path, column_names, column_types, num_rows, begin_row, end_row, _keep_alive,
        } => {
            let new_begin = begin_row + begin;
            let new_end = begin_row + end;
            assert!(new_end <= *end_row);
            Ok(Arc::new(PlannerNode::new(
                LogicalOp::SFrameSource {
                    path: path.clone(),
                    column_names: column_names.clone(),
                    column_types: column_types.clone(),
                    num_rows: *num_rows,
                    begin_row: new_begin,
                    end_row: new_end,
                    _keep_alive: _keep_alive.clone(),
                },
                vec![],
            )))
        }
        LogicalOp::MaterializedSource { data } => {
            let indices: Vec<usize> = (begin as usize..end as usize).collect();
            let sliced = data.take(&indices)?;
            Ok(PlannerNode::materialized(sliced))
        }
        LogicalOp::Range { start, step, .. } => {
            Ok(PlannerNode::range(
                start + (begin as i64) * step,
                *step,
                end - begin,
            ))
        }
        LogicalOp::Project { .. }
        | LogicalOp::Transform { .. }
        | LogicalOp::BinaryTransform { .. }
        | LogicalOp::GeneralizedTransform { .. } => {
            let new_input = slice_recursive(&plan.inputs[0], begin, end)?;
            Ok(Arc::new(PlannerNode::new(
                plan.op.clone_op(),
                vec![new_input],
            )))
        }
        LogicalOp::Append => {
            let left_len = plan.inputs[0].length().ok_or_else(|| {
                SFrameError::Format("Append input has unknown length".to_string())
            })?;

            if end <= left_len {
                slice_recursive(&plan.inputs[0], begin, end)
            } else if begin >= left_len {
                slice_recursive(&plan.inputs[1], begin - left_len, end - left_len)
            } else {
                let left_sliced = slice_recursive(&plan.inputs[0], begin, left_len)?;
                let right_sliced = slice_recursive(&plan.inputs[1], 0, end - left_len)?;
                Ok(PlannerNode::append(left_sliced, right_sliced))
            }
        }
        LogicalOp::Union => {
            let mut input_lengths: Vec<u64> = Vec::new();
            for input in &plan.inputs {
                let l = input.length().ok_or_else(|| {
                    SFrameError::Format("Union input has unknown length".to_string())
                })?;
                input_lengths.push(l);
            }

            let mut offset: u64 = 0;
            let mut new_inputs: Vec<Arc<PlannerNode>> = Vec::new();

            for (i, &len) in input_lengths.iter().enumerate() {
                let input_begin = offset;
                let input_end = offset + len;

                if begin >= input_end || end <= input_begin {
                    offset = input_end;
                    continue;
                }

                let local_begin = if begin > input_begin { begin - input_begin } else { 0 };
                let local_end = if end < input_end { end - input_begin } else { len };

                new_inputs.push(slice_recursive(&plan.inputs[i], local_begin, local_end)?);
                offset = input_end;
            }

            match new_inputs.len() {
                0 => {
                    let dtypes = plan_output_types(plan);
                    Ok(PlannerNode::materialized(SFrameRows::empty(&dtypes)))
                }
                1 => Ok(new_inputs.into_iter().next().unwrap()),
                _ => Ok(PlannerNode::union(new_inputs)),
            }
        }
        _ => Err(SFrameError::Format(
            "Unexpected operator in slice_recursive".to_string()
        )),
    }
}

fn plan_output_types(plan: &Arc<PlannerNode>) -> Vec<FlexTypeEnum> {
    match &plan.op {
        LogicalOp::SFrameSource { column_types, .. } => column_types.clone(),
        LogicalOp::MaterializedSource { data } => data.dtypes(),
        LogicalOp::Range { .. } => vec![FlexTypeEnum::Integer],
        LogicalOp::Project { column_indices } => {
            let input_types = plan_output_types(&plan.inputs[0]);
            column_indices.iter().map(|&i| input_types[i]).collect()
        }
        LogicalOp::Transform { output_type, .. } => {
            let mut types = plan_output_types(&plan.inputs[0]);
            types.push(*output_type);
            types
        }
        LogicalOp::BinaryTransform { output_type, .. } => {
            let mut types = plan_output_types(&plan.inputs[0]);
            types.push(*output_type);
            types
        }
        LogicalOp::GeneralizedTransform { output_types, .. } => output_types.clone(),
        LogicalOp::Append | LogicalOp::Union => plan_output_types(&plan.inputs[0]),
        LogicalOp::Filter { .. } | LogicalOp::LogicalFilter => {
            plan_output_types(&plan.inputs[0])
        }
        LogicalOp::Reduce { .. } => plan_output_types(&plan.inputs[0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{ColumnData, SFrameRows};

    // Length tests
    #[test]
    fn test_length_sframe_source() {
        let node = PlannerNode::sframe_source("test.sf", vec!["col".into()], vec![FlexTypeEnum::Integer], 100);
        assert_eq!(node.length(), Some(100));
    }

    #[test]
    fn test_length_range() {
        assert_eq!(PlannerNode::range(0, 1, 50).length(), Some(50));
    }

    #[test]
    fn test_length_materialized() {
        let col = ColumnData::Integer(vec![Some(1), Some(2), Some(3)]);
        let batch = SFrameRows::new(vec![col]).unwrap();
        assert_eq!(PlannerNode::materialized(batch).length(), Some(3));
    }

    #[test]
    fn test_length_project() {
        let src = PlannerNode::sframe_source("t.sf", vec!["a".into(), "b".into()], vec![FlexTypeEnum::Integer, FlexTypeEnum::Float], 100);
        assert_eq!(PlannerNode::project(src, vec![0]).length(), Some(100));
    }

    #[test]
    fn test_length_transform() {
        let src = PlannerNode::range(0, 1, 42);
        let xform = PlannerNode::transform(src, 0, Arc::new(|v: &FlexType| v.clone()), FlexTypeEnum::Integer);
        assert_eq!(xform.length(), Some(42));
    }

    #[test]
    fn test_length_append() {
        let a = PlannerNode::range(0, 1, 30);
        let b = PlannerNode::range(0, 1, 20);
        assert_eq!(PlannerNode::append(a, b).length(), Some(50));
    }

    #[test]
    fn test_length_union() {
        let a = PlannerNode::range(0, 1, 10);
        let b = PlannerNode::range(0, 1, 20);
        let c = PlannerNode::range(0, 1, 30);
        assert_eq!(PlannerNode::union(vec![a, b, c]).length(), Some(60));
    }

    #[test]
    fn test_length_filter_unknown() {
        let src = PlannerNode::range(0, 1, 100);
        let filtered = PlannerNode::filter(src, 0, Arc::new(|_: &FlexType| true));
        assert_eq!(filtered.length(), None);
    }

    #[test]
    fn test_length_logical_filter_unknown() {
        let data = PlannerNode::range(0, 1, 100);
        let mask = PlannerNode::range(0, 1, 100);
        assert_eq!(PlannerNode::logical_filter(data, mask).length(), None);
    }

    #[test]
    fn test_length_append_one_unknown() {
        let a = PlannerNode::range(0, 1, 30);
        let b = PlannerNode::filter(PlannerNode::range(0, 1, 100), 0, Arc::new(|_: &FlexType| true));
        assert_eq!(PlannerNode::append(a, b).length(), None);
    }

    // Slice tests
    #[test]
    fn test_slice_sframe_source() {
        let src = PlannerNode::sframe_source("test.sf", vec!["col".into()], vec![FlexTypeEnum::Integer], 100);
        let sliced = slice_plan(&src, 10, 30).unwrap();
        assert_eq!(sliced.length(), Some(20));
        match &sliced.op {
            LogicalOp::SFrameSource { begin_row, end_row, .. } => {
                assert_eq!(*begin_row, 10);
                assert_eq!(*end_row, 30);
            }
            _ => panic!("expected SFrameSource"),
        }
    }

    #[test]
    fn test_slice_range() {
        let src = PlannerNode::range(10, 3, 100);
        let sliced = slice_plan(&src, 5, 15).unwrap();
        assert_eq!(sliced.length(), Some(10));
        match &sliced.op {
            LogicalOp::Range { start, step, count } => {
                assert_eq!(*start, 25); // 10 + 5*3
                assert_eq!(*step, 3);
                assert_eq!(*count, 10);
            }
            _ => panic!("expected Range"),
        }
    }

    #[test]
    fn test_slice_materialized() {
        let col = ColumnData::Integer(vec![Some(10), Some(20), Some(30), Some(40), Some(50)]);
        let batch = SFrameRows::new(vec![col]).unwrap();
        let src = PlannerNode::materialized(batch);
        let sliced = slice_plan(&src, 1, 4).unwrap();
        assert_eq!(sliced.length(), Some(3));
    }

    #[test]
    fn test_slice_through_linear_ops() {
        let src = PlannerNode::sframe_source("t.sf", vec!["a".into(), "b".into()], vec![FlexTypeEnum::Integer, FlexTypeEnum::Float], 100);
        let proj = PlannerNode::project(src, vec![0]);
        let xform = PlannerNode::transform(proj, 0, Arc::new(|v: &FlexType| v.clone()), FlexTypeEnum::Integer);
        let sliced = slice_plan(&xform, 20, 40).unwrap();
        assert_eq!(sliced.length(), Some(20));
    }

    #[test]
    fn test_slice_append() {
        let a = PlannerNode::sframe_source("a.sf", vec!["col".into()], vec![FlexTypeEnum::Integer], 60);
        let b = PlannerNode::sframe_source("b.sf", vec!["col".into()], vec![FlexTypeEnum::Integer], 40);
        let appended = PlannerNode::append(a, b);

        // Entirely within first input
        assert_eq!(slice_plan(&appended, 10, 30).unwrap().length(), Some(20));
        // Entirely within second input
        assert_eq!(slice_plan(&appended, 70, 90).unwrap().length(), Some(20));
        // Spanning both
        assert_eq!(slice_plan(&appended, 50, 80).unwrap().length(), Some(30));
    }

    #[test]
    fn test_slice_rejects_sublinear() {
        let src = PlannerNode::range(0, 1, 100);
        let filtered = PlannerNode::filter(src, 0, Arc::new(|_: &FlexType| true));
        assert!(slice_plan(&filtered, 0, 10).is_err());
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let src = PlannerNode::range(0, 1, 100);
        assert!(slice_plan(&src, 0, 101).is_err());
        assert!(slice_plan(&src, 50, 30).is_err());
    }

    #[test]
    fn test_slice_already_sliced_source() {
        let src = PlannerNode::sframe_source("test.sf", vec!["col".into()], vec![FlexTypeEnum::Integer], 100);
        let s1 = slice_plan(&src, 20, 80).unwrap();
        let s2 = slice_plan(&s1, 5, 15).unwrap();
        assert_eq!(s2.length(), Some(10));
        match &s2.op {
            LogicalOp::SFrameSource { begin_row, end_row, .. } => {
                assert_eq!(*begin_row, 25);
                assert_eq!(*end_row, 35);
            }
            _ => panic!("expected SFrameSource"),
        }
    }

    #[test]
    fn test_slice_empty() {
        let src = PlannerNode::range(0, 1, 100);
        let sliced = slice_plan(&src, 5, 5).unwrap();
        assert_eq!(sliced.length(), Some(0));
    }

    // Content hash tests

    #[test]
    fn test_same_source_same_hash() {
        let a = PlannerNode::sframe_source(
            "test.sf",
            vec!["col".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );
        let b = PlannerNode::sframe_source(
            "test.sf",
            vec!["col".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );
        assert_eq!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn test_different_source_different_hash() {
        let a = PlannerNode::sframe_source(
            "a.sf",
            vec!["col".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );
        let b = PlannerNode::sframe_source(
            "b.sf",
            vec!["col".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );
        assert_ne!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn test_transform_different_id() {
        let src = PlannerNode::sframe_source(
            "test.sf",
            vec!["col".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );
        let t1 = PlannerNode::transform(
            src.clone(),
            0,
            Arc::new(|v: &FlexType| v.clone()),
            FlexTypeEnum::Integer,
        );
        let t2 = PlannerNode::transform(
            src,
            0,
            Arc::new(|v: &FlexType| v.clone()),
            FlexTypeEnum::Integer,
        );
        // Different closures (different node ids) should produce different hashes
        assert_ne!(t1.content_hash(), t2.content_hash());
    }

    #[test]
    fn test_transform_on_same_input() {
        let src_a = PlannerNode::sframe_source(
            "a.sf",
            vec!["col".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );
        let src_b = PlannerNode::sframe_source(
            "b.sf",
            vec!["col".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );
        // Same closure (same node), but different inputs
        let func: Arc<dyn Fn(&FlexType) -> FlexType + Send + Sync> =
            Arc::new(|v: &FlexType| v.clone());
        let t1 = PlannerNode::transform(src_a, 0, func.clone(), FlexTypeEnum::Integer);
        let t2 = PlannerNode::transform(src_b, 0, func, FlexTypeEnum::Integer);
        // Different input hashes should produce different output hashes
        assert_ne!(t1.content_hash(), t2.content_hash());
    }

    #[test]
    fn test_sliced_sources_same_hash() {
        let a = slice_plan(
            &PlannerNode::sframe_source(
                "test.sf",
                vec!["col".into()],
                vec![FlexTypeEnum::Integer],
                100,
            ),
            10,
            50,
        ).unwrap();
        let b = slice_plan(
            &PlannerNode::sframe_source(
                "test.sf",
                vec!["col".into()],
                vec![FlexTypeEnum::Integer],
                100,
            ),
            10,
            50,
        ).unwrap();
        assert_eq!(a.content_hash(), b.content_hash());
    }
}
