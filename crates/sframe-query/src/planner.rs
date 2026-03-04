//! Logical query planner.
//!
//! DAG of PlannerNodes representing query operations. Nodes are Arc-shared
//! so the same subexpression can appear in multiple places.

use std::io::{Read, Write};
use std::sync::Arc;

use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::SFrameRows;

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
    /// Create a source node that reads an SFrame from disk.
    pub fn sframe_source(
        path: &str,
        column_names: Vec<String>,
        column_types: Vec<FlexTypeEnum>,
        num_rows: u64,
    ) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::SFrameSource {
                path: path.to_string(),
                column_names,
                column_types,
                num_rows,
                begin_row: 0,
                end_row: num_rows,
                _keep_alive: None,
            },
            inputs: vec![],
        })
    }

    /// Create a source node backed by a cache:// path with a keep-alive guard.
    pub fn sframe_source_cached(
        path: &str,
        column_names: Vec<String>,
        column_types: Vec<FlexTypeEnum>,
        num_rows: u64,
        keep_alive: Arc<dyn Send + Sync>,
    ) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::SFrameSource {
                path: path.to_string(),
                column_names,
                column_types,
                num_rows,
                begin_row: 0,
                end_row: num_rows,
                _keep_alive: Some(keep_alive),
            },
            inputs: vec![],
        })
    }

    /// Create a materialized source from in-memory data.
    pub fn materialized(data: SFrameRows) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::MaterializedSource {
                data: Arc::new(data),
            },
            inputs: vec![],
        })
    }

    /// Project specific columns from the input.
    pub fn project(input: Arc<PlannerNode>, column_indices: Vec<usize>) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::Project { column_indices },
            inputs: vec![input],
        })
    }

    /// Filter rows from the input.
    pub fn filter(
        input: Arc<PlannerNode>,
        column: usize,
        predicate: Arc<dyn Fn(&FlexType) -> bool + Send + Sync>,
    ) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::Filter { column, predicate },
            inputs: vec![input],
        })
    }

    /// Transform a column.
    pub fn transform(
        input: Arc<PlannerNode>,
        input_column: usize,
        func: Arc<dyn Fn(&FlexType) -> FlexType + Send + Sync>,
        output_type: FlexTypeEnum,
    ) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::Transform {
                input_column,
                func,
                output_type,
            },
            inputs: vec![input],
        })
    }

    /// Binary transform on two columns.
    pub fn binary_transform(
        input: Arc<PlannerNode>,
        left_column: usize,
        right_column: usize,
        func: Arc<dyn Fn(&FlexType, &FlexType) -> FlexType + Send + Sync>,
        output_type: FlexTypeEnum,
    ) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::BinaryTransform {
                left_column,
                right_column,
                func,
                output_type,
            },
            inputs: vec![input],
        })
    }

    /// Generalized transform producing multiple output columns.
    pub fn generalized_transform(
        input: Arc<PlannerNode>,
        func: Arc<dyn Fn(&[FlexType]) -> Vec<FlexType> + Send + Sync>,
        output_types: Vec<FlexTypeEnum>,
    ) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::GeneralizedTransform {
                func,
                output_types,
            },
            inputs: vec![input],
        })
    }

    /// Append two inputs vertically.
    pub fn append(left: Arc<PlannerNode>, right: Arc<PlannerNode>) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::Append,
            inputs: vec![left, right],
        })
    }

    /// Generate a range of integers.
    pub fn range(start: i64, step: i64, count: u64) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::Range { start, step, count },
            inputs: vec![],
        })
    }

    /// Reduce the input using an aggregator.
    pub fn reduce(input: Arc<PlannerNode>, aggregator: Arc<dyn Aggregator>) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::Reduce { aggregator },
            inputs: vec![input],
        })
    }

    /// Logical filter: emit rows from `data` where `mask` is truthy.
    ///
    /// `mask` must produce a single-column stream of the same row count as `data`.
    pub fn logical_filter(data: Arc<PlannerNode>, mask: Arc<PlannerNode>) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::LogicalFilter,
            inputs: vec![data, mask],
        })
    }

    /// Union of multiple inputs.
    pub fn union(inputs: Vec<Arc<PlannerNode>>) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::Union,
            inputs,
        })
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

        let result = Arc::new(PlannerNode {
            op: new_op,
            inputs: new_inputs,
        });
        memo.insert(id, result.clone());
        result
    }

    let mut memo = HashMap::new();
    walk(plan, begin_row, end_row, &mut memo)
}
