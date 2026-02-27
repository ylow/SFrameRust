//! Logical query planner.
//!
//! DAG of PlannerNodes representing query operations. Nodes are Arc-shared
//! so the same subexpression can appear in multiple places.

use std::io::{Read, Write};
use std::sync::Arc;

use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::SFrameRows;

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

    /// An in-memory batch of rows (materialized data).
    MaterializedSource {
        data: Arc<SFrameRows>,
    },
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

    /// Union of multiple inputs.
    pub fn union(inputs: Vec<Arc<PlannerNode>>) -> Arc<Self> {
        Arc::new(PlannerNode {
            op: LogicalOp::Union,
            inputs,
        })
    }
}
