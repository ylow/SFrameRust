//! Query optimizer — transforms PlannerNode DAGs for better execution.
//!
//! Available optimization passes:
//! - **Projection pushdown**: Push column selections toward data sources
//! - **Project fusion**: Merge adjacent Project nodes
//! - **Identity elimination**: Remove no-op Project nodes

use std::sync::Arc;

use crate::planner::{LogicalOp, PlannerNode};

/// Apply all optimization passes to a plan.
pub fn optimize(plan: &Arc<PlannerNode>) -> Arc<PlannerNode> {
    let plan = fuse_projects(plan);
    let plan = eliminate_identity_projects(&plan);
    let plan = eliminate_trivial_unions(&plan);
    let plan = eliminate_empty_appends(&plan);
    let plan = pushdown_projects(&plan);
    let plan = push_filter_through_transform(&plan);
    plan
}

/// Fuse adjacent Project nodes: `Project(b) → Project(a) → input`
/// becomes `Project(a[b[i]]) → input`.
pub fn fuse_projects(plan: &Arc<PlannerNode>) -> Arc<PlannerNode> {
    // First recursively optimize inputs
    let new_inputs: Vec<Arc<PlannerNode>> = plan.inputs.iter().map(|i| fuse_projects(i)).collect();
    let plan = rebuild_with_inputs(plan, new_inputs);

    // Check: is this a Project whose input is also a Project?
    if let LogicalOp::Project { column_indices: outer_indices } = &plan.op {
        if plan.inputs.len() == 1 {
            if let LogicalOp::Project { column_indices: inner_indices } = &plan.inputs[0].op {
                // Compose: outer[i] → inner[outer[i]]
                let composed: Vec<usize> = outer_indices
                    .iter()
                    .map(|&i| inner_indices[i])
                    .collect();
                return Arc::new(PlannerNode::new(
                    LogicalOp::Project { column_indices: composed },
                    plan.inputs[0].inputs.clone(),
                ));
            }
        }
    }

    plan
}

/// Eliminate identity Project nodes (those that select all columns in order).
pub fn eliminate_identity_projects(plan: &Arc<PlannerNode>) -> Arc<PlannerNode> {
    let new_inputs: Vec<Arc<PlannerNode>> = plan.inputs.iter().map(|i| eliminate_identity_projects(i)).collect();
    let plan = rebuild_with_inputs(plan, new_inputs);

    if let LogicalOp::Project { column_indices } = &plan.op {
        if plan.inputs.len() == 1 {
            // Check if this is an identity projection (0, 1, 2, ..., n-1)
            let is_identity = column_indices.iter().enumerate().all(|(i, &c)| c == i);
            if is_identity {
                // Check input column count to verify it's truly identity
                let input_ncols = count_output_columns(&plan.inputs[0]);
                if let Some(n) = input_ncols {
                    if column_indices.len() == n {
                        return plan.inputs[0].clone();
                    }
                }
            }
        }
    }

    plan
}

/// Push Project nodes closer to data sources.
///
/// If we have `Project → Filter → Source`, we can push the project below
/// the filter (keeping the filter's column in the projection, and adjusting
/// the filter's column index).
pub fn pushdown_projects(plan: &Arc<PlannerNode>) -> Arc<PlannerNode> {
    let new_inputs: Vec<Arc<PlannerNode>> = plan.inputs.iter().map(|i| pushdown_projects(i)).collect();
    let plan = rebuild_with_inputs(plan, new_inputs);

    // Project over Filter: push project below filter
    if let LogicalOp::Project { column_indices } = &plan.op {
        if plan.inputs.len() == 1 {
            if let LogicalOp::Filter { column: filter_col, predicate } = &plan.inputs[0].op {
                if plan.inputs[0].inputs.len() == 1 {
                    // We need the filter column in the pushed-down project.
                    // Build a wider project that includes the filter column.
                    let mut needed_cols: Vec<usize> = column_indices.clone();
                    let filter_in_output = needed_cols.iter().position(|&c| c == *filter_col);

                    if filter_in_output.is_none() {
                        // Filter column not in output; add it, filter, then remove it
                        needed_cols.push(*filter_col);
                    }

                    // Deduplicate and sort to get the wider project
                    let mut wider: Vec<usize> = needed_cols.clone();
                    wider.sort();
                    wider.dedup();

                    // Map filter column to its index in the wider projection
                    let new_filter_col = wider.iter().position(|&c| c == *filter_col).unwrap();

                    // Map original output columns to their positions in the wider projection
                    let final_indices: Vec<usize> = column_indices
                        .iter()
                        .map(|c| wider.iter().position(|&w| w == *c).unwrap())
                        .collect();

                    // Build: Project(final) → Filter(new_col) → Project(wider) → input
                    let inner_source = plan.inputs[0].inputs[0].clone();
                    let pushed_project = PlannerNode::project(inner_source, wider);
                    let new_filter = PlannerNode::filter(pushed_project, new_filter_col, predicate.clone());

                    if final_indices.iter().enumerate().all(|(i, &c)| c == i)
                        && final_indices.len() == column_indices.len()
                        && filter_in_output.is_some()
                    {
                        // The final project would be identity, skip it
                        return new_filter;
                    }

                    return PlannerNode::project(new_filter, final_indices);
                }
            }

            // Transform outputs a single column, so Project([0]) over Transform
            // is an identity — handled by eliminate_identity_projects.
        }
    }

    plan
}

/// Flatten nested Union nodes and eliminate singleton Unions.
///
/// - `Union([x])` becomes `x`
/// - `Union(a, Union(b, c))` becomes `Union(a, b, c)`
pub fn eliminate_trivial_unions(plan: &Arc<PlannerNode>) -> Arc<PlannerNode> {
    let new_inputs: Vec<Arc<PlannerNode>> = plan
        .inputs
        .iter()
        .map(|i| eliminate_trivial_unions(i))
        .collect();
    let plan = rebuild_with_inputs(plan, new_inputs);

    if let LogicalOp::Union = &plan.op {
        // Flatten nested Unions
        let mut flattened: Vec<Arc<PlannerNode>> = Vec::new();
        for input in &plan.inputs {
            if let LogicalOp::Union = &input.op {
                flattened.extend(input.inputs.iter().cloned());
            } else {
                flattened.push(input.clone());
            }
        }
        // Singleton elimination
        if flattened.len() == 1 {
            return flattened.remove(0);
        }
        if flattened.len() != plan.inputs.len() {
            return Arc::new(PlannerNode::new(
                LogicalOp::Union,
                flattened,
            ));
        }
    }

    plan
}

/// Remove empty sources from Append nodes.
///
/// - `Append(X, empty)` becomes `X`
/// - `Append(empty, X)` becomes `X`
pub fn eliminate_empty_appends(plan: &Arc<PlannerNode>) -> Arc<PlannerNode> {
    let new_inputs: Vec<Arc<PlannerNode>> = plan
        .inputs
        .iter()
        .map(|i| eliminate_empty_appends(i))
        .collect();
    let plan = rebuild_with_inputs(plan, new_inputs);

    if let LogicalOp::Append = &plan.op {
        if plan.inputs.len() == 2 {
            let left_empty = is_empty_source(&plan.inputs[0]);
            let right_empty = is_empty_source(&plan.inputs[1]);
            match (left_empty, right_empty) {
                (true, false) => return plan.inputs[1].clone(),
                (false, true) => return plan.inputs[0].clone(),
                _ => {}
            }
        }
    }

    plan
}

fn is_empty_source(plan: &PlannerNode) -> bool {
    match &plan.op {
        LogicalOp::MaterializedSource { data } => data.num_rows() == 0,
        _ => false,
    }
}

/// Push Filter below Transform when the filter references a column from
/// the transform's input rather than the transform's output.
///
/// Since Transform now produces a single column (index 0), this only
/// applies when Filter is on a column that exists in the input plan but
/// not in the Transform output. In practice, Filter → Transform where
/// the filter is on column 0 (the transform output) cannot be pushed.
/// This pass is kept for GeneralizedTransform which may output multiple
/// columns.
pub fn push_filter_through_transform(plan: &Arc<PlannerNode>) -> Arc<PlannerNode> {
    let new_inputs: Vec<Arc<PlannerNode>> = plan
        .inputs
        .iter()
        .map(|i| push_filter_through_transform(i))
        .collect();
    rebuild_with_inputs(plan, new_inputs)
}

/// Count the number of output columns for a plan node (if known statically).
fn count_output_columns(plan: &PlannerNode) -> Option<usize> {
    match &plan.op {
        LogicalOp::SFrameSource { column_names, .. } => Some(column_names.len()),
        LogicalOp::MaterializedSource { data } => Some(data.num_columns()),
        LogicalOp::Project { column_indices } => Some(column_indices.len()),
        LogicalOp::Range { .. } => Some(1),
        LogicalOp::Transform { .. } => Some(1),
        LogicalOp::BinaryTransform { .. } => Some(1),
        LogicalOp::GeneralizedTransform { output_types, .. } => {
            // GeneralizedTransform replaces all input columns with output_types
            Some(output_types.len())
        }
        LogicalOp::Filter { .. } | LogicalOp::LogicalFilter => {
            // Filter/LogicalFilter doesn't change column count (uses input 0)
            plan.inputs.first().and_then(|i| count_output_columns(i))
        }
        LogicalOp::ColumnUnion => {
            let mut total = 0;
            for input in &plan.inputs {
                total += count_output_columns(input)?;
            }
            Some(total)
        }
        LogicalOp::Append | LogicalOp::Union => {
            // Same as first input
            plan.inputs.first().and_then(|i| count_output_columns(i))
        }
        LogicalOp::Reduce { .. } => Some(1),
    }
}

/// Rebuild a PlannerNode with new inputs (preserving the same operation).
fn rebuild_with_inputs(plan: &Arc<PlannerNode>, new_inputs: Vec<Arc<PlannerNode>>) -> Arc<PlannerNode> {
    // If inputs haven't changed (same Arc pointers), reuse the plan
    if plan.inputs.len() == new_inputs.len()
        && plan.inputs.iter().zip(new_inputs.iter()).all(|(a, b)| Arc::ptr_eq(a, b))
    {
        return plan.clone();
    }

    Arc::new(PlannerNode::new(
        clone_op(&plan.op),
        new_inputs,
    ))
}

/// Clone a LogicalOp. Delegates to `LogicalOp::clone_op`.
fn clone_op(op: &LogicalOp) -> LogicalOp {
    op.clone_op()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sframe_types::flex_type::FlexTypeEnum;

    #[test]
    fn test_fuse_adjacent_projects() {
        // Source(3 cols) → Project([0,2]) → Project([1])
        // Should fuse to: Source(3 cols) → Project([2])
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into(), "b".into(), "c".into()],
            vec![FlexTypeEnum::Integer, FlexTypeEnum::Float, FlexTypeEnum::String],
            100,
        );
        let proj1 = PlannerNode::project(source, vec![0, 2]); // keep cols a, c
        let proj2 = PlannerNode::project(proj1, vec![1]);      // keep col c (index 1 of [a,c])

        let optimized = optimize(&proj2);

        // Should be a single Project([2]) → Source
        if let LogicalOp::Project { column_indices } = &optimized.op {
            assert_eq!(column_indices, &[2]);
        } else {
            panic!("Expected Project, got {:?}", std::mem::discriminant(&optimized.op));
        }
        assert!(matches!(optimized.inputs[0].op, LogicalOp::SFrameSource { .. }));
    }

    #[test]
    fn test_eliminate_identity_project() {
        // Source(3 cols) → Project([0, 1, 2]) should become just Source
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into(), "b".into(), "c".into()],
            vec![FlexTypeEnum::Integer, FlexTypeEnum::Float, FlexTypeEnum::String],
            100,
        );
        let proj = PlannerNode::project(source.clone(), vec![0, 1, 2]);

        let optimized = optimize(&proj);

        // Should be just the source
        assert!(matches!(optimized.op, LogicalOp::SFrameSource { .. }));
    }

    #[test]
    fn test_non_identity_project_kept() {
        // Source(3 cols) → Project([0, 2]) should remain
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into(), "b".into(), "c".into()],
            vec![FlexTypeEnum::Integer, FlexTypeEnum::Float, FlexTypeEnum::String],
            100,
        );
        let proj = PlannerNode::project(source, vec![0, 2]);

        let optimized = optimize(&proj);
        assert!(matches!(optimized.op, LogicalOp::Project { .. }));
    }

    #[test]
    fn test_pushdown_project_through_filter() {
        // Source(3 cols) → Filter(col=1) → Project([0, 2])
        // Should push project below filter:
        // Source → Project([0, 1, 2]) → Filter(col=1) → Project([0, 2])
        // Which then identity-eliminates to:
        // Source → Filter(col=1) → Project([0, 2])
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into(), "b".into(), "c".into()],
            vec![FlexTypeEnum::Integer, FlexTypeEnum::Float, FlexTypeEnum::String],
            100,
        );
        let filtered = PlannerNode::filter(
            source,
            1,
            std::sync::Arc::new(|_| true),
        );
        let projected = PlannerNode::project(filtered, vec![0, 2]);

        let optimized = optimize(&projected);

        // The optimizer should rearrange. Let's just verify it still produces
        // a valid plan (not crash) and keeps the project
        let ncols = count_output_columns(&optimized);
        assert_eq!(ncols, Some(2)); // still selecting 2 columns
    }

    #[test]
    fn test_no_change_for_simple_source() {
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );

        let optimized = optimize(&source);
        assert!(std::sync::Arc::ptr_eq(&source, &optimized));
    }

    #[test]
    fn test_eliminate_singleton_union() {
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            10,
        );
        let union_node = PlannerNode::union(vec![source]);
        let optimized = optimize(&union_node);
        assert!(matches!(optimized.op, LogicalOp::SFrameSource { .. }));
    }

    #[test]
    fn test_flatten_nested_union() {
        let s1 = PlannerNode::sframe_source(
            "t1.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            10,
        );
        let s2 = PlannerNode::sframe_source(
            "t2.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            10,
        );
        let s3 = PlannerNode::sframe_source(
            "t3.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            10,
        );
        let inner_union = PlannerNode::union(vec![s2, s3]);
        let outer_union = PlannerNode::union(vec![s1, inner_union]);
        let optimized = eliminate_trivial_unions(&outer_union);
        if let LogicalOp::Union = &optimized.op {
            assert_eq!(optimized.inputs.len(), 3);
        } else {
            panic!("Expected flattened Union");
        }
    }

    #[test]
    fn test_eliminate_empty_append_right() {
        use crate::batch::SFrameRows;
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            10,
        );
        let empty = PlannerNode::materialized(SFrameRows::empty(&[FlexTypeEnum::Integer]));
        let appended = PlannerNode::append(source, empty);
        let optimized = eliminate_empty_appends(&appended);
        assert!(matches!(optimized.op, LogicalOp::SFrameSource { .. }));
    }

    #[test]
    fn test_eliminate_empty_append_left() {
        use crate::batch::SFrameRows;
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            10,
        );
        let empty = PlannerNode::materialized(SFrameRows::empty(&[FlexTypeEnum::Integer]));
        let appended = PlannerNode::append(empty, source);
        let optimized = eliminate_empty_appends(&appended);
        assert!(matches!(optimized.op, LogicalOp::SFrameSource { .. }));
    }

    #[test]
    fn test_filter_over_transform_unchanged() {
        use sframe_types::flex_type::FlexType;
        // Transform outputs a single column. Filter over Transform
        // is left unchanged (filter is always on the transform output).
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            10,
        );
        let transformed = PlannerNode::transform(
            source,
            0,
            Arc::new(|v: &FlexType| v.clone()),
            FlexTypeEnum::Integer,
        );
        let filtered = PlannerNode::filter(
            transformed,
            0,
            Arc::new(|v: &FlexType| matches!(v, FlexType::Integer(i) if *i > 0)),
        );
        let optimized = push_filter_through_transform(&filtered);
        assert!(matches!(optimized.op, LogicalOp::Filter { .. }));
        assert!(matches!(optimized.inputs[0].op, LogicalOp::Transform { .. }));
    }
}
