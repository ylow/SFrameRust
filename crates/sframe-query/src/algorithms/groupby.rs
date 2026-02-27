//! Hash-based groupby algorithm.
//!
//! Groups rows by key columns and applies aggregators to produce
//! grouped/aggregated results.

use std::collections::HashMap;

use futures::StreamExt;

use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::algorithms::aggregators::AggSpec;
use crate::batch::{ColumnData, SFrameRows};
use crate::execute::BatchStream;
use crate::planner::Aggregator;

/// Perform a groupby operation on a batch stream.
///
/// Groups by `key_columns` and applies `agg_specs` to produce the aggregated result.
/// Returns a single SFrameRows batch with one row per group.
pub async fn groupby(
    mut input: BatchStream,
    key_columns: &[usize],
    agg_specs: &[AggSpec],
) -> Result<SFrameRows> {
    // Map from group key → per-agg-spec aggregators
    let mut groups: HashMap<Vec<FlexTypeKey>, Vec<Box<dyn Aggregator>>> = HashMap::new();

    while let Some(batch_result) = input.next().await {
        let batch = batch_result?;
        for row_idx in 0..batch.num_rows() {
            // Extract group key
            let key: Vec<FlexTypeKey> = key_columns
                .iter()
                .map(|&col| FlexTypeKey(batch.column(col).get(row_idx)))
                .collect();

            // Get or create aggregators for this group
            let aggs = groups.entry(key).or_insert_with(|| {
                agg_specs
                    .iter()
                    .map(|spec| spec.aggregator.box_clone())
                    .collect()
            });

            // Feed each aggregator its column value
            for (i, spec) in agg_specs.iter().enumerate() {
                let val = batch.column(spec.column).get(row_idx);
                aggs[i].add(&[val]);
            }
        }
    }

    // Output schema: [key_columns...] [agg_output_columns...]
    let num_key_cols = key_columns.len();
    let num_agg_specs = agg_specs.len();

    if groups.is_empty() {
        let mut dtypes = Vec::new();
        for _ in 0..num_key_cols {
            dtypes.push(FlexTypeEnum::Integer); // placeholder
        }
        for _ in agg_specs {
            dtypes.push(FlexTypeEnum::Integer);
        }
        return Ok(SFrameRows::empty(&dtypes));
    }

    // Finalize all groups first, collecting keys and agg results as FlexType rows
    let mut key_values: Vec<Vec<FlexType>> = Vec::with_capacity(groups.len());
    let mut agg_values: Vec<Vec<FlexType>> = Vec::with_capacity(groups.len());

    for (key, mut aggs) in groups {
        key_values.push(key.into_iter().map(|k| k.0).collect());
        let mut row_aggs = Vec::with_capacity(num_agg_specs);
        for agg in aggs.iter_mut() {
            row_aggs.push(agg.finalize());
        }
        agg_values.push(row_aggs);
    }

    // Determine output types from actual values
    let key_types: Vec<FlexTypeEnum> = (0..num_key_cols)
        .map(|i| {
            key_values
                .iter()
                .map(|row| row[i].type_enum())
                .find(|&t| t != FlexTypeEnum::Undefined)
                .unwrap_or(FlexTypeEnum::Integer)
        })
        .collect();

    let agg_types: Vec<FlexTypeEnum> = (0..num_agg_specs)
        .map(|i| {
            agg_values
                .iter()
                .map(|row| row[i].type_enum())
                .find(|&t| t != FlexTypeEnum::Undefined)
                .unwrap_or(FlexTypeEnum::Integer)
        })
        .collect();

    // Build columns
    let mut columns: Vec<ColumnData> = key_types
        .iter()
        .chain(agg_types.iter())
        .map(|&dt| ColumnData::empty(dt))
        .collect();

    for (group_idx, keys) in key_values.iter().enumerate() {
        for (i, val) in keys.iter().enumerate() {
            columns[i].push(val)?;
        }
        for (i, val) in agg_values[group_idx].iter().enumerate() {
            columns[num_key_cols + i].push(val)?;
        }
    }

    SFrameRows::new(columns)
}

/// Wrapper around FlexType to implement Hash + Eq for use as HashMap keys.
#[derive(Clone, Debug)]
struct FlexTypeKey(FlexType);

impl PartialEq for FlexTypeKey {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (FlexType::Integer(a), FlexType::Integer(b)) => a == b,
            (FlexType::Float(a), FlexType::Float(b)) => a.to_bits() == b.to_bits(),
            (FlexType::String(a), FlexType::String(b)) => a == b,
            (FlexType::Undefined, FlexType::Undefined) => true,
            _ => false,
        }
    }
}

impl Eq for FlexTypeKey {}

impl std::hash::Hash for FlexTypeKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(&self.0).hash(state);
        match &self.0 {
            FlexType::Integer(i) => i.hash(state),
            FlexType::Float(f) => f.to_bits().hash(state),
            FlexType::String(s) => s.hash(state),
            FlexType::Undefined => {}
            FlexType::Vector(v) => {
                v.len().hash(state);
                for f in v.iter() {
                    f.to_bits().hash(state);
                }
            }
            FlexType::List(l) => l.len().hash(state),
            FlexType::Dict(d) => d.len().hash(state),
            FlexType::DateTime(dt) => {
                dt.posix_timestamp.hash(state);
                dt.microsecond.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::aggregators::*;
    use crate::execute::compile;
    use crate::planner::PlannerNode;
    use futures::stream;

    #[tokio::test]
    async fn test_groupby_simple() {
        // Data: city, score
        // Phoenix, 10
        // Scottsdale, 20
        // Phoenix, 30
        // Scottsdale, 40
        let rows = vec![
            vec![FlexType::String("Phoenix".into()), FlexType::Integer(10)],
            vec![FlexType::String("Scottsdale".into()), FlexType::Integer(20)],
            vec![FlexType::String("Phoenix".into()), FlexType::Integer(30)],
            vec![FlexType::String("Scottsdale".into()), FlexType::Integer(40)],
        ];
        let dtypes = [FlexTypeEnum::String, FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));

        let result = groupby(
            input,
            &[0], // group by city
            &[
                AggSpec::sum(1, "total_score"),
                AggSpec::count(1, "count"),
                AggSpec::mean(1, "avg_score"),
            ],
        )
        .await
        .unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 4); // 1 key + 3 aggs

        // Collect results into a map for order-independent checking
        let mut results: HashMap<String, Vec<FlexType>> = HashMap::new();
        for i in 0..result.num_rows() {
            let row = result.row(i);
            let city = match &row[0] {
                FlexType::String(s) => s.to_string(),
                other => panic!("Expected String, got {:?}", other),
            };
            results.insert(city, row[1..].to_vec());
        }

        // Phoenix: sum=40, count=2, mean=20.0
        let phoenix = &results["Phoenix"];
        assert_eq!(phoenix[0], FlexType::Integer(40));
        assert_eq!(phoenix[1], FlexType::Integer(2));
        match &phoenix[2] {
            FlexType::Float(v) => assert!((v - 20.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }

        // Scottsdale: sum=60, count=2, mean=30.0
        let scottsdale = &results["Scottsdale"];
        assert_eq!(scottsdale[0], FlexType::Integer(60));
        assert_eq!(scottsdale[1], FlexType::Integer(2));
        match &scottsdale[2] {
            FlexType::Float(v) => assert!((v - 30.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../samples", manifest)
    }

    #[tokio::test]
    async fn test_groupby_business_sf() {
        // Group business.sf by state, count businesses per state
        use sframe_storage::sframe_reader::SFrameReader;

        let path = format!("{}/business.sf", samples_dir());
        let reader = SFrameReader::open(&path).unwrap();
        let col_names = reader.column_names().to_vec();
        let col_types: Vec<FlexTypeEnum> = reader
            .group_index
            .columns
            .iter()
            .map(|c| c.dtype)
            .collect();
        let num_rows = reader.num_rows();

        let source = PlannerNode::sframe_source(&path, col_names, col_types, num_rows);
        let stream = compile(&source).unwrap();

        // state is column 10
        let result = groupby(
            stream,
            &[10], // group by state
            &[AggSpec::count(0, "count")],
        )
        .await
        .unwrap();

        // Should have at least one state group
        assert!(result.num_rows() > 0);

        // Total count across all groups should equal 11536
        let mut total = 0i64;
        for i in 0..result.num_rows() {
            let row = result.row(i);
            match &row[1] {
                FlexType::Integer(c) => total += c,
                other => panic!("Expected Integer count, got {:?}", other),
            }
        }
        assert_eq!(total, 11536);
    }
}
