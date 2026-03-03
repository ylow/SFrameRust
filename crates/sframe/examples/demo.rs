//! Comprehensive demo of the SFrame/SArray API.
//!
//! Walks through loading, inspecting, filtering, sorting, transforming,
//! aggregating, joining, and saving data using the bundled business.csv
//! sample dataset (11,536 Yelp businesses).
//!
//! Run with:
//!   cargo run -p sframe --example demo

use std::sync::Arc;

use sframe::{SArray, SFrame};
use sframe_query::algorithms::aggregators::AggSpec;
use sframe_query::algorithms::sort::SortOrder;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

fn main() {
    let csv_path = format!(
        "{}/../../samples/business.csv",
        env!("CARGO_MANIFEST_DIR")
    );

    // ── 1. Load & Inspect ─────────────────────────────────────────────
    section("1. Load & Inspect");

    let sf = SFrame::from_csv(&csv_path, None).expect("failed to load CSV");
    println!("Loaded {} rows x {} columns", sf.num_rows().unwrap(), sf.num_columns());
    println!("\nSchema:");
    for (name, dtype) in sf.schema() {
        println!("  {:<20} {:?}", name, dtype);
    }

    println!("\nFirst 5 rows:");
    let head = sf.head(5).unwrap();
    println!("{}", head);

    // ── 2. Column Access ──────────────────────────────────────────────
    section("2. Column Access — SArray basics");

    let stars = sf.column("stars").unwrap();
    println!("stars column: dtype={:?}, len={}", stars.dtype(), stars.len().unwrap());
    println!("First 8 values: {:?}", stars.head(8).unwrap());

    let city = sf.column("city").unwrap();
    println!("\ncity column: dtype={:?}", city.dtype());
    println!("First 5 values: {:?}", city.head(5).unwrap());

    // ── 3. Reductions ─────────────────────────────────────────────────
    section("3. Reductions — summary statistics");

    println!("stars.sum()     = {:?}", stars.sum().unwrap());
    println!("stars.mean()    = {:?}", stars.mean().unwrap());
    println!("stars.min_val() = {:?}", stars.min_val().unwrap());
    println!("stars.max_val() = {:?}", stars.max_val().unwrap());
    println!("stars.std_dev() = {:?}", stars.std_dev(1).unwrap());

    let review_count = sf.column("review_count").unwrap();
    println!("\nreview_count.sum()  = {:?}", review_count.sum().unwrap());
    println!("review_count.mean() = {:?}", review_count.mean().unwrap());

    // ── 4. Filtering ──────────────────────────────────────────────────
    section("4. Filtering — high-rated businesses");

    let high_rated = sf
        .filter("stars", Arc::new(|v| matches!(v, FlexType::Float(f) if *f >= 4.5)))
        .unwrap();
    println!(
        "Businesses with stars >= 4.5: {} (out of {})",
        high_rated.num_rows().unwrap(),
        sf.num_rows().unwrap()
    );
    println!("\nSample:");
    println!("{}", high_rated.head(5).unwrap());

    // ── 5. Sorting ────────────────────────────────────────────────────
    section("5. Sorting — top businesses by review count");

    let sorted = sf
        .sort(&[("review_count", SortOrder::Descending)])
        .unwrap();
    println!("Top 5 by review_count:");
    let top5 = sorted.head(5).unwrap();
    println!("{}", top5);

    // Multi-key sort
    let multi_sorted = sf
        .sort(&[
            ("state", SortOrder::Ascending),
            ("stars", SortOrder::Descending),
        ])
        .unwrap();
    println!("Sorted by state ASC, stars DESC — first 5:");
    println!("{}", multi_sorted.head(5).unwrap());

    // ── 6. Column Transforms ──────────────────────────────────────────
    section("6. Column Transforms — derived columns");

    // Scalar arithmetic: stars * 20 → percentage
    let pct = stars.mul_scalar(FlexType::Float(20.0));
    println!("stars * 20 (first 5): {:?}", pct.head(5).unwrap());

    // Apply: classify review_count
    let rc_class = review_count.apply(
        Arc::new(|v| match v {
            FlexType::Integer(n) if *n >= 100 => FlexType::String("popular".into()),
            FlexType::Integer(n) if *n >= 20 => FlexType::String("moderate".into()),
            _ => FlexType::String("niche".into()),
        }),
        FlexTypeEnum::String,
    );
    println!("review_count classified (first 8): {:?}", rc_class.head(8).unwrap());

    // Add derived column to frame
    let sf_with_class = sf.add_column("popularity", rc_class).unwrap();
    println!("\nSchema after add_column:");
    for (name, dtype) in sf_with_class.schema() {
        println!("  {:<20} {:?}", name, dtype);
    }

    // ── 7. GroupBy ────────────────────────────────────────────────────
    section("7. GroupBy — aggregate by state");

    let stars_idx = col_idx(&sf, "stars");
    let rc_idx = col_idx(&sf, "review_count");

    let by_state = sf
        .groupby(
            &["state"],
            vec![
                AggSpec::count(0, "num_businesses"),
                AggSpec::mean(stars_idx, "avg_stars"),
                AggSpec::sum(rc_idx, "total_reviews"),
            ],
        )
        .unwrap()
        .sort(&[("num_businesses", SortOrder::Descending)])
        .unwrap();

    println!("Businesses by state (top 10):");
    println!("{}", by_state.head(10).unwrap());

    // Groupby with multiple keys
    let by_state_stars = sf
        .groupby(
            &["state", "stars"],
            vec![AggSpec::count(0, "count")],
        )
        .unwrap()
        .sort(&[("count", SortOrder::Descending)])
        .unwrap();

    println!("Top 5 (state, stars) combinations:");
    println!("{}", by_state_stars.head(5).unwrap());

    // ── 8. Join ───────────────────────────────────────────────────────
    section("8. Join — enrich with external data");

    // Build a small lookup table of state abbreviations → full names
    let state_names = SFrame::from_columns(vec![
        (
            "state",
            SArray::from_vec(
                vec!["AZ", "NV", "WI", "PA", "NC", "SC", "IL", "CA"]
                    .into_iter()
                    .map(|s| FlexType::String(s.into()))
                    .collect(),
                FlexTypeEnum::String,
            )
            .unwrap(),
        ),
        (
            "state_name",
            SArray::from_vec(
                vec![
                    "Arizona",
                    "Nevada",
                    "Wisconsin",
                    "Pennsylvania",
                    "North Carolina",
                    "South Carolina",
                    "Illinois",
                    "California",
                ]
                .into_iter()
                .map(|s| FlexType::String(s.into()))
                .collect(),
                FlexTypeEnum::String,
            )
            .unwrap(),
        ),
    ])
    .unwrap();

    println!("State lookup table:");
    println!("{}", state_names);

    let enriched = by_state
        .join(&state_names, "state", "state", sframe_query::algorithms::join::JoinType::Left)
        .unwrap();

    println!("After left join with state names (top 5):");
    println!("{}", enriched.head(5).unwrap());

    // ── 9. Unique & TopK ──────────────────────────────────────────────
    section("9. Unique & TopK");

    let unique_states = sf.column("state").unwrap().unique().unwrap();
    println!("Unique states: {:?}", unique_states.to_vec().unwrap());

    let unique_cities = sf.column("city").unwrap().unique().unwrap();
    println!("Number of unique cities: {}", unique_cities.len().unwrap());

    let top_reviewed = sf.topk("review_count", 5, false).unwrap();
    println!("\nTop 5 by review_count:");
    println!("{}", top_reviewed);

    // ── 10. Missing Values ────────────────────────────────────────────
    section("10. Missing Values");

    // Create a column with some missing values
    let with_nulls = SArray::from_vec(
        vec![
            FlexType::Integer(10),
            FlexType::Undefined,
            FlexType::Integer(30),
            FlexType::Undefined,
            FlexType::Integer(50),
        ],
        FlexTypeEnum::Integer,
    )
    .unwrap();

    println!("Array with nulls: {:?}", with_nulls.to_vec().unwrap());
    println!("countna: {}", with_nulls.countna().unwrap());
    println!("is_na:   {:?}", with_nulls.is_na().to_vec().unwrap());

    let filled = with_nulls.fillna(FlexType::Integer(0));
    println!("fillna(0): {:?}", filled.to_vec().unwrap());

    let dropped = with_nulls.dropna();
    println!("dropna:    {:?}", dropped.to_vec().unwrap());

    // ── 11. String Operations ─────────────────────────────────────────
    section("11. String Operations");

    let has_grill = sf.column("name").unwrap().contains("Grill");
    let grill_count: i64 = has_grill
        .to_vec()
        .unwrap()
        .iter()
        .filter(|v| matches!(v, FlexType::Integer(1)))
        .count() as i64;
    println!("Businesses with 'Grill' in name: {}", grill_count);

    let pizza_biz = sf
        .filter("name", Arc::new(|v| {
            matches!(v, FlexType::String(s) if s.contains("Grill"))
        }))
        .unwrap();
    println!("Grill places (first 5):");
    println!(
        "{}",
        pizza_biz
            .select(&["name", "city", "stars"])
            .unwrap()
            .head(5)
            .unwrap()
    );

    // ── 12. Rolling Aggregations ──────────────────────────────────────
    section("12. Rolling Aggregations");

    let values = SArray::from_vec(
        (1..=20).map(|i| FlexType::Float(i as f64)).collect(),
        FlexTypeEnum::Float,
    )
    .unwrap();

    let rolling_avg = values.rolling_mean(2, 2, 1).unwrap();
    println!("Values (first 10):     {:?}", values.head(10).unwrap());
    println!("Rolling mean (w=5, first 10): {:?}", rolling_avg.head(10).unwrap());

    // ── 13. Type Casting ──────────────────────────────────────────────
    section("13. Type Casting");

    let ints = SArray::from_vec(
        vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)],
        FlexTypeEnum::Integer,
    )
    .unwrap();

    let as_float = ints.astype(FlexTypeEnum::Float, false);
    let as_string = ints.astype(FlexTypeEnum::String, false);
    println!("int:    {:?}", ints.to_vec().unwrap());
    println!("float:  {:?}", as_float.to_vec().unwrap());
    println!("string: {:?}", as_string.to_vec().unwrap());

    // ── 14. Save / Load Roundtrip ─────────────────────────────────────
    section("14. Save / Load Roundtrip");

    let tmp = tempfile::tempdir().unwrap();
    let sf_path = format!("{}/demo.sf", tmp.path().display());
    let csv_out = format!("{}/demo.csv", tmp.path().display());
    let json_out = format!("{}/demo.jsonl", tmp.path().display());

    // Save as SFrame binary
    sf.save(&sf_path).unwrap();
    let reloaded = SFrame::read(&sf_path).unwrap();
    println!(
        "SFrame save/load: {} rows x {} cols (matches: {})",
        reloaded.num_rows().unwrap(),
        reloaded.num_columns(),
        reloaded.num_rows().unwrap() == sf.num_rows().unwrap()
    );

    // Save as CSV
    sf.head(100).unwrap().to_csv(&csv_out, None).unwrap();
    let from_csv = SFrame::from_csv(&csv_out, None).unwrap();
    println!("CSV roundtrip (100 rows): {} rows", from_csv.num_rows().unwrap());

    // Save as JSON Lines
    sf.head(100).unwrap().to_json(&json_out).unwrap();
    let from_json = SFrame::from_json(&json_out).unwrap();
    println!("JSON roundtrip (100 rows): {} rows", from_json.num_rows().unwrap());

    println!("\n  Demo complete.");
}

// ── Helpers ───────────────────────────────────────────────────────────

fn section(title: &str) {
    println!("\n{}", "=".repeat(60));
    println!("  {}", title);
    println!("{}\n", "=".repeat(60));
}

fn col_idx(sf: &SFrame, name: &str) -> usize {
    sf.column_names()
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("column '{}' not found", name))
}
