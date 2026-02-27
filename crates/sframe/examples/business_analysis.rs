//! Example: Load business.csv and do some exploratory analysis.
//!
//! Run with:
//!   cargo run -p sframe --example business_analysis

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

    // ── 1. Load CSV ─────────────────────────────────────────────────
    println!("Loading business.csv ...");
    let sf = SFrame::from_csv(&csv_path, None).expect("failed to load CSV");
    println!("Loaded {} rows x {} columns\n", sf.num_rows().unwrap(), sf.num_columns());

    // Print schema
    println!("Schema:");
    for (name, dtype) in sf.schema() {
        println!("  {:<20} {}", name, dtype);
    }
    println!();

    // Show first few rows
    println!("First 5 rows:");
    println!("{}\n", sf.head(5).unwrap());

    // ── 2. Filter: highly-rated businesses (stars >= 4.0) ───────────
    let highly_rated = sf
        .filter(
            "stars",
            Arc::new(|v| matches!(v, FlexType::Float(f) if *f >= 4.0)),
        )
        .expect("filter failed");
    println!(
        "Highly rated (>= 4 stars): {} / {} businesses ({:.1}%)\n",
        highly_rated.num_rows().unwrap(),
        sf.num_rows().unwrap(),
        highly_rated.num_rows().unwrap() as f64 / sf.num_rows().unwrap() as f64 * 100.0,
    );

    // ── 3. Select a few columns and sort by review_count DESC ───────
    let top_reviewed = sf
        .select(&["name", "city", "state", "stars", "review_count"])
        .expect("select failed")
        .sort(&[("review_count", SortOrder::Descending)])
        .expect("sort failed");
    println!("Top 10 most-reviewed businesses:");
    println!("{}\n", top_reviewed.head(10).unwrap());

    // ── 4. Groupby state: count + average stars ─────────────────────
    let stars_col = sf
        .column_names()
        .iter()
        .position(|n| n == "stars")
        .unwrap();
    let review_col = sf
        .column_names()
        .iter()
        .position(|n| n == "review_count")
        .unwrap();

    let by_state = sf
        .groupby(
            &["state"],
            vec![
                AggSpec::count(stars_col, "num_businesses"),
                AggSpec::mean(stars_col, "avg_stars"),
                AggSpec::sum(review_col, "total_reviews"),
            ],
        )
        .expect("groupby failed")
        .sort(&[("num_businesses", SortOrder::Descending)])
        .expect("sort failed");
    println!("Business stats by state:");
    println!("{}\n", by_state);

    // ── 5. Add a derived column: stars * review_count ───────────────
    let stars_arr = sf.column("stars").unwrap().clone();
    let reviews_arr = sf.column("review_count").unwrap().clone();

    // Materialize both to build a combined metric
    let stars_vec = stars_arr.to_vec().unwrap();
    let reviews_vec = reviews_arr.to_vec().unwrap();

    let weighted: Vec<FlexType> = stars_vec
        .iter()
        .zip(reviews_vec.iter())
        .map(|(s, r)| {
            let star_val = match s {
                FlexType::Float(f) => *f,
                FlexType::Integer(i) => *i as f64,
                _ => 0.0,
            };
            let rev_val = match r {
                FlexType::Integer(i) => *i as f64,
                FlexType::Float(f) => *f,
                _ => 0.0,
            };
            FlexType::Float(star_val * rev_val)
        })
        .collect();

    let weighted_col =
        SArray::from_vec(weighted, FlexTypeEnum::Float).expect("from_vec failed");

    let enriched = sf
        .select(&["name", "city", "stars", "review_count"])
        .unwrap()
        .add_column("weighted_score", weighted_col)
        .expect("add_column failed")
        .sort(&[("weighted_score", SortOrder::Descending)])
        .expect("sort failed");

    println!("Top 10 by weighted score (stars * review_count):");
    println!("{}\n", enriched.head(10).unwrap());

    // ── 6. Save a filtered subset to disk ───────────────────────────
    let tmp = tempfile::tempdir().expect("tempdir failed");
    let out_path = tmp.path().join("top_businesses.sf");
    let out_str = out_path.to_str().unwrap();

    let to_save = highly_rated
        .select(&["business_id", "name", "city", "state", "stars", "review_count"])
        .unwrap();
    to_save.save(out_str).expect("save failed");
    println!(
        "Saved {} highly-rated businesses to {}",
        to_save.num_rows().unwrap(),
        out_str
    );

    // Re-read to verify
    let reloaded = SFrame::read(out_str).expect("reload failed");
    println!(
        "Re-loaded: {} rows x {} columns",
        reloaded.num_rows().unwrap(),
        reloaded.num_columns()
    );
    println!("{}", reloaded.head(5).unwrap());
}
