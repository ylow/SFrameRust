//! Benchmark: measure throughput of core SFrame operations.
//!
//! Generates synthetic data at configurable scale, times each operation,
//! and prints a summary table with rows/sec throughput.
//!
//! Configuration via environment variables:
//!   SFRAME_BENCH_ROWS  — number of rows (default: 1,000,000)
//!
//! Run with:
//!   cargo run -p sframe --release --example benchmark

use std::sync::Arc;
use std::time::Instant;

use sframe::{SArray, SFrame, SFrameStreamWriter};
use sframe_query::algorithms::aggregators::AggSpec;
use sframe_query::algorithms::sort::SortOrder;
use sframe_query::batch::{ColumnData, SFrameRows};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

fn num_rows() -> usize {
    std::env::var("SFRAME_BENCH_ROWS")
        .ok()
        .and_then(|s| s.replace('_', "").parse().ok())
        .unwrap_or(1_000_000)
}

fn main() {
    let n = num_rows();
    let sf_path = format!("bench.sf");
    let csv_path = format!("bench.csv");

    // Force-initialize the global cache so the spill directory is created.
    let cache_dir = sframe_io::cache_fs::global_cache_fs()
        .root()
        .to_string_lossy()
        .to_string();
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    println!();
    println!("SFrameRust Benchmark");
    println!("{}", "-".repeat(66));
    println!("  Rows:      {:>12}", format_num(n));
    println!("  Cols:      {:>12}", 5);
    println!("  Threads:   {:>12}", n_threads);
    println!("  Cache dir: {}", cache_dir);
    println!("{}", "-".repeat(66));
    println!("  {:<38} {:>10} {:>12}", "Operation", "Time", "Rows/sec");
    println!("{}", "-".repeat(66));

    // ── Data Generation ───────────────────────────────────────────────
    let t = Instant::now();
    generate_data(&sf_path, n);
    report("Generate SFrame (streaming write)", n, t);

    // ── SFrame Read ───────────────────────────────────────────────────
    let t = Instant::now();
    let sf = SFrame::read(&sf_path).unwrap();
    let nrows = sf.num_rows().unwrap(); // triggers full lazy scan
    assert_eq!(nrows, n as u64);
    report("SFrame read + full scan", n, t);

    // ── CSV Write ─────────────────────────────────────────────────────
    let t = Instant::now();
    sf.to_csv(&csv_path, None).unwrap();
    report("CSV write", n, t);

    // ── CSV Read (streaming + parallel) ───────────────────────────────
    let t = Instant::now();
    let from_csv = SFrame::from_csv(&csv_path, None).unwrap();
    let csv_rows = from_csv.num_rows().unwrap();
    assert_eq!(csv_rows, n as u64);
    report("CSV read (streaming + parallel)", n, t);

    // ── Filter ────────────────────────────────────────────────────────
    let filtered = sf
        .filter(
            "score",
            Arc::new(|v| matches!(v, FlexType::Integer(i) if *i > 500)),
        )
        .unwrap();
    println!();
    println!("  Filter query plan:");
    for line in filtered.explain().lines() {
        println!("    {}", line);
    }
    println!();
    let t = Instant::now();
    let filt_n = filtered.num_rows().unwrap() as usize;
    report(
        &format!("Filter (score > 500) -> {} rows", format_num(filt_n)),
        n,
        t,
    );

    // ── Filter 2 ────────────────────────────────────────────────────────
    let filtered = sf
        .logical_filter(sf.column("score").unwrap().gt_scalar(FlexType::Integer(500)))
        .unwrap();
    println!();
    println!("  Filter 2 query plan:");
    for line in filtered.explain().lines() {
        println!("    {}", line);
    }
    println!();
    let t = Instant::now();
    let filt_n = filtered.num_rows().unwrap() as usize;
    report(
        &format!("Filter (score > 500) -> {} rows", format_num(filt_n)),
        n,
        t,
    );

    // ── Sort ──────────────────────────────────────────────────────────
    // Sort on a subset to keep runtime reasonable
    let sort_input = if n > 2_000_000 {
        sf.head(2_000_000).unwrap()
    } else {
        sf.head(n).unwrap()
    };
    let sort_n = sort_input.num_rows().unwrap() as usize;
    let t = Instant::now();
    let sorted = sort_input.sort(&[("value", SortOrder::Ascending)]).unwrap();
    let _ = sorted.num_rows().unwrap();
    report(
        &format!("Sort by value ({} rows)", format_num(sort_n)),
        sort_n,
        t,
    );

    // ── GroupBy ───────────────────────────────────────────────────────
    let value_idx = col_idx(&sf, "value");
    let score_idx = col_idx(&sf, "score");

    let t = Instant::now();
    let grouped = sf
        .groupby(
            &["category"],
            vec![
                AggSpec::count(0, "count"),
                AggSpec::mean(value_idx, "avg_value"),
                AggSpec::sum(score_idx, "total_score"),
            ],
        )
        .unwrap();
    let group_n = grouped.num_rows().unwrap();
    report(&format!("GroupBy category -> {} groups", group_n), n, t);

    // ── Join ──────────────────────────────────────────────────────────
    // Build a small lookup table to join against
    let categories: Vec<FlexType> = (0..100)
        .map(|i| FlexType::String(format!("cat_{:03}", i).into()))
        .collect();
    let weights: Vec<FlexType> = (0..100)
        .map(|i| FlexType::Float(1.0 + (i as f64) * 0.1))
        .collect();
    let lookup = SFrame::from_columns(vec![
        (
            "category",
            SArray::from_vec(categories, FlexTypeEnum::String).unwrap(),
        ),
        (
            "weight",
            SArray::from_vec(weights, FlexTypeEnum::Float).unwrap(),
        ),
    ])
    .unwrap();

    let t = Instant::now();
    let joined = sf
        .join(
            &lookup,
            "category",
            "category",
            sframe_query::algorithms::join::JoinType::Inner,
        )
        .unwrap();
    let join_n = joined.num_rows().unwrap() as usize;
    report(
        &format!("Join with 100-row lookup -> {} rows", format_num(join_n)),
        n,
        t,
    );

    // ── Unique (on category column) ───────────────────────────────────
    let t = Instant::now();
    let uniq = sf.column("category").unwrap().unique().unwrap();
    let uniq_n = uniq.len().unwrap();
    report(&format!("Unique(category) -> {} values", uniq_n), n, t);

    // ── Chained Pipeline ──────────────────────────────────────────────
    let t = Instant::now();
    let pipeline = sf
        .filter(
            "score",
            Arc::new(|v| matches!(v, FlexType::Integer(i) if *i > 500)),
        )
        .unwrap()
        .groupby(
            &["label"],
            vec![
                AggSpec::count(0, "count"),
                AggSpec::mean(value_idx, "avg_value"),
            ],
        )
        .unwrap()
        .sort(&[("count", SortOrder::Descending)])
        .unwrap();
    let pipe_n = pipeline.num_rows().unwrap();
    report(
        &format!("Pipeline: filter+groupby+sort -> {} rows", pipe_n),
        n,
        t,
    );

    // ── SArray Reductions ─────────────────────────────────────────────
    let t = Instant::now();
    let _ = sf.column("value").unwrap().sum().unwrap();
    let _ = sf.column("value").unwrap().mean().unwrap();
    let _ = sf.column("value").unwrap().min_val().unwrap();
    let _ = sf.column("value").unwrap().max_val().unwrap();
    let _ = sf.column("value").unwrap().std_dev(1).unwrap();
    report("5x reductions (sum/mean/min/max/std)", n, t);

    println!("{}", "-".repeat(66));
    println!();
}

// ── Data Generation ───────────────────────────────────────────────────

fn generate_data(path: &str, n: usize) {
    let col_names = &["id", "category", "value", "score", "label"];
    let col_types = &[
        FlexTypeEnum::Integer,
        FlexTypeEnum::String,
        FlexTypeEnum::Float,
        FlexTypeEnum::Integer,
        FlexTypeEnum::String,
    ];

    let mut writer =
        SFrameStreamWriter::new(path, col_names, col_types).expect("failed to create writer");

    let batch_size = 50_000;
    let mut offset = 0usize;

    while offset < n {
        let end = (offset + batch_size).min(n);
        let ids: Vec<Option<i64>> = (offset..end).map(|i| Some(i as i64)).collect();
        let categories: Vec<Option<Arc<str>>> = (offset..end)
            .map(|i| Some(Arc::from(format!("cat_{:03}", i % 100).as_str())))
            .collect();
        let values: Vec<Option<f64>> = (offset..end).map(|i| Some(pseudo_random_f64(i))).collect();
        let scores: Vec<Option<i64>> = (offset..end)
            .map(|i| Some((pseudo_random_u64(i) % 1001) as i64))
            .collect();
        let labels: Vec<Option<Arc<str>>> = (offset..end)
            .map(|i| Some(Arc::from(format!("label_{}", i % 10).as_str())))
            .collect();

        let batch = SFrameRows::new(vec![
            ColumnData::Integer(ids),
            ColumnData::String(categories),
            ColumnData::Float(values),
            ColumnData::Integer(scores),
            ColumnData::String(labels),
        ])
        .unwrap();

        writer.write_batch(&batch).unwrap();
        offset = end;

        // Progress indicator for large datasets
        if n >= 1_000_000 && offset % 500_000 == 0 {
            eprint!("\r  Generating... {:.0}%", 100.0 * offset as f64 / n as f64);
        }
    }

    if n >= 1_000_000 {
        eprint!("\r  Generating... done    \n");
    }

    writer.finish().unwrap();
}

// ── Reporting ─────────────────────────────────────────────────────────

fn report(label: &str, rows: usize, start: Instant) {
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    let rows_per_sec = if secs > 0.0 {
        rows as f64 / secs
    } else {
        f64::INFINITY
    };

    let time_str = if secs >= 1.0 {
        format!("{:.2}s", secs)
    } else {
        format!("{:.0}ms", secs * 1000.0)
    };

    println!(
        "  {:<38} {:>10} {:>12}",
        truncate(label, 38),
        time_str,
        format_num(rows_per_sec as usize)
    );
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}

fn format_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

fn col_idx(sf: &SFrame, name: &str) -> usize {
    sf.column_names()
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("column '{}' not found", name))
}

// ── Deterministic pseudo-random (no rand dependency) ──────────────────

fn pseudo_random_u64(seed: usize) -> u64 {
    // SplitMix64
    let mut x = seed as u64;
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn pseudo_random_f64(seed: usize) -> f64 {
    (pseudo_random_u64(seed) >> 11) as f64 / (1u64 << 53) as f64 * 1000.0
}
