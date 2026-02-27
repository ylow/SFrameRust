//! Benchmark: Large-scale SFrame with lazy evaluation and external memory compute.
//!
//! Demonstrates:
//! 1. Streaming data generation to disk in constant memory
//! 2. Lazy open — metadata read only, no data loaded
//! 3. Streaming filter — scans all rows without materializing the full dataset
//! 4. Streaming groupby — aggregates with O(groups) memory, not O(rows)
//! 5. Lazy pipeline: filter → sort on the reduced subset
//! 6. Full-dataset sort (opt-in with SFRAME_BENCH_FULL_SORT=1)
//!
//! Default configuration generates ~150M rows (~10 GB uncompressed).
//! Override with: SFRAME_BENCH_ROWS=50000000 cargo run ...
//!
//! Run with:
//!   cargo run -p sframe --release --example large_benchmark

use std::sync::Arc;
use std::time::Instant;

use sframe::{SFrame, SFrameStreamWriter};
use sframe_query::algorithms::aggregators::AggSpec;
use sframe_query::algorithms::sort::SortOrder;
use sframe_query::batch::{ColumnData, SFrameRows};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

// ── Configuration ──────────────────────────────────────────────────────

fn num_rows() -> u64 {
    std::env::var("SFRAME_BENCH_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150_000_000)
}

const BATCH_SIZE: usize = 500_000;
const NUM_USERS: i64 = 1_000_000;
const NUM_PRODUCTS: i64 = 100_000;

const CATEGORIES: &[&str] = &[
    "Electronics", "Clothing", "Food", "Home", "Sports",
    "Books", "Toys", "Health", "Auto", "Garden",
    "Music", "Office",
];

const REGIONS: &[&str] = &[
    "North", "South", "East", "West", "Central",
    "Northeast", "Southeast", "Northwest", "Southwest", "Midwest",
];

// ── Simple xorshift64 PRNG (no external dependency) ───────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    fn next_i64_range(&mut self, max: i64) -> i64 {
        (self.next_u64() % max as u64) as i64
    }

    #[inline]
    fn next_f64_range(&mut self, min: f64, max: f64) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        min + t * (max - min)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn dir_size_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut val = bytes as f64;
    for &unit in UNITS {
        if val < 1024.0 {
            return format!("{:.2} {}", val, unit);
        }
        val /= 1024.0;
    }
    format!("{:.2} PB", val)
}

fn separator(title: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  {title}");
    println!("═══════════════════════════════════════════════════════════");
}

// ── Phase 1: Streaming data generation ────────────────────────────────

fn generate_and_write(path: &str, total_rows: u64) {
    separator(&format!("Phase 1: Generate {} rows to disk (streaming)", format_count(total_rows)));

    let col_names: &[&str] = &[
        "user_id", "product_id", "category", "amount", "quantity", "score", "region",
    ];
    let col_types: &[FlexTypeEnum] = &[
        FlexTypeEnum::Integer,  // user_id
        FlexTypeEnum::Integer,  // product_id
        FlexTypeEnum::String,   // category
        FlexTypeEnum::Float,    // amount
        FlexTypeEnum::Integer,  // quantity
        FlexTypeEnum::Float,    // score
        FlexTypeEnum::String,   // region
    ];

    let mut writer = SFrameStreamWriter::new(path, col_names, col_types)
        .expect("failed to create writer");

    // Pre-allocate Arc<str> for categories and regions to avoid per-row heap allocs.
    let cat_arcs: Vec<Arc<str>> = CATEGORIES.iter().map(|&s| Arc::from(s)).collect();
    let reg_arcs: Vec<Arc<str>> = REGIONS.iter().map(|&s| Arc::from(s)).collect();

    let mut rng = Rng::new(42);
    let num_batches = ((total_rows as usize) + BATCH_SIZE - 1) / BATCH_SIZE;
    let t0 = Instant::now();

    for batch_idx in 0..num_batches {
        let rows_in_batch = if batch_idx == num_batches - 1 {
            (total_rows as usize) - batch_idx * BATCH_SIZE
        } else {
            BATCH_SIZE
        };

        let mut user_ids = Vec::with_capacity(rows_in_batch);
        let mut product_ids = Vec::with_capacity(rows_in_batch);
        let mut categories: Vec<Option<Arc<str>>> = Vec::with_capacity(rows_in_batch);
        let mut amounts = Vec::with_capacity(rows_in_batch);
        let mut quantities = Vec::with_capacity(rows_in_batch);
        let mut scores = Vec::with_capacity(rows_in_batch);
        let mut regions: Vec<Option<Arc<str>>> = Vec::with_capacity(rows_in_batch);

        for _ in 0..rows_in_batch {
            user_ids.push(Some(rng.next_i64_range(NUM_USERS)));
            product_ids.push(Some(rng.next_i64_range(NUM_PRODUCTS)));
            categories.push(Some(Arc::clone(&cat_arcs[rng.next_u64() as usize % cat_arcs.len()])));
            amounts.push(Some(rng.next_f64_range(1.0, 10_000.0)));
            quantities.push(Some(rng.next_i64_range(100) + 1));
            scores.push(Some(rng.next_f64_range(0.0, 100.0)));
            regions.push(Some(Arc::clone(&reg_arcs[rng.next_u64() as usize % reg_arcs.len()])));
        }

        let batch = SFrameRows::new(vec![
            ColumnData::Integer(user_ids),
            ColumnData::Integer(product_ids),
            ColumnData::String(categories),
            ColumnData::Float(amounts),
            ColumnData::Integer(quantities),
            ColumnData::Float(scores),
            ColumnData::String(regions),
        ])
        .expect("batch creation failed");

        writer.write_batch(&batch).expect("write_batch failed");

        let rows_written = std::cmp::min(
            ((batch_idx + 1) * BATCH_SIZE) as u64,
            total_rows,
        );
        if (batch_idx + 1) % 20 == 0 || batch_idx == num_batches - 1 {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = rows_written as f64 / elapsed / 1_000_000.0;
            println!(
                "  Written {:>6.1}M / {:.1}M rows  ({:.1} M rows/sec)",
                rows_written as f64 / 1e6,
                total_rows as f64 / 1e6,
                rate,
            );
        }
    }

    writer.finish().expect("finish failed");
    let elapsed = t0.elapsed();

    let disk_bytes = dir_size_bytes(std::path::Path::new(path));
    let logical_bytes = total_rows * 68; // approximate bytes per row of raw data

    println!();
    println!("  Generation complete.");
    println!("    Time:             {:.1}s ({:.1} M rows/sec)", elapsed.as_secs_f64(), total_rows as f64 / elapsed.as_secs_f64() / 1e6);
    println!("    On-disk size:     {} (LZ4 compressed)", human_bytes(disk_bytes));
    println!("    Logical size:     ~{} (uncompressed)", human_bytes(logical_bytes));
    println!("    Compression:      {:.1}x", logical_bytes as f64 / disk_bytes as f64);
}

// ── Phase 2: Lazy open ────────────────────────────────────────────────

fn lazy_open(path: &str) -> SFrame {
    separator("Phase 2: Open SFrame (lazy — metadata only)");

    let t0 = Instant::now();
    let sf = SFrame::read(path).expect("read failed");
    let open_time = t0.elapsed();

    println!(
        "  Opened {} rows x {} columns in {:.4}s (no data loaded)",
        format_count(sf.num_rows().unwrap()),
        sf.num_columns(),
        open_time.as_secs_f64(),
    );
    println!("  Schema:");
    for (name, dtype) in sf.schema() {
        println!("    {:<15} {:?}", name, dtype);
    }

    sf
}

// ── Phase 3: Streaming filter ─────────────────────────────────────────

fn bench_filter(sf: &SFrame) {
    separator("Phase 3: Streaming filter — amount > 9000.0 (top ~10%)");

    // Creating the filter plan is instant (lazy)
    let t0 = Instant::now();
    let filtered = sf
        .filter(
            "amount",
            Arc::new(|v| matches!(v, FlexType::Float(f) if *f > 9000.0)),
        )
        .expect("filter failed");
    let plan_time = t0.elapsed();
    println!("  Plan created in {:.6}s (no data touched)", plan_time.as_secs_f64());

    // Materializing the count streams through all rows on disk
    let t0 = Instant::now();
    let n_filtered = filtered.num_rows().unwrap();
    let scan_time = t0.elapsed();
    let total = sf.num_rows().unwrap();

    println!(
        "  Scanned {} rows, {} passed filter ({:.2}%)",
        format_count(total),
        format_count(n_filtered),
        n_filtered as f64 / total as f64 * 100.0,
    );
    println!(
        "  Streaming scan time: {:.2}s ({:.1} M rows/sec)",
        scan_time.as_secs_f64(),
        total as f64 / scan_time.as_secs_f64() / 1e6,
    );
}

// ── Phase 4: Streaming groupby ────────────────────────────────────────

fn bench_groupby(sf: &SFrame) {
    separator("Phase 4: Streaming groupby — by category");

    let amount_idx = col_idx(sf, "amount");
    let score_idx = col_idx(sf, "score");
    let qty_idx = col_idx(sf, "quantity");

    let t0 = Instant::now();
    let grouped = sf
        .groupby(
            &["category"],
            vec![
                AggSpec::count(amount_idx, "count"),
                AggSpec::sum(amount_idx, "total_amount"),
                AggSpec::mean(score_idx, "avg_score"),
                AggSpec::sum(qty_idx, "total_quantity"),
            ],
        )
        .expect("groupby failed")
        .sort(&[("total_amount", SortOrder::Descending)])
        .expect("sort failed");
    let elapsed = t0.elapsed();

    println!("{}", grouped);
    println!(
        "  GroupBy ({} groups) + sort in {:.2}s ({:.1} M rows/sec)",
        grouped.num_rows().unwrap(),
        elapsed.as_secs_f64(),
        sf.num_rows().unwrap() as f64 / elapsed.as_secs_f64() / 1e6,
    );
    println!("  Memory: O(groups) = O({}) — NOT O(rows)", grouped.num_rows().unwrap());
}

// ── Phase 5: Lazy pipeline — filter → groupby → sort ──────────────────

fn bench_pipeline(sf: &SFrame) {
    separator("Phase 5: Chained lazy pipeline — filter → groupby → sort");
    println!("  Pipeline: amount > 5000 → groupby region → sort by total_amount");

    let amount_idx = col_idx(sf, "amount");

    let t0 = Instant::now();
    let result = sf
        .filter(
            "amount",
            Arc::new(|v| matches!(v, FlexType::Float(f) if *f > 5000.0)),
        )
        .unwrap()
        .groupby(
            &["region"],
            vec![
                AggSpec::count(amount_idx, "count"),
                AggSpec::sum(amount_idx, "total_amount"),
                AggSpec::mean(amount_idx, "avg_amount"),
            ],
        )
        .unwrap()
        .sort(&[("total_amount", SortOrder::Descending)])
        .unwrap();
    let elapsed = t0.elapsed();

    println!("{}", result);
    println!(
        "  Full pipeline in {:.2}s ({:.1} M rows/sec)",
        elapsed.as_secs_f64(),
        sf.num_rows().unwrap() as f64 / elapsed.as_secs_f64() / 1e6,
    );
}

// ── Phase 6: Sort (filtered subset) ───────────────────────────────────

fn bench_sort_filtered(sf: &SFrame) {
    separator("Phase 6: Sort filtered subset — top 10 highest amounts");
    println!("  Pipeline: filter amount > 9000 → sort by amount DESC → head(10)");

    let t0 = Instant::now();
    let filtered = sf
        .filter(
            "amount",
            Arc::new(|v| matches!(v, FlexType::Float(f) if *f > 9000.0)),
        )
        .unwrap();
    let n = filtered.num_rows().unwrap();

    let sorted = filtered
        .select(&["user_id", "category", "amount", "score", "region"])
        .unwrap()
        .sort(&[("amount", SortOrder::Descending)])
        .unwrap();
    let top = sorted.head(10).unwrap();
    let elapsed = t0.elapsed();

    println!("{}", top);
    println!(
        "  Filtered to {} rows, sorted, took top 10 in {:.2}s",
        format_count(n),
        elapsed.as_secs_f64(),
    );
}

// ── Phase 7 (opt-in): Full-dataset sort ───────────────────────────────

fn bench_full_sort(sf: &SFrame) {
    separator("Phase 7: Full-dataset sort (all rows by amount DESC)");
    println!("  WARNING: This materializes ALL rows into memory for sorting.");
    println!("  For {}M rows, expect ~{} RAM usage.",
        sf.num_rows().unwrap() as f64 / 1e6,
        human_bytes(sf.num_rows().unwrap() * 96), // ~96 bytes/row in ColumnData
    );
    println!();

    let t0 = Instant::now();
    let sorted = sf
        .select(&["user_id", "category", "amount", "region"])
        .unwrap()
        .sort(&[("amount", SortOrder::Descending)])
        .unwrap();
    let top = sorted.head(10).unwrap();
    let elapsed = t0.elapsed();

    println!("{}", top);
    println!(
        "  Full sort of {} rows in {:.2}s ({:.1} M rows/sec)",
        format_count(sf.num_rows().unwrap()),
        elapsed.as_secs_f64(),
        sf.num_rows().unwrap() as f64 / elapsed.as_secs_f64() / 1e6,
    );
}

// ── Utility ────────────────────────────────────────────────────────────

fn col_idx(sf: &SFrame, name: &str) -> usize {
    sf.column_names()
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("column '{}' not found", name))
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() {
    let total_rows = num_rows();
    let do_full_sort = std::env::var("SFRAME_BENCH_FULL_SORT").is_ok();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       SFrame Large-Scale Benchmark                       ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Rows:    {:>12}                                    ║", format_count(total_rows));
    println!("║  Columns: 7 (3 int, 2 float, 2 string)                  ║");
    println!("║  Est. uncompressed: ~{}                          ║", human_bytes(total_rows * 68));
    println!("╚═══════════════════════════════════════════════════════════╝");

    let tmp = tempfile::tempdir().expect("failed to create tempdir");
    let sf_path = tmp.path().join("benchmark.sf");
    let path_str = sf_path.to_str().unwrap();

    // Phase 1: Streaming generation (constant memory)
    generate_and_write(path_str, total_rows);

    // Phase 2: Lazy open
    let sf = lazy_open(path_str);

    // Phase 3: Streaming filter
    bench_filter(&sf);

    // Phase 4: Streaming groupby
    bench_groupby(&sf);

    // Phase 5: Chained pipeline
    bench_pipeline(&sf);

    // Phase 6: Sort on filtered subset
    bench_sort_filtered(&sf);

    // Phase 7: Full sort (opt-in)
    if do_full_sort {
        bench_full_sort(&sf);
    } else {
        separator("Phase 7: Full-dataset sort — SKIPPED");
        println!("  Set SFRAME_BENCH_FULL_SORT=1 to enable.");
        println!("  Requires ~{} RAM for {} rows.", human_bytes(total_rows * 96), format_count(total_rows));
    }

    // Summary
    separator("Done");
    println!("  Temporary SFrame at: {}", path_str);
    println!("  Will be cleaned up on exit.");
}
