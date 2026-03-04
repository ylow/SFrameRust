# SFrame

A scalable, out-of-core columnar data store written in Rust, with Python bindings.

SFrame provides disk-backed tabular data structures that can handle datasets
larger than memory. It supports lazy evaluation, data-parallel query execution,
and external-memory algorithms (sort, groupby, join) that spill to disk when
needed.

This is a Rust port of the [C++ SFrame](https://github.com/turi-code/SFrame)
library, reading the V2 on-disk format.

## Features

- **Out-of-core**: Operates on data larger than RAM via streaming I/O and
  disk-backed spill
- **Columnar compression**: Frame-of-reference bit packing, dictionary encoding,
  LZ4 block compression
- **Lazy evaluation**: Operations build a query plan; execution is deferred
  until materialization
- **Data-parallel**: Queries automatically split across threads by row range
- **Pluggable storage**: VFS trait with local filesystem, HTTP, S3, and HDFS
  backends
- **Python bindings**: PyO3-based Python package with a pandas-like API

## Workspace Structure

| Crate | Description |
|-------|-------------|
| `sframe-types` | Core data types (`FlexType`, error types, varint encoding) |
| `sframe-config` | Runtime configuration via environment variables |
| `sframe-io` | VFS trait and filesystem backends (local, HTTP, S3, HDFS, cache) |
| `sframe-storage` | V2 format reader/writer, block codecs, segment handling |
| `sframe-query` | Query planner, optimizer, and algorithms (groupby, join, sort, CSV/JSON) |
| `sframe` | High-level API: `SFrame`, `SArray`, `SFrameStreamWriter` |
| `sframe-py` | Python bindings via PyO3 |

## Quick Start (Rust)

Add to your `Cargo.toml`:

```toml
[dependencies]
sframe = { path = "crates/sframe" }
sframe-types = { path = "crates/sframe-types" }
sframe-query = { path = "crates/sframe-query" }
```

```rust
use std::sync::Arc;
use sframe::{SArray, SFrame};
use sframe_query::algorithms::aggregators::AggSpec;
use sframe_query::algorithms::sort::SortOrder;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

fn main() {
    // Load a CSV
    let sf = SFrame::from_csv("data.csv", None).unwrap();
    println!("{}", sf.head(5).unwrap());

    // Filter rows
    let filtered = sf
        .filter("stars", Arc::new(|v| matches!(v, FlexType::Float(f) if *f >= 4.0)))
        .unwrap();

    // Groupby with aggregation
    let grouped = sf.groupby(
        &["state"],
        vec![
            AggSpec::count(0, "count"),
            AggSpec::mean(sf.column_index("stars").unwrap(), "avg_stars"),
        ],
    ).unwrap();

    // Sort
    let sorted = grouped.sort(&[("count", SortOrder::Descending)]).unwrap();
    println!("{}", sorted);

    // Save as SFrame
    sf.save("output.sf").unwrap();
}
```

## Quick Start (Python)

```bash
cd crates/sframe-py
pip install maturin
maturin develop
```

```python
from sframe import SFrame, SArray, aggregate

# Create from columns
sf = SFrame.from_columns({
    "name": SArray(["Alice", "Bob", "Carol"]),
    "city": SArray(["NYC", "LA", "NYC"]),
    "score": SArray([90, 85, 95]),
})

# Filter, sort, aggregate
print(sf.head(5))
grouped = sf.groupby(["city"], {"avg_score": aggregate.MEAN("score")})
print(grouped)

# Read/write SFrame format
sf.save("output.sf")
sf2 = SFrame.read("output.sf")

# CSV export
sf.to_csv("output.csv")
```

## Building

```bash
# Build all crates
cargo build

# Build with optional S3 and HTTP support
cargo build -p sframe-io --features s3,http

# Run tests
cargo test

# Run examples
cargo run -p sframe --example demo
cargo run -p sframe --example benchmark
```

## Configuration

Runtime behavior is controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SFRAME_CACHE_CAPACITY` | 2 GiB | Total VFS cache size |
| `SFRAME_CACHE_CAPACITY_PER_FILE` | 2 GiB | Per-file cache limit |
| `SFRAME_SOURCE_BATCH_SIZE` | 4096 | Rows per source batch |
| `SFRAME_ROWS_PER_SEGMENT` | 1,000,000 | Rows per on-disk segment |
| `SFRAME_SORT_BUFFER_SIZE` | 256 MiB | External sort memory budget |
| `SFRAME_GROUPBY_BUFFER_NUM_ROWS` | 1,000,000 | Groupby memory budget (rows) |
| `SFRAME_JOIN_BUFFER_NUM_CELLS` | 50,000,000 | Join memory budget (cells) |
| `SFRAME_SOURCE_PREFETCH_SEGMENTS` | 2 | Segments to prefetch ahead |

## API Overview

### SArray

In-memory or disk-backed typed column supporting:

- **Arithmetic**: `add`, `sub`, `mul`, `div` (scalar and element-wise)
- **Comparisons**: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- **Reductions**: `sum`, `mean`, `min_val`, `max_val`, `std_dev`, `variance`
- **Transforms**: `apply`, `filter`, `astype`, `clip`, `sort`, `unique`, `sample`
- **Missing values**: `countna`, `dropna`, `fillna`, `is_na`
- **String ops**: `contains`, `count_bag_of_words`, `count_ngrams`
- **Rolling**: `rolling_sum`, `rolling_mean`, `rolling_min`, `rolling_max`
- **Approximate**: `approx_count_distinct`, `frequent_items`

### SFrame

Disk-backed table of named `SArray` columns:

- **I/O**: `read`, `save`, `from_csv`, `from_json`, `to_csv`, `to_json`
- **Schema**: `num_rows`, `num_columns`, `column_names`, `column_types`, `schema`
- **Selection**: `column`, `select`, `head`, `tail`, `filter`, `sample`, `topk`
- **Mutation**: `add_column`, `remove_column`, `replace_column`, `rename`
- **Aggregation**: `groupby` with count, sum, mean, min, max, variance, std,
  count_distinct, concat, select_one
- **Joins**: `join` / `join_on` (inner, left, right, full)
- **Sorting**: Multi-key sort with external-memory spill
- **Reshaping**: `pack_columns`, `unpack_column`, `stack`

### SFrameStreamWriter

Streaming writer for building SFrames incrementally:

```rust
let mut writer = SFrameStreamWriter::new(
    "output.sf",
    vec!["id".into(), "value".into()],
    vec![FlexTypeEnum::Integer, FlexTypeEnum::Float],
)?;
writer.write_batch(vec![vec![FlexType::Integer(1)], vec![FlexType::Float(3.14)]])?;
writer.finish()?;
```

See [docs/api-reference.md](docs/api-reference.md) for the full API reference.
