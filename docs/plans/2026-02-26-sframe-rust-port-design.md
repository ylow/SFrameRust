# SFrame Rust Port — Design Document

**Date:** 2026-02-26
**Status:** Approved
**Approach:** Bottom-up, layer by layer

## Background

SFrame is a C++ out-of-core column store system originally built for Python (via
Cython/CPPIPC bindings). This project ports the core system to Rust, deprecating
the Python binding layer (Unity, gl_sframe, CPPIPC) and focusing on:

- Core dataframe disk representation (V2 format, backward compatible)
- Logical and physical query execution (async streams replacing boost coroutines)
- Virtual filesystem abstraction (local-only initially, extensible to S3/HTTP/etc.)
- Algorithms: sort, join, groupby, CSV parsing

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target API | Rust-native first | Python bindings (PyO3) can come later as a separate crate |
| Disk format | Compatible with C++ V2 | Existing .sframe files load directly, no migration tooling |
| Type system | Core 7 (no IMAGE) | INTEGER, FLOAT, STRING, VECTOR, LIST, DICT, DATETIME, UNDEFINED |
| Filesystem | Local only, well-abstracted | VFS trait enables S3/HTTP backends later |
| Execution model | Async streams | Pull-based Stream<Item=Result<SFrameRows>>, replaces boost coroutines |
| Memory format | Custom SFrameRows | Typed column vectors, no Arrow dependency |
| Operators | Full set minus Python lambdas | source, transform, generalized_transform, filter, project, append, reduce, binary_transform, union, range |
| Query optimization | Deferred | Planner node structure supports future optimization passes |
| Crate layout | Cargo workspace | 5 crates with enforced dependency direction |

---

## Workspace Layout

```
sframe-rust/
├── Cargo.toml              (workspace root)
├── crates/
│   ├── sframe-types/       FlexibleType, type system, serialization primitives
│   ├── sframe-io/          Virtual filesystem trait + local backend + CacheFs
│   ├── sframe-storage/     V2 disk format: block manager, segment reader/writer
│   ├── sframe-query/       Operators, planner, async execution, algorithms
│   └── sframe/             Top-level SFrame/SArray API, pretty printing
├── samples/
│   ├── business.csv
│   └── business.sf/
└── docs/plans/
```

**Dependency graph** (arrows = "depends on"):
```
sframe → sframe-query → sframe-storage → sframe-io
                                        → sframe-types
                       → sframe-types
         sframe-types
         sframe-io
```

**Key external dependencies:**
- `lz4_flex` — block compression (V2 format compat)
- `serde` + `serde_json` — .sidx JSON parsing
- `tokio` — async runtime
- `futures` — Stream trait
- `bytes` — buffer management
- `thiserror` — error types

---

## Crate 1: sframe-types

Foundation crate. No dependencies on other sframe crates.

### FlexType

Tagged enum mirroring the C++ `flexible_type`:

```rust
pub enum FlexType {
    Integer(i64),
    Float(f64),
    String(Arc<str>),
    Vector(Arc<[f64]>),
    List(Arc<[FlexType]>),
    Dict(Arc<[(FlexType, FlexType)]>),
    DateTime(FlexDateTime),
    Undefined,
}
```

`Arc<str>` / `Arc<[T]>` for immutable shared ownership — values are frequently
shared across operators and column slices without cloning backing data.

### FlexDateTime

```rust
pub struct FlexDateTime {
    pub posix_timestamp: i64,
    pub tz_offset_quarter_hours: i8,
    pub microsecond: u32,
}
```

### FlexTypeEnum

Type tag enum matching C++ values for format compatibility:

```rust
#[repr(u8)]
pub enum FlexTypeEnum {
    Integer = 0,
    Float = 1,
    String = 2,
    Vector = 3,
    List = 4,
    Dict = 5,
    DateTime = 6,
    Undefined = 7,
}
```

### Type Conversions

`From`/`TryFrom` between compatible types matching the C++ conversion matrix:
Integer <-> Float <-> String, etc. `PartialOrd`/`Ord` for sort comparators.

### Serialization Primitives

Compatible with GraphLab's `oarchive`/`iarchive`:

```rust
pub trait Deserialize: Sized {
    fn deserialize(reader: &mut impl Read) -> Result<Self>;
}
pub trait Serialize {
    fn serialize(&self, writer: &mut impl Write) -> Result<()>;
}
```

Little-endian throughout. Covers:
- Fixed-width integers (u8, u16, u32, u64, i64)
- Variable-length integer encoding (1-9 byte scheme, distinct from LEB128)
- FlexType recursive serialization (for List, Dict, DateTime)
- STL-compatible container serialization (Vec length prefix + elements)

### Variable-Length Integer Encoding

The V2 format uses a specific scheme:
- 1 byte (7 bits): `[value << 1 | 0]`
- 2 bytes (14 bits): `[value << 2 | 1]` (LE)
- 3 bytes (21 bits): `[value << 3 | 3]`
- 4 bytes (28 bits): `[value << 4 | 7]`
- 5 bytes (35 bits): `[value << 5 | 15]`
- 6 bytes (42 bits): `[value << 6 | 31]`
- 7 bytes (49 bits): `[value << 7 | 63]`
- 9 bytes (64 bits): `[0x7F, 8 bytes of value]`

---

## Crate 2: sframe-io

Virtual filesystem abstraction. Depends on nothing from the workspace.

### VirtualFileSystem Trait

```rust
#[async_trait]
pub trait VirtualFileSystem: Send + Sync {
    async fn open_read(&self, path: &str) -> Result<Box<dyn ReadableFile>>;
    async fn open_write(&self, path: &str) -> Result<Box<dyn WritableFile>>;
    async fn stat(&self, path: &str) -> Result<FileStat>;
    async fn list_dir(&self, path: &str) -> Result<Vec<DirEntry>>;
    async fn remove(&self, path: &str) -> Result<()>;
    async fn mkdir_p(&self, path: &str) -> Result<()>;
}

pub trait ReadableFile: Send {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize>;
    fn seek(&mut self, pos: u64) -> Result<()>;
    fn size(&self) -> Result<u64>;
}

pub trait WritableFile: Send {
    fn write(&mut self, buf: &[u8]) -> Result<usize>;
    fn flush(&mut self) -> Result<()>;
}
```

Async boundary at `open_read`/`open_write` (where network latency matters for
future S3 backend). File handle operations are synchronous — for S3 the
ReadableFile impl would wrap a buffered/prefetched byte stream internally.

### FileSystemRegistry

Routes URL prefixes to backends:

```rust
pub struct FileSystemRegistry {
    backends: Vec<(String, Arc<dyn VirtualFileSystem>)>,
}

impl FileSystemRegistry {
    pub fn resolve(&self, url: &str) -> Result<(&dyn VirtualFileSystem, &str)>;
}
```

Default setup registers `LocalFileSystem` for plain paths / `file://` URLs, and
`CacheFs` for `cache://` URLs.

### LocalFileSystem

Straightforward `std::fs` wrapper implementing `VirtualFileSystem`.

### CacheFs — Reference-Counted Ephemeral Storage

`cache://` is a first-class VFS backend for ephemeral materialized data. It
implements `VirtualFileSystem` like any other backend (uniform read/write paths,
.sidx/.frame_idx files use string paths). Separately, it tracks reference counts
for file lifetime management:

```rust
pub struct CacheFs {
    root: PathBuf,
    counter: AtomicU64,
    refcounts: Mutex<HashMap<PathBuf, usize>>,
}

impl CacheFs {
    pub fn retain(&self, path: &str);
    pub fn release(&self, path: &str);  // deletes file when refcount hits 0
}
```

SArray/SFrame objects hold RAII guards for cache-backed files:

```rust
pub struct CacheGuard {
    path: String,
    fs: Arc<CacheFs>,
}
impl Clone for CacheGuard { /* calls fs.retain() */ }
impl Drop for CacheGuard { /* calls fs.release() */ }
```

Design rationale: the VFS remains uniform — all code that reads/writes SFrames
uses string paths through the VFS. The refcounting is an orthogonal concern
layered on top. .sidx and .frame_idx files contain plain string paths regardless
of whether they live in cache:// or on real filesystems.

---

## Crate 3: sframe-storage

V2 disk format reader/writer. Depends on `sframe-types` and `sframe-io`.

### Archive Structure

An SFrame on disk is a dir_archive:

```
my_frame.sf/
├── dir_archive.ini            INI: archive metadata
├── objects.bin                 Empty (legacy)
├── m_<hash>.frame_idx         INI: column names, row count, column->sidx mapping
├── m_<hash>.sidx              JSON: column types, segment sizes, segment file refs
├── m_<hash>.0000              Binary: segment 0 (blocks for all columns)
├── m_<hash>.0001              Binary: segment 1 (if multi-segment)
└── ...
```

### Index Structs

```rust
/// Parsed from .frame_idx (INI format)
pub struct FrameIndex {
    pub version: u32,
    pub num_columns: usize,
    pub nrows: u64,
    pub column_names: Vec<String>,
    pub column_files: Vec<String>,  // "m_<hash>.sidx:0", "m_<hash>.sidx:1", ...
}

/// Parsed from .sidx (JSON format)
pub struct GroupIndex {
    pub version: u32,
    pub nsegments: usize,
    pub segment_files: Vec<String>,
    pub columns: Vec<ColumnIndex>,
}

pub struct ColumnIndex {
    pub dtype: FlexTypeEnum,
    pub segment_sizes: Vec<u64>,
    pub content_type: String,
}
```

### Segment File Layout

```
[Block 0, padded to 4K] [Block 1, padded to 4K] ... [Block N] [BlockInfo array] [footer_size: u64]
```

Reading:
1. Seek to last 8 bytes -> `footer_size` (LE u64)
2. Seek to `file_size - footer_size - 8` -> deserialize `Vec<Vec<BlockInfo>>`
3. Each BlockInfo gives offset, compressed/decompressed sizes, element count, flags

```rust
pub struct BlockInfo {
    pub offset: u64,
    pub length: u64,       // on-disk (compressed) size
    pub block_size: u64,   // decompressed size
    pub num_elem: u64,
    pub flags: u64,        // LZ4_COMPRESSION=1, IS_FLEXIBLE_TYPE=2, MULTIPLE_TYPE=4, ENCODING_EXT=8
}
```

### Block Encoding/Decoding

After LZ4 decompression (if flagged), typed blocks are decoded:

**Header:** byte 0 = num_types (0=empty, 1=homogeneous, 2=has undefineds, >2=mixed).
Byte 1 = FlexTypeEnum if num_types is 1 or 2.

**Type-specific codecs:**

| Type | Encoding |
|------|----------|
| Integer | Frame-of-reference in groups of 128. Header: codec (plain/delta/delta-neg) + bit width. Variable-length min value. Packed differences at 0/1/2/4/8/16/32/64 bits. |
| Float | Reserved byte selects: legacy (bit-rotated doubles -> FoR) or integer encoding (whole-number floats -> FoR) |
| String | Dictionary (<=64 unique): dict + FoR indices. Direct (>64 unique): FoR lengths + concatenated bytes. |
| Vector | FoR-encoded lengths + flattened doubles via float encoding |
| List/Dict/DateTime | GraphLab archive serialization (byte-for-byte compatible) |

**Undefined bitmap:** When num_types=2, a dense bitset precedes the typed data
indicating which elements are UNDEFINED.

### Segment Reader

```rust
pub struct SegmentReader {
    file: Box<dyn ReadableFile>,
    block_index: Vec<Vec<BlockInfo>>,  // [column][block]
}

impl SegmentReader {
    pub fn read_block(&mut self, column: usize, block: usize) -> Result<Vec<u8>>;
    pub fn read_column(&mut self, column: usize) -> Result<Vec<FlexType>>;
}
```

### Segment Writer (Streaming)

```rust
pub struct SegmentWriter {
    file: Box<dyn WritableFile>,
    columns: Vec<ColumnEncoder>,
    block_infos: Vec<Vec<BlockInfo>>,
    rows_written: u64,
}

impl SegmentWriter {
    pub fn new(file: Box<dyn WritableFile>, column_types: &[FlexTypeEnum]) -> Self;
    pub fn write_batch(&mut self, batch: &SFrameRows) -> Result<()>;
    pub fn finish(self) -> Result<SegmentMetadata>;
}
```

### SFrame Writer (Multi-Segment Orchestrator)

```rust
pub struct SFrameWriter {
    fs: Arc<dyn VirtualFileSystem>,
    path: String,
    column_names: Vec<String>,
    column_types: Vec<FlexTypeEnum>,
}

impl SFrameWriter {
    pub fn new(fs: Arc<dyn VirtualFileSystem>, path: &str, columns: &[(&str, FlexTypeEnum)]) -> Result<Self>;
    pub fn new_segment(&mut self) -> Result<SegmentWriter>;
    pub fn finish(self, segments: Vec<SegmentMetadata>) -> Result<()>;
}
```

A query plan with N parallel partitions creates N SegmentWriters. Each writes
independently. The final `finish()` assembles .sidx, .frame_idx, and
dir_archive.ini from the collected SegmentMetadata.

---

## Crate 4: sframe-query

Largest crate. Logical planner, async execution engine, and algorithms.
Depends on `sframe-types`, `sframe-io`, `sframe-storage`.

### Batch Representation

```rust
pub struct SFrameRows {
    columns: Vec<ColumnData>,
    num_rows: usize,
}

pub enum ColumnData {
    Integer(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
    String(Vec<Option<Arc<str>>>),
    Vector(Vec<Option<Arc<[f64]>>>),
    List(Vec<Option<Arc<[FlexType]>>>),
    Dict(Vec<Option<Arc<[(FlexType, FlexType)]>>>),
    DateTime(Vec<Option<FlexDateTime>>),
}
```

Typed column vectors avoid per-value tag overhead. `Option` wraps handle
UNDEFINED/NULL.

### Logical Planner

DAG of PlannerNodes:

```rust
pub struct PlannerNode {
    pub op: LogicalOp,
    pub inputs: Vec<Arc<PlannerNode>>,
}

pub enum LogicalOp {
    SArraySource { .. },
    SFrameSource { .. },
    Project { column_indices: Vec<usize> },
    Filter { predicate: Arc<dyn Fn(&FlexType) -> bool + Send + Sync> },
    Transform { func: .., output_type: FlexTypeEnum },
    BinaryTransform { func: .., output_type: FlexTypeEnum },
    GeneralizedTransform { func: .., output_types: Vec<FlexTypeEnum> },
    Append,
    Range { start: i64, step: i64, count: u64 },
    Reduce { aggregator: Arc<dyn Aggregator> },
    Union,
}
```

Nodes are `Arc`-shared (DAG, not tree) — same subexpression can appear in
multiple places.

Optimization passes are deferred. The compile path goes directly from logical
DAG to async streams.

### Physical Execution

Each operator compiles to an async stream:

```rust
pub type BatchStream = Pin<Box<dyn Stream<Item = Result<SFrameRows>> + Send>>;

pub fn compile(node: &PlannerNode) -> Result<BatchStream>;
```

Operators compose by wrapping input streams. The tokio runtime handles
scheduling — Rust `.await` replaces the C++ coroutine context switches.

### Algorithms

Sub-modules within `sframe-query/src/algorithms/`:

**Sort** — External columnar sort (EC sort):
1. Quantile sketch on sort keys -> partition pivots
2. Scatter rows into buckets via forward map
3. In-memory sort per bucket
4. Merge sorted buckets -> output via SegmentWriters

**Join** — GRACE hash join:
1. Hash-partition both inputs into N partitions (shuffle)
2. Per partition: build hash table on right, probe with left
3. Emit matches with NULL padding for outer joins
4. Supports INNER, LEFT, RIGHT, FULL

**Groupby** — Hash-based parallel aggregation:
1. Hash group keys into aggregator map
2. Spill to disk if map exceeds memory budget
3. Merge partial results
4. Pluggable aggregators via Aggregator trait

**CSV Parser** — Parallel tokenizer with type inference:
1. Split file at line boundaries into chunks
2. Each chunk parsed independently
3. Type inference with configurable hints
4. Output assembled into SFrame

### Aggregator Trait

```rust
pub trait Aggregator: Send + Sync {
    fn add(&mut self, values: &[&FlexType]);
    fn merge(&mut self, other: &dyn Aggregator);
    fn finalize(&mut self) -> FlexType;
    fn output_type(&self, input_types: &[FlexTypeEnum]) -> FlexTypeEnum;
}
```

Built-in: Sum, Count, Mean, Min, Max, Variance, StdDev, ArgMin, ArgMax, Concat.

### Not Porting

- `lambda_transform` (Python-specific)
- Distributed RPC layer
- Fiber system (replaced by tokio async)

---

## Crate 5: sframe (top-level)

User-facing API. Depends on all other crates.

### Core Structs

```rust
pub struct SFrame {
    columns: Vec<SArray>,
    column_names: Vec<String>,
}

pub struct SArray {
    plan: Option<Arc<PlannerNode>>,     // lazy query plan (if unevaluated)
    storage: Option<SArrayStorage>,     // materialized backing (if any)
    dtype: FlexTypeEnum,
    len: u64,
}
```

Lazy by default. Operations build PlannerNode DAGs. Materialization happens on
`.head()`, `.iter_rows()`, `.save()`, `.materialize()`, or `Display`.

### API Surface

```rust
impl SFrame {
    // Construction
    pub fn read(fs: &FileSystemRegistry, path: &str) -> Result<Self>;
    pub fn from_csv(fs: &FileSystemRegistry, path: &str, opts: CsvOptions) -> Result<Self>;
    pub fn from_columns(columns: Vec<(&str, SArray)>) -> Result<Self>;

    // Column access
    pub fn column(&self, name: &str) -> Result<&SArray>;
    pub fn select(&self, names: &[&str]) -> Result<SFrame>;
    pub fn add_column(&self, name: &str, col: SArray) -> Result<SFrame>;
    pub fn remove_column(&self, name: &str) -> Result<SFrame>;

    // Row operations
    pub fn filter(&self, column: &str, pred: impl Fn(&FlexType) -> bool) -> SFrame;
    pub fn append(&self, other: &SFrame) -> Result<SFrame>;
    pub fn head(&self, n: usize) -> Result<SFrame>;
    pub fn sort(&self, keys: &[(&str, SortOrder)]) -> Result<SFrame>;
    pub fn join(&self, other: &SFrame, on: JoinOn, how: JoinType) -> Result<SFrame>;
    pub fn groupby(&self, keys: &[&str], aggs: Vec<AggSpec>) -> Result<SFrame>;

    // Materialization
    pub fn materialize(&self) -> Result<SFrame>;
    pub fn save(&self, fs: &FileSystemRegistry, path: &str) -> Result<()>;
    pub fn iter_rows(&self) -> Result<impl Iterator<Item = Vec<FlexType>>>;

    // Info
    pub fn num_rows(&self) -> u64;
    pub fn num_columns(&self) -> usize;
    pub fn column_names(&self) -> &[String];
    pub fn column_types(&self) -> Vec<FlexTypeEnum>;
    pub fn schema(&self) -> Vec<(String, FlexTypeEnum)>;
}
```

### Pretty Printing

`Display` impl materializes first/last few rows and formats as ASCII table:

```
+-------------+------------+-------+
| business_id | city       | stars |
+-------------+------------+-------+
| 0FNFSzC...  | Phoenix    | 5.0   |
| ...         | ...        | ...   |
+-------------+------------+-------+
[11536 rows x 12 columns]
```

### Runtime

The top-level crate owns the tokio runtime. Public API is synchronous —
`materialize()`/`save()`/`iter_rows()` internally block on async execution:

```rust
impl SFrame {
    pub fn materialize(&self) -> Result<SFrame> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(self.materialize_async())
    }
}
```

---

## Implementation Order

Bottom-up, layer by layer:

1. **sframe-types** — FlexType, serialization, variable-length encoding
2. **sframe-io** — VFS trait, LocalFileSystem, CacheFs
3. **sframe-storage** — V2 reader (block decoding, segment reader, index parsing), then V2 writer
4. **sframe-query** — SFrameRows, operators, planner, async execution, then algorithms (sort, join, groupby, CSV)
5. **sframe** — SFrame/SArray API, lazy evaluation, pretty printing

Each layer is testable in isolation. The sample `business.sf/` serves as the
primary integration test for the storage layer — read the file and verify against
`business.csv`.
