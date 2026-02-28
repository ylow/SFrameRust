//! Centralized configuration constants for the SFrame query engine.
//!
//! Analogous to C++ `sframe_constants.hpp`. Controls memory budgets,
//! batch sizes, and thresholds for out-of-core algorithms.
//!
//! Configuration values are initialized from environment variables on
//! first access via `SFrameConfig::global()`:
//!
//! - `SFRAME_SOURCE_BATCH_SIZE`: rows per batch (default 4096)
//! - `SFRAME_SORT_BUFFER_SIZE`: memory budget for sort (default 256MB)
//! - `SFRAME_GROUPBY_BUFFER_NUM_ROWS`: groupby hash table limit (default 1M)
//! - `SFRAME_JOIN_BUFFER_NUM_CELLS`: join hash table limit (default 50M)
//! - `SFRAME_SOURCE_PREFETCH_SEGMENTS`: lazy source prefetch (default 2)

use std::sync::LazyLock;

/// Configuration for the SFrame query engine.
///
/// All sizes are in bytes unless otherwise noted. Use [`SFrameConfig::default()`]
/// for reasonable defaults tuned to datasets up to ~100M rows.
#[derive(Debug, Clone)]
pub struct SFrameConfig {
    /// Batch size for source operators (rows per batch).
    pub source_batch_size: usize,

    /// Maximum rows per segment before auto-splitting on write.
    pub rows_per_segment: u64,

    /// Memory budget for in-memory sort. If the estimated data size
    /// exceeds this, external sort is used instead.
    pub sort_memory_budget: usize,

    /// Maximum number of rows in a groupby hash table per output segment
    /// before spilling to disk.
    pub groupby_buffer_num_rows: usize,

    /// Maximum number of cells (rows × columns) for the hash side of a
    /// join before GRACE partitioned join kicks in.
    pub join_buffer_num_cells: usize,

    /// Number of segments to prefetch for lazy source reading.
    pub source_prefetch_segments: usize,
}

impl Default for SFrameConfig {
    fn default() -> Self {
        SFrameConfig {
            source_batch_size: 4096,
            rows_per_segment: 1_000_000,
            sort_memory_budget: 256 * 1024 * 1024, // 256 MB
            groupby_buffer_num_rows: 1_048_576,     // 1M rows
            join_buffer_num_cells: 50_000_000,      // 50M cells
            source_prefetch_segments: 2,
        }
    }
}

static GLOBAL_CONFIG: LazyLock<SFrameConfig> = LazyLock::new(|| {
    let mut config = SFrameConfig::default();
    if let Ok(val) = std::env::var("SFRAME_SOURCE_BATCH_SIZE") {
        if let Ok(n) = val.parse::<usize>() {
            config.source_batch_size = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_SORT_BUFFER_SIZE") {
        if let Ok(n) = sframe_config::parse_byte_size(&val) {
            config.sort_memory_budget = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_GROUPBY_BUFFER_NUM_ROWS") {
        if let Ok(n) = val.parse::<usize>() {
            config.groupby_buffer_num_rows = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_JOIN_BUFFER_NUM_CELLS") {
        if let Ok(n) = val.parse::<usize>() {
            config.join_buffer_num_cells = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_SOURCE_PREFETCH_SEGMENTS") {
        if let Ok(n) = val.parse::<usize>() {
            config.source_prefetch_segments = n;
        }
    }
    config
});

impl SFrameConfig {
    /// Return the global config, initialized from environment variables on first access.
    pub fn global() -> &'static SFrameConfig {
        &GLOBAL_CONFIG
    }
}
