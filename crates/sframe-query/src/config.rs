//! Centralized configuration constants for the SFrame query engine.
//!
//! Analogous to C++ `sframe_constants.hpp`. Controls memory budgets,
//! batch sizes, and thresholds for out-of-core algorithms.

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
    /// exceeds this, EC-Sort (external columnar sort) is used instead.
    pub sort_memory_budget: usize,

    /// Maximum number of rows in a groupby hash table per output segment
    /// before spilling to disk.
    pub groupby_buffer_num_rows: usize,

    /// Maximum number of cells (rows × columns) for the hash side of a
    /// join before GRACE partitioned join kicks in.
    pub join_buffer_num_cells: usize,
}

impl Default for SFrameConfig {
    fn default() -> Self {
        SFrameConfig {
            source_batch_size: 4096,
            rows_per_segment: 1_000_000,
            sort_memory_budget: 256 * 1024 * 1024, // 256 MB
            groupby_buffer_num_rows: 1_048_576,     // 1M rows
            join_buffer_num_cells: 50_000_000,      // 50M cells
        }
    }
}

impl SFrameConfig {
    /// Return the global default config.
    pub fn global() -> &'static SFrameConfig {
        &DEFAULT_CONFIG
    }
}

static DEFAULT_CONFIG: SFrameConfig = SFrameConfig {
    source_batch_size: 4096,
    rows_per_segment: 1_000_000,
    sort_memory_budget: 256 * 1024 * 1024,
    groupby_buffer_num_rows: 1_048_576,
    join_buffer_num_cells: 50_000_000,
};
