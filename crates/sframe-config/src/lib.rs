//! Global configuration for the SFrame runtime.
//!
//! All engine settings live here. Values are initialized from environment
//! variables on first access. Cache capacity can be overridden at runtime;
//! all other settings are immutable after initialization.
//!
//! # Environment Variables
//!
//! All prefixed with `SFRAME_`:
//! - `SFRAME_CACHE_CAPACITY`: CacheFs in-memory store limit (default 2G)
//! - `SFRAME_CACHE_CAPACITY_PER_FILE`: Max single file in cache (default 2G)
//! - `SFRAME_SOURCE_BATCH_SIZE`: Rows per batch (default 4096)
//! - `SFRAME_ROWS_PER_SEGMENT`: Max rows per segment (default 1000000)
//! - `SFRAME_SORT_BUFFER_SIZE`: Sort memory budget (default 256M)
//! - `SFRAME_GROUPBY_BUFFER_NUM_ROWS`: Groupby hash table limit (default 1048576)
//! - `SFRAME_JOIN_BUFFER_NUM_CELLS`: Join hash table limit (default 50000000)
//! - `SFRAME_SOURCE_PREFETCH_SEGMENTS`: Lazy source prefetch (default 2)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_CACHE_CAPACITY: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
const DEFAULT_CACHE_CAPACITY_PER_FILE: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Global configuration for the SFrame engine.
///
/// Cache capacity fields use AtomicUsize for runtime mutation. All other
/// fields are immutable after initialization.
pub struct SFrameConfig {
    // --- Mutable at runtime ---
    cache_capacity: AtomicUsize,
    cache_capacity_per_file: AtomicUsize,

    // --- Immutable after init ---
    /// Batch size for source operators (rows per batch).
    pub source_batch_size: usize,
    /// Maximum rows per segment before auto-splitting on write.
    pub rows_per_segment: u64,
    /// Memory budget for in-memory sort. If estimated data size exceeds
    /// this, external sort is used.
    pub sort_memory_budget: usize,
    /// Maximum number of rows in a groupby hash table per segment before
    /// spilling to disk.
    pub groupby_buffer_num_rows: usize,
    /// Maximum number of cells (rows * columns) for the hash side of a
    /// join before GRACE partitioned join kicks in.
    pub join_buffer_num_cells: usize,
    /// Number of segments to prefetch for lazy source reading.
    pub source_prefetch_segments: usize,
}

impl SFrameConfig {
    /// Get the cache capacity (mutable at runtime).
    pub fn cache_capacity(&self) -> usize {
        self.cache_capacity.load(Ordering::Relaxed)
    }

    /// Set the cache capacity.
    pub fn set_cache_capacity(&self, bytes: usize) {
        self.cache_capacity.store(bytes, Ordering::Relaxed);
    }

    /// Get the per-file cache capacity (mutable at runtime).
    pub fn cache_capacity_per_file(&self) -> usize {
        self.cache_capacity_per_file.load(Ordering::Relaxed)
    }

    /// Set the per-file cache capacity.
    pub fn set_cache_capacity_per_file(&self, bytes: usize) {
        self.cache_capacity_per_file.store(bytes, Ordering::Relaxed);
    }
}

static GLOBAL_CONFIG: LazyLock<SFrameConfig> = LazyLock::new(|| {
    let mut cache_cap = DEFAULT_CACHE_CAPACITY;
    let mut cache_cap_per_file = DEFAULT_CACHE_CAPACITY_PER_FILE;
    let mut source_batch_size: usize = 4096;
    let mut rows_per_segment: u64 = 1_000_000;
    let mut sort_memory_budget: usize = 256 * 1024 * 1024;
    let mut groupby_buffer_num_rows: usize = 1_048_576;
    let mut join_buffer_num_cells: usize = 50_000_000;
    let mut source_prefetch_segments: usize = 2;

    if let Ok(val) = std::env::var("SFRAME_CACHE_CAPACITY") {
        if let Ok(n) = parse_byte_size(&val) {
            cache_cap = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_CACHE_CAPACITY_PER_FILE") {
        if let Ok(n) = parse_byte_size(&val) {
            cache_cap_per_file = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_SOURCE_BATCH_SIZE") {
        if let Ok(n) = val.parse::<usize>() {
            source_batch_size = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_ROWS_PER_SEGMENT") {
        if let Ok(n) = val.parse::<u64>() {
            rows_per_segment = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_SORT_BUFFER_SIZE") {
        if let Ok(n) = parse_byte_size(&val) {
            sort_memory_budget = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_GROUPBY_BUFFER_NUM_ROWS") {
        if let Ok(n) = val.parse::<usize>() {
            groupby_buffer_num_rows = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_JOIN_BUFFER_NUM_CELLS") {
        if let Ok(n) = val.parse::<usize>() {
            join_buffer_num_cells = n;
        }
    }
    if let Ok(val) = std::env::var("SFRAME_SOURCE_PREFETCH_SEGMENTS") {
        if let Ok(n) = val.parse::<usize>() {
            source_prefetch_segments = n;
        }
    }

    SFrameConfig {
        cache_capacity: AtomicUsize::new(cache_cap),
        cache_capacity_per_file: AtomicUsize::new(cache_cap_per_file),
        source_batch_size,
        rows_per_segment,
        sort_memory_budget,
        groupby_buffer_num_rows,
        join_buffer_num_cells,
        source_prefetch_segments,
    }
});

/// Return the global config, initialized from environment variables on first access.
pub fn global() -> &'static SFrameConfig {
    &GLOBAL_CONFIG
}

// ---------------------------------------------------------------------------
// Backward-compatible free functions
// ---------------------------------------------------------------------------

/// Get the maximum total bytes for the CacheFs in-memory store.
pub fn get_cache_capacity() -> usize {
    global().cache_capacity()
}

/// Set the maximum total bytes for the CacheFs in-memory store.
pub fn set_cache_capacity(bytes: usize) {
    global().set_cache_capacity(bytes);
}

/// Get the maximum size of a single file in the CacheFs in-memory store.
pub fn get_cache_capacity_per_file() -> usize {
    global().cache_capacity_per_file()
}

/// Set the maximum size of a single file in the CacheFs in-memory store.
pub fn set_cache_capacity_per_file(bytes: usize) {
    global().set_cache_capacity_per_file(bytes);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Parse a byte size string. Supports plain integers and suffixes:
/// `K`/`KB`, `M`/`MB`, `G`/`GB` (case-insensitive).
pub fn parse_byte_size(s: &str) -> Result<usize, ()> {
    let s = s.trim();
    let (num_str, multiplier) = if let Some(n) = s.strip_suffix("GB").or_else(|| s.strip_suffix("gb")).or_else(|| s.strip_suffix("G").or_else(|| s.strip_suffix("g"))) {
        (n.trim(), 1024 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("MB").or_else(|| s.strip_suffix("mb")).or_else(|| s.strip_suffix("M").or_else(|| s.strip_suffix("m"))) {
        (n.trim(), 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("KB").or_else(|| s.strip_suffix("kb")).or_else(|| s.strip_suffix("K").or_else(|| s.strip_suffix("k"))) {
        (n.trim(), 1024)
    } else {
        (s, 1)
    };
    num_str.parse::<usize>().map(|n| n * multiplier).map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_byte_size() {
        assert_eq!(parse_byte_size("1024"), Ok(1024));
        assert_eq!(parse_byte_size("10K"), Ok(10 * 1024));
        assert_eq!(parse_byte_size("10KB"), Ok(10 * 1024));
        assert_eq!(parse_byte_size("5M"), Ok(5 * 1024 * 1024));
        assert_eq!(parse_byte_size("5MB"), Ok(5 * 1024 * 1024));
        assert_eq!(parse_byte_size("2G"), Ok(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_byte_size("2GB"), Ok(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_byte_size(" 100 "), Ok(100));
        assert!(parse_byte_size("abc").is_err());
        assert!(parse_byte_size("").is_err());
    }

    #[test]
    fn test_defaults() {
        let config = global();
        assert!(config.cache_capacity() > 0);
        assert!(config.cache_capacity_per_file() > 0);
        assert_eq!(config.source_batch_size, 4096);
        assert_eq!(config.rows_per_segment, 1_000_000);
    }

    #[test]
    fn test_set_get_cache() {
        let config = global();
        let original = config.cache_capacity();
        config.set_cache_capacity(999);
        assert_eq!(config.cache_capacity(), 999);
        config.set_cache_capacity(original);
    }
}
