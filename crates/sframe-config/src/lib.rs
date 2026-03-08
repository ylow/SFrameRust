//! Global configuration for the SFrame runtime.
//!
//! All engine settings live here. Values are initialized from environment
//! variables on first access. All numeric settings can be overridden at
//! runtime; `cache_dir` is immutable after initialization.
//!
//! # Environment Variables
//!
//! All prefixed with `SFRAME_`:
//! - `SFRAME_CACHE_DIR`: On-disk cache directory (default `/var/tmp/sframe` on Linux,
//!   `temp_dir()/sframe` elsewhere). Files are written under a `{pid}/` subdirectory.
//! - `SFRAME_CACHE_CAPACITY`: CacheFs in-memory store limit (default 2G)
//! - `SFRAME_CACHE_CAPACITY_PER_FILE`: Max single file in cache (default 2G)
//! - `SFRAME_SOURCE_BATCH_SIZE`: Rows per batch (default 4096)
//! - `SFRAME_SORT_MAX_MEMORY`: Max total memory for external sort phase (default 4G)
//! - `SFRAME_GROUPBY_BUFFER_NUM_ROWS`: Groupby hash table limit (default 1048576)
//! - `SFRAME_JOIN_BUFFER_NUM_CELLS`: Join hash table limit (default 50000000)
//! - `SFRAME_SOURCE_PREFETCH_SEGMENTS`: Lazy source prefetch (default 2)
//! - `SFRAME_MAX_BLOCKS_IN_CACHE`: Max blocks per CachedSegmentReader (default 64)

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_CACHE_CAPACITY: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
const DEFAULT_CACHE_CAPACITY_PER_FILE: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
const DEFAULT_SOURCE_BATCH_SIZE: usize = 4096;
const DEFAULT_SORT_MAX_MEMORY: usize = 4 * 1024 * 1024 * 1024; // 4 GiB
const DEFAULT_GROUPBY_BUFFER_NUM_ROWS: usize = 1_048_576;
const DEFAULT_JOIN_BUFFER_NUM_CELLS: usize = 50_000_000;
const DEFAULT_SOURCE_PREFETCH_SEGMENTS: usize = 2;
const DEFAULT_MAX_BLOCKS_IN_CACHE: usize = 64;

/// Global configuration for the SFrame engine.
///
/// All numeric fields use `AtomicUsize` for safe runtime mutation.
/// `cache_dir` is immutable after initialization.
pub struct SFrameConfig {
    cache_capacity: AtomicUsize,
    cache_capacity_per_file: AtomicUsize,
    source_batch_size: AtomicUsize,
    sort_max_memory: AtomicUsize,
    groupby_buffer_num_rows: AtomicUsize,
    join_buffer_num_cells: AtomicUsize,
    source_prefetch_segments: AtomicUsize,
    max_blocks_in_cache: AtomicUsize,

    /// Root directory for on-disk cache storage.
    /// Within this directory, files are written under a `{pid}/` subdirectory.
    /// Immutable after initialization.
    pub cache_dir: PathBuf,
}

impl SFrameConfig {
    /// Get the cache capacity.
    pub fn cache_capacity(&self) -> usize {
        self.cache_capacity.load(Ordering::Relaxed)
    }

    /// Set the cache capacity.
    pub fn set_cache_capacity(&self, bytes: usize) {
        self.cache_capacity.store(bytes, Ordering::Relaxed);
    }

    /// Get the per-file cache capacity.
    pub fn cache_capacity_per_file(&self) -> usize {
        self.cache_capacity_per_file.load(Ordering::Relaxed)
    }

    /// Set the per-file cache capacity.
    pub fn set_cache_capacity_per_file(&self, bytes: usize) {
        self.cache_capacity_per_file.store(bytes, Ordering::Relaxed);
    }

    /// Get the source batch size (rows per batch).
    pub fn source_batch_size(&self) -> usize {
        self.source_batch_size.load(Ordering::Relaxed)
    }

    /// Set the source batch size.
    pub fn set_source_batch_size(&self, rows: usize) {
        self.source_batch_size.store(rows, Ordering::Relaxed);
    }

    /// Get the maximum memory for external sort.
    pub fn sort_max_memory(&self) -> usize {
        self.sort_max_memory.load(Ordering::Relaxed)
    }

    /// Set the maximum memory for external sort.
    pub fn set_sort_max_memory(&self, bytes: usize) {
        self.sort_max_memory.store(bytes, Ordering::Relaxed);
    }

    /// Get the groupby buffer row limit.
    pub fn groupby_buffer_num_rows(&self) -> usize {
        self.groupby_buffer_num_rows.load(Ordering::Relaxed)
    }

    /// Set the groupby buffer row limit.
    pub fn set_groupby_buffer_num_rows(&self, rows: usize) {
        self.groupby_buffer_num_rows.store(rows, Ordering::Relaxed);
    }

    /// Get the join buffer cell limit.
    pub fn join_buffer_num_cells(&self) -> usize {
        self.join_buffer_num_cells.load(Ordering::Relaxed)
    }

    /// Set the join buffer cell limit.
    pub fn set_join_buffer_num_cells(&self, cells: usize) {
        self.join_buffer_num_cells.store(cells, Ordering::Relaxed);
    }

    /// Get the source prefetch segment count.
    pub fn source_prefetch_segments(&self) -> usize {
        self.source_prefetch_segments.load(Ordering::Relaxed)
    }

    /// Set the source prefetch segment count.
    pub fn set_source_prefetch_segments(&self, segments: usize) {
        self.source_prefetch_segments.store(segments, Ordering::Relaxed);
    }

    /// Get the max blocks in cache per CachedSegmentReader.
    pub fn max_blocks_in_cache(&self) -> usize {
        self.max_blocks_in_cache.load(Ordering::Relaxed)
    }

    /// Set the max blocks in cache per CachedSegmentReader.
    pub fn set_max_blocks_in_cache(&self, blocks: usize) {
        self.max_blocks_in_cache.store(blocks, Ordering::Relaxed);
    }

    /// Reset all settings to their defaults (or environment variable overrides).
    pub fn reset(&self) {
        // Re-read from environment, falling back to compiled defaults.
        self.set_cache_capacity(read_byte_env("SFRAME_CACHE_CAPACITY", DEFAULT_CACHE_CAPACITY));
        self.set_cache_capacity_per_file(read_byte_env(
            "SFRAME_CACHE_CAPACITY_PER_FILE",
            DEFAULT_CACHE_CAPACITY_PER_FILE,
        ));
        self.set_source_batch_size(read_usize_env(
            "SFRAME_SOURCE_BATCH_SIZE",
            DEFAULT_SOURCE_BATCH_SIZE,
        ));
        self.set_sort_max_memory(read_byte_env(
            "SFRAME_SORT_MAX_MEMORY",
            DEFAULT_SORT_MAX_MEMORY,
        ));
        self.set_groupby_buffer_num_rows(read_usize_env(
            "SFRAME_GROUPBY_BUFFER_NUM_ROWS",
            DEFAULT_GROUPBY_BUFFER_NUM_ROWS,
        ));
        self.set_join_buffer_num_cells(read_usize_env(
            "SFRAME_JOIN_BUFFER_NUM_CELLS",
            DEFAULT_JOIN_BUFFER_NUM_CELLS,
        ));
        self.set_source_prefetch_segments(read_usize_env(
            "SFRAME_SOURCE_PREFETCH_SEGMENTS",
            DEFAULT_SOURCE_PREFETCH_SEGMENTS,
        ));
        self.set_max_blocks_in_cache(read_usize_env(
            "SFRAME_MAX_BLOCKS_IN_CACHE",
            DEFAULT_MAX_BLOCKS_IN_CACHE,
        ));
    }
}

fn read_usize_env(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn read_byte_env(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| parse_byte_size(&v).ok())
        .unwrap_or(default)
}

static GLOBAL_CONFIG: LazyLock<SFrameConfig> = LazyLock::new(|| {
    let cache_dir = match std::env::var("SFRAME_CACHE_DIR") {
        Ok(val) => PathBuf::from(val),
        Err(_) => default_cache_dir(),
    };

    let config = SFrameConfig {
        cache_capacity: AtomicUsize::new(DEFAULT_CACHE_CAPACITY),
        cache_capacity_per_file: AtomicUsize::new(DEFAULT_CACHE_CAPACITY_PER_FILE),
        source_batch_size: AtomicUsize::new(DEFAULT_SOURCE_BATCH_SIZE),
        sort_max_memory: AtomicUsize::new(DEFAULT_SORT_MAX_MEMORY),
        groupby_buffer_num_rows: AtomicUsize::new(DEFAULT_GROUPBY_BUFFER_NUM_ROWS),
        join_buffer_num_cells: AtomicUsize::new(DEFAULT_JOIN_BUFFER_NUM_CELLS),
        source_prefetch_segments: AtomicUsize::new(DEFAULT_SOURCE_PREFETCH_SEGMENTS),
        max_blocks_in_cache: AtomicUsize::new(DEFAULT_MAX_BLOCKS_IN_CACHE),
        cache_dir,
    };
    // Apply environment variable overrides.
    config.reset();
    config
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

/// Default cache directory: `/var/tmp/sframe` on Linux, `temp_dir()/sframe` elsewhere.
fn default_cache_dir() -> PathBuf {
    if cfg!(target_os = "linux") {
        PathBuf::from("/var/tmp/sframe")
    } else {
        std::env::temp_dir().join("sframe")
    }
}

/// Parse a byte size string. Supports plain integers and suffixes:
/// `K`/`KB`, `M`/`MB`, `G`/`GB` (case-insensitive).
// Changing to a richer error type would break the public API for minimal benefit.
#[allow(clippy::result_unit_err)]
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
        assert_eq!(config.source_batch_size(), 4096);
    }

    #[test]
    fn test_set_get_cache() {
        let config = global();
        let original = config.cache_capacity();
        config.set_cache_capacity(999);
        assert_eq!(config.cache_capacity(), 999);
        config.set_cache_capacity(original);
    }

    #[test]
    fn test_set_get_all_fields() {
        let config = global();

        let orig_batch = config.source_batch_size();
        config.set_source_batch_size(8192);
        assert_eq!(config.source_batch_size(), 8192);
        config.set_source_batch_size(orig_batch);

        let orig_sort = config.sort_max_memory();
        config.set_sort_max_memory(1024);
        assert_eq!(config.sort_max_memory(), 1024);
        config.set_sort_max_memory(orig_sort);

        let orig_groupby = config.groupby_buffer_num_rows();
        config.set_groupby_buffer_num_rows(500);
        assert_eq!(config.groupby_buffer_num_rows(), 500);
        config.set_groupby_buffer_num_rows(orig_groupby);

        let orig_join = config.join_buffer_num_cells();
        config.set_join_buffer_num_cells(100);
        assert_eq!(config.join_buffer_num_cells(), 100);
        config.set_join_buffer_num_cells(orig_join);

        let orig_prefetch = config.source_prefetch_segments();
        config.set_source_prefetch_segments(4);
        assert_eq!(config.source_prefetch_segments(), 4);
        config.set_source_prefetch_segments(orig_prefetch);

        let orig_blocks = config.max_blocks_in_cache();
        config.set_max_blocks_in_cache(128);
        assert_eq!(config.max_blocks_in_cache(), 128);
        config.set_max_blocks_in_cache(orig_blocks);
    }

    #[test]
    fn test_reset() {
        let config = global();
        config.set_source_batch_size(9999);
        assert_eq!(config.source_batch_size(), 9999);
        config.reset();
        // After reset, should be back to default (or env override).
        assert_eq!(config.source_batch_size(), DEFAULT_SOURCE_BATCH_SIZE);
    }
}
