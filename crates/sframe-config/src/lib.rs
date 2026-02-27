//! Global configuration for the SFrame runtime.
//!
//! Provides runtime-configurable globals analogous to the C++ SFrame's
//! `globals/` system. Values are initialized from environment variables
//! on first access and can be overridden at runtime via setter functions.
//!
//! # Cache configuration
//!
//! - `SFRAME_CACHE_CAPACITY`: Maximum total bytes for the CacheFs in-memory
//!   store. Files that fit within this budget are kept in RAM rather than
//!   spilling to disk. Default: 2 GiB.
//!
//! - `SFRAME_CACHE_CAPACITY_PER_FILE`: Maximum size of a single file that
//!   can be stored in the CacheFs in-memory store. Files larger than this
//!   always go to disk. Default: 128 MiB.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_CACHE_CAPACITY: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
const DEFAULT_CACHE_CAPACITY_PER_FILE: usize = 128 * 1024 * 1024; // 128 MiB

// ---------------------------------------------------------------------------
// Atomic globals
// ---------------------------------------------------------------------------

static FILEIO_MAXIMUM_CACHE_CAPACITY: AtomicUsize =
    AtomicUsize::new(DEFAULT_CACHE_CAPACITY);
static FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE: AtomicUsize =
    AtomicUsize::new(DEFAULT_CACHE_CAPACITY_PER_FILE);

static INIT: Once = Once::new();

/// Ensure environment variable overrides are applied (idempotent).
fn ensure_init() {
    INIT.call_once(|| {
        if let Ok(val) = std::env::var("SFRAME_CACHE_CAPACITY") {
            if let Ok(n) = parse_byte_size(&val) {
                FILEIO_MAXIMUM_CACHE_CAPACITY.store(n, Ordering::Relaxed);
            }
        }
        if let Ok(val) = std::env::var("SFRAME_CACHE_CAPACITY_PER_FILE") {
            if let Ok(n) = parse_byte_size(&val) {
                FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE.store(n, Ordering::Relaxed);
            }
        }
    });
}

/// Parse a byte size string. Supports plain integers and suffixes:
/// `K`/`KB`, `M`/`MB`, `G`/`GB` (case-insensitive).
fn parse_byte_size(s: &str) -> Result<usize, ()> {
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Get the maximum total bytes for the CacheFs in-memory store.
pub fn get_cache_capacity() -> usize {
    ensure_init();
    FILEIO_MAXIMUM_CACHE_CAPACITY.load(Ordering::Relaxed)
}

/// Set the maximum total bytes for the CacheFs in-memory store.
pub fn set_cache_capacity(bytes: usize) {
    ensure_init();
    FILEIO_MAXIMUM_CACHE_CAPACITY.store(bytes, Ordering::Relaxed);
}

/// Get the maximum size of a single file in the CacheFs in-memory store.
pub fn get_cache_capacity_per_file() -> usize {
    ensure_init();
    FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE.load(Ordering::Relaxed)
}

/// Set the maximum size of a single file in the CacheFs in-memory store.
pub fn set_cache_capacity_per_file(bytes: usize) {
    ensure_init();
    FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE.store(bytes, Ordering::Relaxed);
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
        // These may have been overridden by env vars in CI, so just check they're nonzero
        assert!(get_cache_capacity() > 0);
        assert!(get_cache_capacity_per_file() > 0);
    }

    #[test]
    fn test_set_get() {
        let original = get_cache_capacity();
        set_cache_capacity(999);
        assert_eq!(get_cache_capacity(), 999);
        set_cache_capacity(original); // restore

        let original_pf = get_cache_capacity_per_file();
        set_cache_capacity_per_file(42);
        assert_eq!(get_cache_capacity_per_file(), 42);
        set_cache_capacity_per_file(original_pf); // restore
    }
}
