//! Reference-counted ephemeral storage backend (`cache://`).
//!
//! `CacheFs` implements `VirtualFileSystem` with a two-tier storage model:
//! - **In-memory**: Small files (≤ per-file limit) are kept entirely in RAM
//!   as long as the total in-memory budget is not exceeded. This avoids disk
//!   I/O entirely for small intermediate results.
//! - **On-disk**: Files that exceed the per-file limit or that don't fit in
//!   the remaining memory budget are written to a temp directory on disk.
//!
//! Files are reference-counted via `CacheGuard`: when a file's reference count
//! drops to zero, it is removed from both memory and disk.
//!
//! Cache capacity is controlled by two globals in `sframe_config`:
//! - `SFRAME_CACHE_CAPACITY` — total in-memory budget (default 2 GiB)
//! - `SFRAME_CACHE_CAPACITY_PER_FILE` — max per-file in-memory size (default 128 MiB)

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use sframe_types::error::Result;

use crate::vfs::{ReadableFile, VirtualFileSystem, WritableFile};

/// Global instance counter for unique root directories.
static INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Global CacheFs instance shared across the process.
static GLOBAL_CACHE_FS: LazyLock<Arc<CacheFs>> = LazyLock::new(|| {
    Arc::new(CacheFs::new().expect("Failed to create global CacheFs"))
});

/// Get the global CacheFs instance.
pub fn global_cache_fs() -> &'static Arc<CacheFs> {
    &GLOBAL_CACHE_FS
}

/// Reference-counted ephemeral filesystem with in-memory caching.
pub struct CacheFs {
    /// Root directory for on-disk spill files.
    root: PathBuf,
    /// Monotonic counter for generating unique file paths.
    counter: AtomicU64,
    /// Reference counts per file path.
    refcounts: Mutex<HashMap<String, usize>>,
    /// In-memory file store. Files small enough to fit in cache live here.
    in_memory: Mutex<HashMap<String, Arc<Vec<u8>>>>,
    /// Total bytes currently stored in memory.
    total_cached_bytes: AtomicUsize,
    /// Max total in-memory bytes (captured from config at construction).
    cache_capacity: usize,
    /// Max per-file in-memory bytes (captured from config at construction).
    cache_capacity_per_file: usize,
    /// Whether to delete the root dir on Drop.
    owns_root: bool,
}

impl CacheFs {
    /// Create a new CacheFs backed by a system temp directory.
    /// Cache capacity limits are read from `sframe_config` at construction time.
    pub fn new() -> Result<Self> {
        let id = INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "sframe_cache_{}_{}",
            std::process::id(),
            id
        ));
        fs::create_dir_all(&root)?;
        Ok(CacheFs {
            root,
            counter: AtomicU64::new(0),
            refcounts: Mutex::new(HashMap::new()),
            in_memory: Mutex::new(HashMap::new()),
            total_cached_bytes: AtomicUsize::new(0),
            cache_capacity: sframe_config::get_cache_capacity(),
            cache_capacity_per_file: sframe_config::get_cache_capacity_per_file(),
            owns_root: true,
        })
    }

    /// Create a CacheFs with explicit capacity limits (for testing).
    pub fn with_limits(cache_capacity: usize, cache_capacity_per_file: usize) -> Result<Self> {
        let id = INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "sframe_cache_{}_{}",
            std::process::id(),
            id
        ));
        fs::create_dir_all(&root)?;
        Ok(CacheFs {
            root,
            counter: AtomicU64::new(0),
            refcounts: Mutex::new(HashMap::new()),
            in_memory: Mutex::new(HashMap::new()),
            total_cached_bytes: AtomicUsize::new(0),
            cache_capacity,
            cache_capacity_per_file,
            owns_root: true,
        })
    }

    /// Create a CacheFs backed by a specific directory (for testing).
    pub fn with_root(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root)?;
        Ok(CacheFs {
            root,
            counter: AtomicU64::new(0),
            refcounts: Mutex::new(HashMap::new()),
            in_memory: Mutex::new(HashMap::new()),
            total_cached_bytes: AtomicUsize::new(0),
            cache_capacity: sframe_config::get_cache_capacity(),
            cache_capacity_per_file: sframe_config::get_cache_capacity_per_file(),
            owns_root: false,
        })
    }

    /// Allocate a new unique path in this cache. The path is a `cache://N` URL.
    /// The file does not exist yet — call `open_write` to create it.
    pub fn alloc_path(&self) -> String {
        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("cache://{}", id)
    }

    /// Allocate a new unique directory path in this cache.
    /// Returns a `cache://dir_N` URL. The directory does not exist yet.
    pub fn alloc_dir(&self) -> String {
        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("cache://dir_{}", id)
    }

    /// Remove all files under a directory prefix and the directory itself.
    pub fn remove_dir(&self, dir_path: &str) -> Result<()> {
        let prefix = format!("{}/", dir_path);
        // Remove all in-memory files under this prefix
        {
            let mut mem = self.in_memory.lock().unwrap();
            let keys_to_remove: Vec<String> = mem
                .keys()
                .filter(|k| k.starts_with(&prefix))
                .cloned()
                .collect();
            for key in &keys_to_remove {
                if let Some(data) = mem.remove(key) {
                    self.total_cached_bytes
                        .fetch_sub(data.len(), Ordering::Relaxed);
                }
            }
        }
        // Clean up refcounts
        {
            let mut rc = self.refcounts.lock().unwrap();
            rc.retain(|k, _| !k.starts_with(&prefix));
        }
        // Remove on-disk directory (best effort)
        let real_path = self.resolve_path(dir_path);
        let _ = fs::remove_dir_all(&real_path);
        Ok(())
    }

    /// Increment the reference count for a cached file.
    pub fn retain(&self, path: &str) {
        let mut rc = self.refcounts.lock().unwrap();
        *rc.entry(path.to_string()).or_insert(0) += 1;
    }

    /// Decrement the reference count. Deletes the file if it reaches zero.
    pub fn release(&self, path: &str) {
        let mut rc = self.refcounts.lock().unwrap();
        if let Some(count) = rc.get_mut(path) {
            *count -= 1;
            if *count == 0 {
                rc.remove(path);
                // Remove from in-memory store
                {
                    let mut mem = self.in_memory.lock().unwrap();
                    if let Some(data) = mem.remove(path) {
                        self.total_cached_bytes
                            .fetch_sub(data.len(), Ordering::Relaxed);
                    }
                }
                // Best-effort delete from disk
                let real_path = self.resolve_path(path);
                let _ = fs::remove_file(&real_path);
            }
        }
    }

    /// Current total bytes stored in memory.
    pub fn total_cached_bytes(&self) -> usize {
        self.total_cached_bytes.load(Ordering::Relaxed)
    }

    /// Number of files currently stored in memory.
    pub fn in_memory_count(&self) -> usize {
        self.in_memory.lock().unwrap().len()
    }

    /// Resolve a `cache://N` path to its real filesystem path.
    fn resolve_path(&self, path: &str) -> PathBuf {
        let name = path.strip_prefix("cache://").unwrap_or(path);
        self.root.join(name)
    }

    /// Try to store data in memory. Returns true if successful.
    fn try_store_in_memory(&self, path: &str, data: Vec<u8>) -> bool {
        let size = data.len();
        let per_file_limit = self.cache_capacity_per_file;
        let total_limit = self.cache_capacity;

        if size > per_file_limit {
            return false;
        }

        // Check if adding this would exceed the total budget.
        // Use a CAS loop to atomically reserve space.
        loop {
            let current = self.total_cached_bytes.load(Ordering::Relaxed);
            if current + size > total_limit {
                return false;
            }
            match self.total_cached_bytes.compare_exchange_weak(
                current,
                current + size,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }

        let mut mem = self.in_memory.lock().unwrap();
        mem.insert(path.to_string(), Arc::new(data));
        true
    }

    /// Write data to disk for the given path.
    fn write_to_disk(&self, path: &str, data: &[u8]) -> Result<()> {
        let real_path = self.resolve_path(path);
        let mut file = BufWriter::new(File::create(&real_path)?);
        file.write_all(data)?;
        file.flush()?;
        Ok(())
    }
}

impl Drop for CacheFs {
    fn drop(&mut self) {
        if self.owns_root {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

// ---------------------------------------------------------------------------
// Readable file implementations
// ---------------------------------------------------------------------------

/// Shared byte buffer that implements `AsRef<[u8]>` for use with `Cursor`.
struct SharedBytes(Arc<Vec<u8>>);

impl AsRef<[u8]> for SharedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Readable file backed by in-memory data.
struct InMemoryReadableFile {
    cursor: Cursor<SharedBytes>,
    size: u64,
}

impl Read for InMemoryReadableFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        Read::read(&mut self.cursor, buf)
    }
}

impl Seek for InMemoryReadableFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl ReadableFile for InMemoryReadableFile {
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }
}

/// Readable file backed by an on-disk file.
struct DiskReadableFile {
    file: BufReader<File>,
    size: u64,
}

impl Read for DiskReadableFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

impl Seek for DiskReadableFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}

impl ReadableFile for DiskReadableFile {
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }
}

// ---------------------------------------------------------------------------
// Writable file implementation
// ---------------------------------------------------------------------------

/// Writable file that buffers in memory, then decides storage tier on flush.
struct CacheFsWritableFile {
    path: String,
    buffer: Vec<u8>,
    cache_fs: Arc<CacheFs>,
    flushed: bool,
}

impl Write for CacheFsWritableFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // No-op: actual flush happens in flush_all()
        Ok(())
    }
}

impl WritableFile for CacheFsWritableFile {
    fn flush_all(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;

        let data = std::mem::take(&mut self.buffer);
        if !self.cache_fs.try_store_in_memory(&self.path, data.clone()) {
            // Didn't fit in memory — write to disk
            self.cache_fs.write_to_disk(&self.path, &data)?;
        }
        Ok(())
    }
}

impl Drop for CacheFsWritableFile {
    fn drop(&mut self) {
        if !self.flushed {
            // Best-effort flush on drop
            let _ = self.flush_all();
        }
    }
}

// ---------------------------------------------------------------------------
// VirtualFileSystem implementation
// ---------------------------------------------------------------------------

impl CacheFs {
    /// Open a file for writing. Requires `self: &Arc<CacheFs>` so the writer
    /// can hold a reference back to the cache for deferred storage decisions.
    pub fn open_cache_write(self: &Arc<Self>, path: &str) -> Result<Box<dyn WritableFile>> {
        Ok(Box::new(CacheFsWritableFile {
            path: path.to_string(),
            buffer: Vec::new(),
            cache_fs: self.clone(),
            flushed: false,
        }))
    }
}

impl VirtualFileSystem for CacheFs {
    fn open_read(&self, path: &str) -> Result<Box<dyn ReadableFile>> {
        // Check in-memory store first
        {
            let mem = self.in_memory.lock().unwrap();
            if let Some(data) = mem.get(path) {
                let size = data.len() as u64;
                return Ok(Box::new(InMemoryReadableFile {
                    cursor: Cursor::new(SharedBytes(data.clone())),
                    size,
                }));
            }
        }

        // Fall back to disk
        let real_path = self.resolve_path(path);
        let file = File::open(&real_path)?;
        let size = file.metadata()?.len();
        Ok(Box::new(DiskReadableFile {
            file: BufReader::new(file),
            size,
        }))
    }

    fn open_write(&self, path: &str) -> Result<Box<dyn WritableFile>> {
        // VFS trait takes &self, but the writer needs Arc<CacheFs>.
        // Fall back to direct disk write when called via the trait.
        let real_path = self.resolve_path(path);
        let file = File::create(&real_path)?;
        Ok(Box::new(DiskWritableFile {
            file: BufWriter::new(file),
        }))
    }

    fn exists(&self, path: &str) -> Result<bool> {
        // Check in-memory first
        {
            let mem = self.in_memory.lock().unwrap();
            if mem.contains_key(path) {
                return Ok(true);
            }
        }
        let real_path = self.resolve_path(path);
        Ok(real_path.exists())
    }

    fn mkdir_p(&self, path: &str) -> Result<()> {
        let real_path = self.resolve_path(path);
        fs::create_dir_all(real_path)?;
        Ok(())
    }

    fn remove(&self, path: &str) -> Result<()> {
        // Remove from in-memory store
        {
            let mut mem = self.in_memory.lock().unwrap();
            if let Some(data) = mem.remove(path) {
                self.total_cached_bytes
                    .fetch_sub(data.len(), Ordering::Relaxed);
            }
        }
        // Remove from disk (best effort)
        let real_path = self.resolve_path(path);
        let _ = fs::remove_file(real_path);
        // Clean up refcount
        let mut rc = self.refcounts.lock().unwrap();
        rc.remove(path);
        Ok(())
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>> {
        let real_path = self.resolve_path(path);
        let mut entries = Vec::new();
        if real_path.exists() {
            for entry in fs::read_dir(real_path)? {
                let entry = entry?;
                if let Some(name) = entry.file_name().to_str() {
                    entries.push(name.to_string());
                }
            }
        }
        Ok(entries)
    }
}

/// Simple disk-only writable file (used when called through VFS trait without Arc).
struct DiskWritableFile {
    file: BufWriter<File>,
}

impl Write for DiskWritableFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

impl WritableFile for DiskWritableFile {
    fn flush_all(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CacheGuard
// ---------------------------------------------------------------------------

/// RAII guard for a reference-counted cached file.
///
/// Calls `retain` on construction and `release` on drop.
/// Clone increments the reference count.
pub struct CacheGuard {
    path: String,
    fs: Arc<CacheFs>,
}

impl CacheGuard {
    /// Create a new guard, incrementing the file's reference count.
    pub fn new(path: String, fs: Arc<CacheFs>) -> Self {
        fs.retain(&path);
        CacheGuard { path, fs }
    }

    /// The cache path this guard refers to.
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl Clone for CacheGuard {
    fn clone(&self) -> Self {
        self.fs.retain(&self.path);
        CacheGuard {
            path: self.path.clone(),
            fs: self.fs.clone(),
        }
    }
}

impl Drop for CacheGuard {
    fn drop(&mut self) {
        self.fs.release(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_fs_write_read() {
        let cache = Arc::new(CacheFs::new().unwrap());
        let path = cache.alloc_path();

        {
            let mut w = cache.open_cache_write(&path).unwrap();
            w.write_all(b"hello cache").unwrap();
            w.flush_all().unwrap();
        }

        {
            let mut r = cache.open_read(&path).unwrap();
            let mut buf = String::new();
            r.read_to_string(&mut buf).unwrap();
            assert_eq!(buf, "hello cache");
        }

        cache.remove(&path).unwrap();
    }

    #[test]
    fn test_small_file_stays_in_memory() {
        let cache = Arc::new(CacheFs::new().unwrap());
        let path = cache.alloc_path();

        {
            let mut w = cache.open_cache_write(&path).unwrap();
            w.write_all(b"small data").unwrap();
            w.flush_all().unwrap();
        }

        // Should be in memory
        assert_eq!(cache.in_memory_count(), 1);
        assert_eq!(cache.total_cached_bytes(), 10);

        // Should NOT be on disk
        let real_path = cache.resolve_path(&path);
        assert!(!real_path.exists());

        // But should still be readable
        {
            let mut r = cache.open_read(&path).unwrap();
            let mut buf = String::new();
            r.read_to_string(&mut buf).unwrap();
            assert_eq!(buf, "small data");
        }

        cache.remove(&path).unwrap();
        assert_eq!(cache.in_memory_count(), 0);
        assert_eq!(cache.total_cached_bytes(), 0);
    }

    #[test]
    fn test_large_file_goes_to_disk() {
        // Use per-instance limits instead of mutating globals
        let cache = Arc::new(CacheFs::with_limits(2 * 1024 * 1024, 10).unwrap());
        let path = cache.alloc_path();

        {
            let mut w = cache.open_cache_write(&path).unwrap();
            // Write more than the 10-byte per-file limit
            w.write_all(b"this is way too large for memory").unwrap();
            w.flush_all().unwrap();
        }

        // Should NOT be in memory
        assert_eq!(cache.in_memory_count(), 0);
        assert_eq!(cache.total_cached_bytes(), 0);

        // Should be on disk
        let real_path = cache.resolve_path(&path);
        assert!(real_path.exists());

        // Should still be readable
        {
            let mut r = cache.open_read(&path).unwrap();
            let mut buf = String::new();
            r.read_to_string(&mut buf).unwrap();
            assert_eq!(buf, "this is way too large for memory");
        }

        cache.remove(&path).unwrap();
    }

    #[test]
    fn test_total_capacity_limit() {
        // Use per-instance limits: total=20, per_file=15
        let cache = Arc::new(CacheFs::with_limits(20, 15).unwrap());

        // Write file1 (10 bytes) — fits in memory
        let path1 = cache.alloc_path();
        {
            let mut w = cache.open_cache_write(&path1).unwrap();
            w.write_all(b"0123456789").unwrap();
            w.flush_all().unwrap();
        }
        assert_eq!(cache.in_memory_count(), 1);
        assert_eq!(cache.total_cached_bytes(), 10);

        // Write file2 (15 bytes) — per-file OK, but total would be 25 > 20
        let path2 = cache.alloc_path();
        {
            let mut w = cache.open_cache_write(&path2).unwrap();
            w.write_all(b"0123456789abcde").unwrap();
            w.flush_all().unwrap();
        }
        // file2 should be on disk, not memory
        assert_eq!(cache.in_memory_count(), 1);
        assert_eq!(cache.total_cached_bytes(), 10);

        // Both should be readable
        {
            let mut r = cache.open_read(&path1).unwrap();
            let mut buf = String::new();
            r.read_to_string(&mut buf).unwrap();
            assert_eq!(buf, "0123456789");
        }
        {
            let mut r = cache.open_read(&path2).unwrap();
            let mut buf = String::new();
            r.read_to_string(&mut buf).unwrap();
            assert_eq!(buf, "0123456789abcde");
        }

        cache.remove(&path1).unwrap();
        cache.remove(&path2).unwrap();
    }

    #[test]
    fn test_cache_guard_refcounting() {
        let cache = Arc::new(CacheFs::new().unwrap());
        let path = cache.alloc_path();

        {
            let mut w = cache.open_cache_write(&path).unwrap();
            w.write_all(b"data").unwrap();
            w.flush_all().unwrap();
        }

        assert!(cache.exists(&path).unwrap());

        let guard1 = CacheGuard::new(path.clone(), cache.clone());
        let guard2 = guard1.clone();

        drop(guard1);
        assert!(cache.exists(&path).unwrap());

        drop(guard2);
        assert!(!cache.exists(&path).unwrap());
    }

    #[test]
    fn test_cache_fs_seek() {
        let cache = Arc::new(CacheFs::new().unwrap());
        let path = cache.alloc_path();

        {
            let mut w = cache.open_cache_write(&path).unwrap();
            w.write_all(&[0u8; 100]).unwrap();
            w.flush_all().unwrap();
        }

        let mut r = cache.open_read(&path).unwrap();
        assert_eq!(r.size().unwrap(), 100);
        r.seek(SeekFrom::Start(90)).unwrap();
        let mut buf = [0u8; 10];
        r.read_exact(&mut buf).unwrap();

        cache.remove(&path).unwrap();
    }

    #[test]
    fn test_cache_fs_multiple_readers() {
        let cache = Arc::new(CacheFs::new().unwrap());
        let path = cache.alloc_path();

        {
            let mut w = cache.open_cache_write(&path).unwrap();
            for i in 0u64..10 {
                w.write_all(&i.to_le_bytes()).unwrap();
            }
            w.flush_all().unwrap();
        }

        let mut r1 = cache.open_read(&path).unwrap();
        let mut r2 = cache.open_read(&path).unwrap();

        r1.seek(SeekFrom::Start(0)).unwrap();
        r2.seek(SeekFrom::Start(40)).unwrap();

        let mut buf = [0u8; 8];
        r1.read_exact(&mut buf).unwrap();
        assert_eq!(u64::from_le_bytes(buf), 0);

        r2.read_exact(&mut buf).unwrap();
        assert_eq!(u64::from_le_bytes(buf), 5);

        cache.remove(&path).unwrap();
    }

    #[test]
    fn test_cache_fs_cleanup_on_drop() {
        let root;
        {
            let cache = CacheFs::new().unwrap();
            root = cache.root.clone();
            let path = cache.alloc_path();
            // Use VFS trait (goes to disk since no Arc)
            let mut w = cache.open_write(&path).unwrap();
            w.write_all(b"temp").unwrap();
            w.flush_all().unwrap();
            assert!(root.exists());
        }
        assert!(!root.exists());
    }

    #[test]
    fn test_in_memory_release_frees_memory() {
        let cache = Arc::new(CacheFs::new().unwrap());
        let path = cache.alloc_path();

        {
            let mut w = cache.open_cache_write(&path).unwrap();
            w.write_all(b"some data here").unwrap();
            w.flush_all().unwrap();
        }

        assert!(cache.total_cached_bytes() > 0);

        let guard = CacheGuard::new(path.clone(), cache.clone());
        drop(guard); // refcount → 0, should free memory

        assert_eq!(cache.total_cached_bytes(), 0);
        assert_eq!(cache.in_memory_count(), 0);
    }
}
