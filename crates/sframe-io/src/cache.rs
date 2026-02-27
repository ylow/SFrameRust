//! Read cache for frequently accessed blocks and file handles.
//!
//! Provides an LRU block cache that wraps any `VirtualFileSystem` and
//! caches file read results to avoid repeated I/O for the same data.

use std::collections::HashMap;
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use sframe_types::error::{Result, SFrameError};

use crate::vfs::{ReadableFile, VirtualFileSystem, WritableFile};

/// A cached file that stores its entire content in memory.
struct CachedReadableFile {
    cursor: Cursor<Vec<u8>>,
    size: u64,
}

impl Read for CachedReadableFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl Seek for CachedReadableFile {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl ReadableFile for CachedReadableFile {
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }
}

/// LRU-like read cache that wraps a VirtualFileSystem.
///
/// Caches entire file contents in memory. Suitable for repeatedly
/// reading the same small-to-medium files (e.g., SFrame metadata files,
/// segment files that are read multiple times).
///
/// The cache has a maximum total size in bytes. When the limit is
/// exceeded, the least recently inserted entries are evicted.
pub struct CachedFileSystem<F: VirtualFileSystem> {
    inner: F,
    cache: Mutex<FileCache>,
    max_cache_bytes: usize,
}

struct FileCache {
    entries: HashMap<String, Arc<Vec<u8>>>,
    insertion_order: Vec<String>,
    total_bytes: usize,
}

impl FileCache {
    fn new() -> Self {
        FileCache {
            entries: HashMap::new(),
            insertion_order: Vec::new(),
            total_bytes: 0,
        }
    }

    fn get(&self, path: &str) -> Option<Arc<Vec<u8>>> {
        self.entries.get(path).cloned()
    }

    fn insert(&mut self, path: String, data: Arc<Vec<u8>>, max_bytes: usize) {
        let data_len = data.len();

        // Evict entries if we'd exceed the budget
        while self.total_bytes + data_len > max_bytes && !self.insertion_order.is_empty() {
            let oldest = self.insertion_order.remove(0);
            if let Some(old_data) = self.entries.remove(&oldest) {
                self.total_bytes -= old_data.len();
            }
        }

        self.total_bytes += data_len;
        self.entries.insert(path.clone(), data);
        self.insertion_order.push(path);
    }

    fn invalidate(&mut self, path: &str) {
        if let Some(data) = self.entries.remove(path) {
            self.total_bytes -= data.len();
            self.insertion_order.retain(|p| p != path);
        }
    }
}

impl<F: VirtualFileSystem> CachedFileSystem<F> {
    /// Create a new cached filesystem with the given cache size limit.
    pub fn new(inner: F, max_cache_bytes: usize) -> Self {
        CachedFileSystem {
            inner,
            cache: Mutex::new(FileCache::new()),
            max_cache_bytes,
        }
    }

    /// Invalidate a specific cache entry.
    pub fn invalidate(&self, path: &str) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.invalidate(path);
        }
    }

    /// Clear the entire cache.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.entries.clear();
            cache.insertion_order.clear();
            cache.total_bytes = 0;
        }
    }

    /// Current cache size in bytes.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().map(|c| c.total_bytes).unwrap_or(0)
    }
}

impl<F: VirtualFileSystem> VirtualFileSystem for CachedFileSystem<F> {
    fn open_read(&self, path: &str) -> Result<Box<dyn ReadableFile>> {
        // Check cache first
        if let Ok(cache) = self.cache.lock() {
            if let Some(data) = cache.get(path) {
                let size = data.len() as u64;
                return Ok(Box::new(CachedReadableFile {
                    cursor: Cursor::new((*data).clone()),
                    size,
                }));
            }
        }

        // Cache miss: read from inner filesystem
        let mut file = self.inner.open_read(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        let data = Arc::new(data);
        let size = data.len() as u64;

        // Store in cache
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(path.to_string(), data.clone(), self.max_cache_bytes);
        }

        Ok(Box::new(CachedReadableFile {
            cursor: Cursor::new((*data).clone()),
            size,
        }))
    }

    fn open_write(&self, path: &str) -> Result<Box<dyn WritableFile>> {
        // Invalidate cache for this path since we're writing to it
        self.invalidate(path);
        self.inner.open_write(path)
    }

    fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path)
    }

    fn mkdir_p(&self, path: &str) -> Result<()> {
        self.inner.mkdir_p(path)
    }

    fn remove(&self, path: &str) -> Result<()> {
        self.invalidate(path);
        self.inner.remove(path)
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>> {
        self.inner.list_dir(path)
    }

    fn read_to_string(&self, path: &str) -> Result<String> {
        // Use cached read
        let mut file = self.open_read(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_fs::LocalFileSystem;
    use std::io::Write as _;

    #[test]
    fn test_cached_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let path_str = path.to_str().unwrap();

        // Write a file
        std::fs::write(&path, "hello cached world").unwrap();

        let fs = CachedFileSystem::new(LocalFileSystem, 1024 * 1024);

        // First read — cache miss
        let content1 = fs.read_to_string(path_str).unwrap();
        assert_eq!(content1, "hello cached world");
        assert!(fs.cache_size() > 0);

        // Second read — cache hit
        let content2 = fs.read_to_string(path_str).unwrap();
        assert_eq!(content2, "hello cached world");
    }

    #[test]
    fn test_cache_invalidation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.txt");
        let path_str = path.to_str().unwrap();

        std::fs::write(&path, "version 1").unwrap();

        let fs = CachedFileSystem::new(LocalFileSystem, 1024 * 1024);

        let v1 = fs.read_to_string(path_str).unwrap();
        assert_eq!(v1, "version 1");

        // Overwrite the file
        std::fs::write(&path, "version 2").unwrap();

        // Still cached (stale)
        let v1_cached = fs.read_to_string(path_str).unwrap();
        assert_eq!(v1_cached, "version 1");

        // Invalidate
        fs.invalidate(path_str);
        let v2 = fs.read_to_string(path_str).unwrap();
        assert_eq!(v2, "version 2");
    }

    #[test]
    fn test_cache_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let fs = CachedFileSystem::new(LocalFileSystem, 50); // Tiny cache: 50 bytes

        // Write files larger than cache
        for i in 0..5 {
            let path = dir.path().join(format!("{}.txt", i));
            std::fs::write(&path, format!("data-{}-padding-padding", i)).unwrap();
            let _ = fs.read_to_string(path.to_str().unwrap()).unwrap();
        }

        // Cache should not exceed limit (approximately)
        assert!(fs.cache_size() <= 100); // Some slack for boundary
    }
}
