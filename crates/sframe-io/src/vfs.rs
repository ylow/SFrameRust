//! Virtual filesystem abstraction.
//!
//! All file access in SFrame goes through this trait, enabling:
//! - Local filesystem (default)
//! - cache:// for reference-counted ephemeral storage
//! - Future: S3, HTTP, HDFS backends

use std::sync::Arc;

use sframe_types::error::Result;
use std::io::{Read, Seek, Write};

use crate::cache_fs::CacheFs;

/// A readable file with seek and size support.
pub trait ReadableFile: Read + Seek + Send {
    /// Total size of the file in bytes.
    fn size(&self) -> Result<u64>;
}

/// A writable file.
pub trait WritableFile: Write + Send {
    /// Flush all buffered data to the underlying storage.
    fn flush_all(&mut self) -> Result<()>;
}

/// Virtual filesystem backend.
pub trait VirtualFileSystem: Send + Sync {
    /// Open a file for reading.
    fn open_read(&self, path: &str) -> Result<Box<dyn ReadableFile>>;

    /// Open a file for writing (creates or truncates).
    fn open_write(&self, path: &str) -> Result<Box<dyn WritableFile>>;

    /// Check if a path exists.
    fn exists(&self, path: &str) -> Result<bool>;

    /// Create directories recursively.
    fn mkdir_p(&self, path: &str) -> Result<()>;

    /// Remove a file.
    fn remove(&self, path: &str) -> Result<()>;

    /// List files in a directory.
    fn list_dir(&self, path: &str) -> Result<Vec<String>>;

    /// Read the entire contents of a file as a string.
    fn read_to_string(&self, path: &str) -> Result<String> {
        let mut file = self.open_read(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(contents)
    }

    /// Write a string to a file (creates or truncates).
    fn write_string(&self, path: &str, content: &str) -> Result<()> {
        let mut file = self.open_write(path)?;
        file.write_all(content.as_bytes())?;
        file.flush_all()?;
        Ok(())
    }
}

/// VFS wrapper around `Arc<CacheFs>` that routes writes through the
/// memory-caching tier (`open_cache_write`).
pub struct ArcCacheFsVfs(pub Arc<CacheFs>);

impl VirtualFileSystem for ArcCacheFsVfs {
    fn open_read(&self, path: &str) -> Result<Box<dyn ReadableFile>> {
        self.0.open_read(path)
    }

    fn open_write(&self, path: &str) -> Result<Box<dyn WritableFile>> {
        self.0.open_cache_write(path)
    }

    fn exists(&self, path: &str) -> Result<bool> {
        self.0.exists(path)
    }

    fn mkdir_p(&self, path: &str) -> Result<()> {
        self.0.mkdir_p(path)
    }

    fn remove(&self, path: &str) -> Result<()> {
        self.0.remove(path)
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>> {
        self.0.list_dir(path)
    }
}
