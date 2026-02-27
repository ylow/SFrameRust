//! Virtual filesystem abstraction.
//!
//! All file access in SFrame goes through this trait, enabling:
//! - Local filesystem (default)
//! - cache:// for reference-counted ephemeral storage
//! - Future: S3, HTTP, HDFS backends

use sframe_types::error::Result;
use std::io::{Read, Seek, Write};

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
}
