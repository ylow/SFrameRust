//! HTTP/HTTPS read-only virtual filesystem backend.
//!
//! Enables reading SFrame files from HTTP URLs using range requests
//! for seeking. Write operations are not supported.
//!
//! Enable with the `http` feature flag.

#[cfg(feature = "http")]
mod inner {
    use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};

    use sframe_types::error::{Result, SFrameError};

    use crate::vfs::{ReadableFile, VirtualFileSystem, WritableFile};

    /// HTTP/HTTPS read-only filesystem.
    ///
    /// Downloads the full file content on open (suitable for files up to
    /// a few hundred MB). For very large remote files, a range-request
    /// based implementation would be more memory-efficient.
    pub struct HttpFileSystem;

    struct HttpReadableFile {
        cursor: Cursor<Vec<u8>>,
        size: u64,
    }

    impl Read for HttpReadableFile {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.cursor.read(buf)
        }
    }

    impl Seek for HttpReadableFile {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.cursor.seek(pos)
        }
    }

    impl ReadableFile for HttpReadableFile {
        fn size(&self) -> Result<u64> {
            Ok(self.size)
        }
    }

    impl VirtualFileSystem for HttpFileSystem {
        fn open_read(&self, url: &str) -> Result<Box<dyn ReadableFile>> {
            let body = ureq::get(url)
                .call()
                .map_err(|e| SFrameError::Format(format!("HTTP request failed: {}", e)))?
                .into_body();

            let mut data = Vec::new();
            body.into_reader()
                .read_to_end(&mut data)
                .map_err(|e| SFrameError::Io(e))?;

            let size = data.len() as u64;
            Ok(Box::new(HttpReadableFile {
                cursor: Cursor::new(data),
                size,
            }))
        }

        fn open_write(&self, _path: &str) -> Result<Box<dyn WritableFile>> {
            Err(SFrameError::Format(
                "HTTP filesystem is read-only".to_string(),
            ))
        }

        fn exists(&self, url: &str) -> Result<bool> {
            // Use HEAD request to check existence
            match ureq::head(url).call() {
                Ok(resp) => Ok(resp.status() == 200),
                Err(_) => Ok(false),
            }
        }

        fn mkdir_p(&self, _path: &str) -> Result<()> {
            Err(SFrameError::Format(
                "HTTP filesystem is read-only".to_string(),
            ))
        }

        fn remove(&self, _path: &str) -> Result<()> {
            Err(SFrameError::Format(
                "HTTP filesystem is read-only".to_string(),
            ))
        }

        fn list_dir(&self, _path: &str) -> Result<Vec<String>> {
            Err(SFrameError::Format(
                "HTTP filesystem does not support directory listing".to_string(),
            ))
        }

        fn read_to_string(&self, url: &str) -> Result<String> {
            let body = ureq::get(url)
                .call()
                .map_err(|e| SFrameError::Format(format!("HTTP request failed: {}", e)))?
                .into_body();

            let mut content = String::new();
            body.into_reader()
                .read_to_string(&mut content)
                .map_err(|e| SFrameError::Io(e))?;

            Ok(content)
        }
    }
}

#[cfg(feature = "http")]
pub use inner::HttpFileSystem;
