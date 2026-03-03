//! S3 virtual filesystem backend.
//!
//! Enables reading/writing SFrame files from Amazon S3.
//! Reads download the full object on open (suitable for segment-sized files).
//! Writes buffer in memory and PutObject on flush.
//!
//! Enable with the `s3` feature flag:
//!   `sframe-io = { features = ["s3"] }`
//!
//! Requires AWS credentials configured via environment variables,
//! config files (~/.aws/), or instance profiles.

/// S3 filesystem configuration.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// S3 bucket name.
    pub bucket: String,
    /// Optional key prefix (virtual directory).
    pub prefix: Option<String>,
    /// AWS region (e.g., "us-east-1"). None for auto-detect.
    pub region: Option<String>,
}

#[cfg(feature = "s3")]
mod inner {
    use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};

    use sframe_types::error::{Result, SFrameError};

    use crate::vfs::{ReadableFile, VirtualFileSystem, WritableFile};

    use super::S3Config;

    /// S3-backed virtual filesystem.
    ///
    /// Reads download the full object into memory (suitable for segment files,
    /// typically < 100 MB each). Writes buffer in memory and upload on flush.
    pub struct S3FileSystem {
        client: aws_sdk_s3::Client,
        bucket: String,
        prefix: String,
        rt: tokio::runtime::Runtime,
    }

    impl S3FileSystem {
        /// Create a new S3FileSystem from config.
        ///
        /// Initializes the AWS SDK client using the default credential chain
        /// (environment variables, ~/.aws/config, instance profiles, SSO).
        pub fn new(config: S3Config) -> Result<Self> {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| SFrameError::Format(format!("Failed to create tokio runtime: {}", e)))?;

            let client = rt.block_on(async {
                let mut aws_config = aws_config::defaults(aws_config::BehaviorVersion::latest());
                if let Some(ref region) = config.region {
                    aws_config = aws_config.region(aws_sdk_s3::config::Region::new(region.clone()));
                }
                let sdk_config = aws_config.load().await;
                aws_sdk_s3::Client::new(&sdk_config)
            });

            let prefix = config.prefix.unwrap_or_default();

            Ok(S3FileSystem {
                client,
                bucket: config.bucket,
                prefix,
                rt,
            })
        }

        /// Resolve a logical path to an S3 key.
        fn resolve_key(&self, path: &str) -> String {
            if self.prefix.is_empty() {
                path.to_string()
            } else {
                format!("{}/{}", self.prefix.trim_end_matches('/'), path)
            }
        }
    }

    // -- ReadableFile --

    struct S3ReadableFile {
        cursor: Cursor<Vec<u8>>,
        size: u64,
    }

    impl Read for S3ReadableFile {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.cursor.read(buf)
        }
    }

    impl Seek for S3ReadableFile {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.cursor.seek(pos)
        }
    }

    impl ReadableFile for S3ReadableFile {
        fn size(&self) -> Result<u64> {
            Ok(self.size)
        }
    }

    // -- WritableFile --

    struct S3WritableFile {
        buffer: Vec<u8>,
        client: aws_sdk_s3::Client,
        bucket: String,
        key: String,
        rt_handle: tokio::runtime::Handle,
    }

    impl Write for S3WritableFile {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.buffer.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(()) // actual upload happens in flush_all
        }
    }

    impl WritableFile for S3WritableFile {
        fn flush_all(&mut self) -> Result<()> {
            let data = std::mem::take(&mut self.buffer);
            let body = aws_sdk_s3::primitives::ByteStream::from(data);

            self.rt_handle.block_on(async {
                self.client
                    .put_object()
                    .bucket(&self.bucket)
                    .key(&self.key)
                    .body(body)
                    .send()
                    .await
                    .map_err(|e| SFrameError::Format(format!("S3 PutObject failed: {}", e)))?;
                Ok(())
            })
        }
    }

    impl Drop for S3WritableFile {
        fn drop(&mut self) {
            if !self.buffer.is_empty() {
                let _ = WritableFile::flush_all(self);
            }
        }
    }

    // -- VirtualFileSystem --

    impl VirtualFileSystem for S3FileSystem {
        fn open_read(&self, path: &str) -> Result<Box<dyn ReadableFile>> {
            let key = self.resolve_key(path);
            let data = self.rt.block_on(async {
                let resp = self
                    .client
                    .get_object()
                    .bucket(&self.bucket)
                    .key(&key)
                    .send()
                    .await
                    .map_err(|e| SFrameError::Format(format!("S3 GetObject failed for '{}': {}", key, e)))?;

                let bytes = resp
                    .body
                    .collect()
                    .await
                    .map_err(|e| SFrameError::Format(format!("S3 read body failed: {}", e)))?
                    .into_bytes();

                Ok::<Vec<u8>, SFrameError>(bytes.to_vec())
            })?;

            let size = data.len() as u64;
            Ok(Box::new(S3ReadableFile {
                cursor: Cursor::new(data),
                size,
            }))
        }

        fn open_write(&self, path: &str) -> Result<Box<dyn WritableFile>> {
            let key = self.resolve_key(path);
            Ok(Box::new(S3WritableFile {
                buffer: Vec::new(),
                client: self.client.clone(),
                bucket: self.bucket.clone(),
                key,
                rt_handle: self.rt.handle().clone(),
            }))
        }

        fn exists(&self, path: &str) -> Result<bool> {
            let key = self.resolve_key(path);
            self.rt.block_on(async {
                match self
                    .client
                    .head_object()
                    .bucket(&self.bucket)
                    .key(&key)
                    .send()
                    .await
                {
                    Ok(_) => Ok(true),
                    Err(e) => {
                        let svc_err = e.into_service_error();
                        if svc_err.is_not_found() {
                            Ok(false)
                        } else {
                            Err(SFrameError::Format(format!("S3 HeadObject failed: {}", svc_err)))
                        }
                    }
                }
            })
        }

        fn mkdir_p(&self, _path: &str) -> Result<()> {
            // S3 has no directories — this is a no-op
            Ok(())
        }

        fn remove(&self, path: &str) -> Result<()> {
            let key = self.resolve_key(path);
            self.rt.block_on(async {
                self.client
                    .delete_object()
                    .bucket(&self.bucket)
                    .key(&key)
                    .send()
                    .await
                    .map_err(|e| SFrameError::Format(format!("S3 DeleteObject failed: {}", e)))?;
                Ok(())
            })
        }

        fn list_dir(&self, path: &str) -> Result<Vec<String>> {
            let prefix = self.resolve_key(path);
            let prefix = if prefix.ends_with('/') {
                prefix
            } else {
                format!("{}/", prefix)
            };

            self.rt.block_on(async {
                let resp = self
                    .client
                    .list_objects_v2()
                    .bucket(&self.bucket)
                    .prefix(&prefix)
                    .delimiter("/")
                    .send()
                    .await
                    .map_err(|e| SFrameError::Format(format!("S3 ListObjectsV2 failed: {}", e)))?;

                let mut entries = Vec::new();

                // Files (objects)
                if let Some(contents) = resp.contents() {
                    for obj in contents {
                        if let Some(key) = obj.key() {
                            // Strip the prefix to return relative names
                            let name = key.strip_prefix(&prefix).unwrap_or(key);
                            if !name.is_empty() {
                                entries.push(name.to_string());
                            }
                        }
                    }
                }

                // "Directories" (common prefixes)
                if let Some(prefixes) = resp.common_prefixes() {
                    for cp in prefixes {
                        if let Some(p) = cp.prefix() {
                            let name = p.strip_prefix(&prefix).unwrap_or(p);
                            let name = name.trim_end_matches('/');
                            if !name.is_empty() {
                                entries.push(name.to_string());
                            }
                        }
                    }
                }

                Ok(entries)
            })
        }
    }
}

#[cfg(feature = "s3")]
pub use inner::S3FileSystem;
