//! S3 virtual filesystem backend.
//!
//! Enables reading/writing SFrame files from Amazon S3 using the AWS SDK.
//! Reads use range requests for seeking; writes use multipart upload.
//!
//! Enable with the `s3` feature flag.
//!
//! Requires AWS credentials configured via environment variables,
//! config files, or instance profiles.

// S3 backend requires the aws-sdk-s3 crate which is a heavy dependency.
// This module provides the VFS trait implementation structure; the actual
// aws-sdk-s3 integration is gated behind the `s3` feature flag.
//
// To use: add `sframe-io = { features = ["s3"] }` to your Cargo.toml,
// then construct an S3FileSystem with bucket and optional prefix.
//
// Future implementation plan:
// - aws-sdk-s3 for API calls
// - Seekable reads via GetObject with Range header
// - Multipart upload for writes (>5MB parts)
// - AWS credential chain (env vars, ~/.aws/config, instance profile, SSO)
// - Region auto-detection from bucket
// - Server-side encryption support

/// S3 filesystem configuration (available when `s3` feature is enabled).
#[derive(Debug, Clone)]
pub struct S3Config {
    /// S3 bucket name.
    pub bucket: String,
    /// Optional key prefix (virtual directory).
    pub prefix: Option<String>,
    /// AWS region (e.g., "us-east-1"). None for auto-detect.
    pub region: Option<String>,
}
