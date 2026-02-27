//! HDFS virtual filesystem backend.
//!
//! Enables reading/writing SFrame files from Hadoop Distributed File System
//! via the WebHDFS REST API.
//!
//! Enable with the `hdfs` feature flag.
//!
//! Requires a running HDFS NameNode with WebHDFS enabled.

// HDFS backend uses the WebHDFS REST API for file operations.
// This module provides the VFS trait implementation structure.
//
// To use: add `sframe-io = { features = ["hdfs"] }` to your Cargo.toml,
// then construct an HdfsFileSystem with the NameNode URL.
//
// Future implementation plan:
// - WebHDFS REST API (http://namenode:50070/webhdfs/v1/)
// - OPEN for reads (with offset/length for seeking)
// - CREATE + APPEND for writes
// - MKDIRS, DELETE, LISTSTATUS for directory operations
// - Kerberos/SPNEGO authentication support
// - HA NameNode failover

/// HDFS filesystem configuration (available when `hdfs` feature is enabled).
#[derive(Debug, Clone)]
pub struct HdfsConfig {
    /// WebHDFS base URL (e.g., "http://namenode:50070").
    pub namenode_url: String,
    /// HDFS user name.
    pub user: Option<String>,
}
