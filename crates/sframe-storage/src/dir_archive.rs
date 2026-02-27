//! Parser for dir_archive.ini files.
//!
//! Format:
//! ```text
//! [archive]
//! version=1
//! num_prefixes=3
//!
//! [metadata]
//! contents=sframe
//!
//! [prefixes]
//! 0000=dir_archive.ini
//! 0001=objects.bin
//! 0002=m_9688d6320ff94822
//! ```

use sframe_types::error::{Result, SFrameError};

/// Parsed dir_archive.ini metadata.
pub struct DirArchive {
    pub version: u32,
    pub contents: String,
    pub prefixes: Vec<String>,
}

impl DirArchive {
    /// Parse a dir_archive.ini file content.
    pub fn parse(content: &str) -> Result<Self> {
        let mut version = 0u32;
        let mut contents = String::new();
        let mut prefixes = Vec::new();
        let mut section = String::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = line[1..line.len() - 1].to_string();
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                match section.as_str() {
                    "archive" => {
                        if key == "version" {
                            version = value.parse().map_err(|_| {
                                SFrameError::Format(format!(
                                    "Invalid version in dir_archive.ini: {}",
                                    value
                                ))
                            })?;
                        }
                    }
                    "metadata" => {
                        if key == "contents" {
                            contents = value.to_string();
                        }
                    }
                    "prefixes" => {
                        prefixes.push(value.to_string());
                    }
                    _ => {}
                }
            }
        }

        Ok(DirArchive {
            version,
            contents,
            prefixes,
        })
    }

    /// Find the data prefix (the one that's not dir_archive.ini or objects.bin).
    pub fn data_prefix(&self) -> Result<&str> {
        for p in &self.prefixes {
            if p != "dir_archive.ini" && p != "objects.bin" {
                return Ok(p);
            }
        }
        Err(SFrameError::Format(
            "No data prefix found in dir_archive.ini".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../samples", manifest)
    }

    #[test]
    fn test_parse_dir_archive() {
        let path = format!("{}/business.sf/dir_archive.ini", samples_dir());
        let content = std::fs::read_to_string(&path).unwrap();
        let archive = DirArchive::parse(&content).unwrap();
        assert_eq!(archive.version, 1);
        assert_eq!(archive.contents, "sframe");
        assert_eq!(archive.prefixes.len(), 3);
        assert_eq!(archive.data_prefix().unwrap(), "m_9688d6320ff94822");
    }
}
