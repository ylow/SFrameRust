//! Local filesystem backend implementing VirtualFileSystem.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use sframe_types::error::Result;

use crate::vfs::{ReadableFile, VirtualFileSystem, WritableFile};

/// Local filesystem implementation of VirtualFileSystem.
pub struct LocalFileSystem;

struct LocalReadableFile {
    file: BufReader<File>,
    size: u64,
}

impl Read for LocalReadableFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

impl Seek for LocalReadableFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}

impl ReadableFile for LocalReadableFile {
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }
}

struct LocalWritableFile {
    file: BufWriter<File>,
}

impl Write for LocalWritableFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

impl WritableFile for LocalWritableFile {
    fn flush_all(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }
}

impl VirtualFileSystem for LocalFileSystem {
    fn open_read(&self, path: &str) -> Result<Box<dyn ReadableFile>> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        Ok(Box::new(LocalReadableFile {
            file: BufReader::new(file),
            size,
        }))
    }

    fn open_write(&self, path: &str) -> Result<Box<dyn WritableFile>> {
        let file = File::create(path)?;
        Ok(Box::new(LocalWritableFile {
            file: BufWriter::new(file),
        }))
    }

    fn exists(&self, path: &str) -> Result<bool> {
        Ok(std::path::Path::new(path).exists())
    }

    fn mkdir_p(&self, path: &str) -> Result<()> {
        fs::create_dir_all(path)?;
        Ok(())
    }

    fn remove(&self, path: &str) -> Result<()> {
        fs::remove_file(path)?;
        Ok(())
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                entries.push(name.to_string());
            }
        }
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_file() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem;
        let path = dir.path().join("test.txt");
        let path_str = path.to_str().unwrap();

        let mut wf = fs.open_write(path_str).unwrap();
        wf.write_all(b"hello world").unwrap();
        wf.flush_all().unwrap();
        drop(wf);

        let mut rf = fs.open_read(path_str).unwrap();
        let mut buf = String::new();
        rf.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "hello world");
    }

    #[test]
    fn test_seek_and_size() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem;
        let path = dir.path().join("data.bin");
        let path_str = path.to_str().unwrap();

        let mut wf = fs.open_write(path_str).unwrap();
        wf.write_all(&[0u8; 100]).unwrap();
        wf.flush_all().unwrap();
        drop(wf);

        let mut rf = fs.open_read(path_str).unwrap();
        assert_eq!(rf.size().unwrap(), 100);
        rf.seek(SeekFrom::Start(90)).unwrap();
        let mut buf = [0u8; 10];
        rf.read_exact(&mut buf).unwrap();
    }

    #[test]
    fn test_exists() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem;
        let path = dir.path().join("nope.txt");
        assert!(!fs.exists(path.to_str().unwrap()).unwrap());

        // Create the file
        let path_str = path.to_str().unwrap();
        let mut wf = fs.open_write(path_str).unwrap();
        wf.flush_all().unwrap();
        drop(wf);
        assert!(fs.exists(path_str).unwrap());
    }

    #[test]
    fn test_read_to_string() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem;
        let path = dir.path().join("msg.txt");
        let path_str = path.to_str().unwrap();

        let mut wf = fs.open_write(path_str).unwrap();
        wf.write_all(b"test content").unwrap();
        wf.flush_all().unwrap();
        drop(wf);

        assert_eq!(fs.read_to_string(path_str).unwrap(), "test content");
    }

    #[test]
    fn test_list_dir() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem;

        // Create some files
        for name in ["a.txt", "b.txt", "c.txt"] {
            let p = dir.path().join(name);
            let mut wf = fs.open_write(p.to_str().unwrap()).unwrap();
            wf.flush_all().unwrap();
        }

        let mut entries = fs.list_dir(dir.path().to_str().unwrap()).unwrap();
        entries.sort();
        assert_eq!(entries, vec!["a.txt", "b.txt", "c.txt"]);
    }
}
