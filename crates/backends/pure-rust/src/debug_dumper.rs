//! Tensor dumper for deterministic parity vs C++ DebugDumper.
//!
//! Binary format matches C++ `debug.h`:
//!   [ndims : i32] [shape[0] : i32] ... [shape[ndims-1] : i32]
//!   [data : f32 x numel]
//!
//! Each tensor is written as `<dump_dir>/<name>.bin`.
//! Int32 tokens are cast to f32 before writing so the Python/C++ loader
//! can compare them as cossim on the float representation.

use std::fs;
use std::path::Path;

/// Optional dumper; no-ops when dir is `None`.
pub struct DebugDumper {
    dir: Option<String>,
}

impl DebugDumper {
    /// Create a dumper. When `dir` is `None` all dump calls are no-ops.
    pub fn new(dir: Option<String>) -> Self {
        if let Some(ref d) = dir {
            let _ = fs::create_dir_all(d);
        }
        Self { dir }
    }

    /// Dump a 1-D f32 tensor `[n]`.
    pub fn dump_1d(&self, name: &str, data: &[f32]) {
        let ndims: i32 = 1;
        let n: i32 = data.len() as i32;
        self.write_bin(name, &[ndims, n], data);
    }

    /// Dump a 2-D f32 tensor `[rows, cols]`.
    pub fn dump_2d(&self, name: &str, rows: usize, cols: usize, data: &[f32]) {
        let ndims: i32 = 2;
        let r: i32 = rows as i32;
        let c: i32 = cols as i32;
        self.write_bin(name, &[ndims, r, c], data);
    }

    /// Dump int32 tokens as f32 (binary identical to
    /// C++ `debug_dump_i32_as_f32`).
    pub fn dump_i32_as_f32(&self, name: &str, shape: &[i32], data: &[i32]) {
        if self.dir.is_none() {
            return;
        }
        let numel: usize = shape.iter().map(|&d| d as usize).product();
        let ndims = shape.len() as i32;
        let mut buf: Vec<f32> = Vec::with_capacity(numel);
        for &v in data.iter().take(numel) {
            buf.push(v as f32);
        }
        let mut header: Vec<i32> = Vec::with_capacity(shape.len() + 1);
        header.push(ndims);
        header.extend_from_slice(shape);
        self.write_bin(name, &header, &buf);
    }

    /// Low-level write: `[header_i32...] [data_f32...]`.
    fn write_bin(&self, name: &str, header: &[i32], data: &[f32]) {
        let dir = match self.dir {
            Some(ref d) => d.clone(),
            None => return,
        };
        let path = Path::new(&dir).join(format!("{name}.bin"));
        let mut buf: Vec<u8> = Vec::with_capacity(
            header.len() * 4 + data.len() * 4,
        );
        for &v in header {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        if let Err(e) = fs::write(&path, &buf) {
            log::error!("[DebugDumper] failed to write {path:?}: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_dump_1d_roundtrip() {
        let tmp = std::env::temp_dir().join("debug_dumper_test_1d");
        let _ = fs::remove_dir_all(&tmp);
        let dumper = DebugDumper::new(Some(tmp.to_string_lossy().to_string()));

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        dumper.dump_1d("test_1d", &data);

        let bin_path = tmp.join("test_1d.bin");
        assert!(bin_path.exists(), "dump file should exist");

        let raw = fs::read(&bin_path).unwrap();
        // Header: [ndims=1, n=4] = 8 bytes
        // Data: 4 floats = 16 bytes
        assert_eq!(raw.len(), 4 + 4 + 16, "total size mismatch");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_noop_when_disabled() {
        let dumper = DebugDumper::new(None);
        dumper.dump_1d("should_not_exist", &[1.0, 2.0]);
        // No crash — that's the test.
    }

    #[test]
    fn test_dump_i32_as_f32() {
        let tmp = std::env::temp_dir().join("debug_dumper_test_i32");
        let _ = fs::remove_dir_all(&tmp);
        let dumper = DebugDumper::new(Some(tmp.to_string_lossy().to_string()));

        let codes: Vec<i32> = vec![1, 2, 3, 65535];
        dumper.dump_i32_as_f32("codes", &[4], &codes);

        let bin_path = tmp.join("codes.bin");
        assert!(bin_path.exists());
        let raw = fs::read(&bin_path).unwrap();
        // Header: [ndims=1, n=4] = 8 bytes
        // Data: 4 floats = 16 bytes → 24 bytes total
        assert_eq!(raw.len(), 24);

        let _ = fs::remove_dir_all(&tmp);
    }
}
