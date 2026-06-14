//! Minimal GGUF Q8_0 tensor reader for the qwen3-tts codec model.
//!
//! Focused on reading Q8_0 quantized weight tensors from the codec GGUF file
//! (`qwen-tokenizer-12hz-Q8_0.gguf`). Dequantizes to `f32` at load time for
//! simplicity. Does NOT cover all GGML types — only Q8_0, I32, and F32.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// ---------------------------------------------------------------------------
// GGUF constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
const GGUF_VERSION: u32 = 3;

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_I32: u32 = 11;
const GGML_TYPE_Q8_0: u32 = 8;

const GGUF_ALIGNMENT: u64 = 32;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single Q8_0 quantized block: 2 bytes (f16 scale) + 32 × i8.
#[derive(Debug, Clone)]
pub struct Q8_0Block {
    pub scale: f32,
    pub values: [i8; 32],
}

/// Metadata for one tensor in the GGUF file.
#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: u32,
    pub shape: Vec<u64>, // stored in reverse (C order: shape[0] is outermost)
    pub offset: u64,     // byte offset in the file (aligned)
}

/// Loaded GGUF file with tensor index and file handle.
#[derive(Debug)]
pub struct GgufFile {
    file: File,
    pub tensors: Vec<TensorInfo>,
    tensor_data_offset: u64,
}

// ---------------------------------------------------------------------------
// Q8_0 block operations
// ---------------------------------------------------------------------------

const Q8_0_BLOCK_SIZE: usize = 34; // 2 (f16 scale) + 32 (i8 values)
const Q8_0_VALUES: usize = 32;

/// Dequantize a single Q8_0 block to 32 f32 values.
#[inline]
pub fn dequantize_q8_0_block(data: &[u8]) -> [f32; Q8_0_VALUES] {
    debug_assert!(data.len() >= Q8_0_BLOCK_SIZE);

    let scale = f16_to_f32(&data[..2]);
    let mut out = [0.0f32; Q8_0_VALUES];
    for i in 0..Q8_0_VALUES {
        out[i] = (data[2 + i] as i8 as f32) * scale;
    }
    out
}

/// Convert 2-byte half-float (IEEE 754-16) to f32.
fn f16_to_f32(bytes: &[u8]) -> f32 {
    debug_assert!(bytes.len() >= 2);
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f16_bits_to_f32(bits)
}

/// Convert a u16 representing an IEEE 754 half-precision float to f32.
fn f16_bits_to_f32(h: u16) -> f32 {
    let raw_sign = (h >> 15) & 0x1;
    let raw_exp = (h >> 10) & 0x1F;   // 5-bit exponent
    let raw_mant = h & 0x03FF;         // 10-bit mantissa

    if raw_exp == 0 {
        // Zero or subnormal
        if raw_mant == 0 {
            // Signed zero
            if raw_sign == 0 {
                0.0
            } else {
                -0.0
            }
        } else {
            // Subnormal: (-1)^s * (m / 2^10) * 2^(-14)
            let m = raw_mant as f32;
            if raw_sign != 0 {
                -(m / 1024.0) * (2.0f32).powi(-14)
            } else {
                (m / 1024.0) * (2.0f32).powi(-14)
            }
        }
    } else if raw_exp == 31 {
        // Infinity or NaN
        let sign_bit = (raw_sign as u32) << 31;
        let exp_mant = 0x7F80_0000 | ((raw_mant as u32) << 13);
        f32::from_bits(sign_bit | exp_mant)
    } else {
        // Normal: (-1)^s * 2^(e-15) * (1 + m/2^10)
        let f32_exp = (raw_exp as i32) - 15 + 127; // f32 bias = 127
        debug_assert!(f32_exp > 0 && f32_exp < 255, "f16 normal mapped outside f32 normal range");
        let f32_bits = ((raw_sign as u32) << 31)
            | ((f32_exp as u32) << 23)
            | ((raw_mant as u32) << 13);
        f32::from_bits(f32_bits)
    }
}

// ---------------------------------------------------------------------------
// GGUF reader implementation
// ---------------------------------------------------------------------------

impl GgufFile {
    /// Open a GGUF file and read its header + tensor index.
    ///
    /// # Errors
    ///
    /// Returns an error message if the file cannot be opened, the magic number
    /// doesn't match, or the version is unsupported.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .map_err(|e| format!("cannot open GGUF file '{}': {e}", path.display()))?;

        // --- Header ---
        let magic = read_u32(&mut file)?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "bad magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"
            ));
        }

        let version = read_u32(&mut file)?;
        if version != GGUF_VERSION {
            return Err(format!(
                "unsupported GGUF version {version} (expected {GGUF_VERSION})"
            ));
        }

        let tensor_count = read_u64(&mut file)?;
        let metadata_kv_count = read_u64(&mut file)?;

        // --- Metadata key-value pairs (skip) ---
        for _ in 0..metadata_kv_count {
            skip_gguf_key_value(&mut file)?;
        }

        // --- Tensor index ---
        // GGUF v3 format: name → n_dims (uint32) → dims (int64 × n_dims) → type (uint32) → offset (uint64)
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut file)?;
            let n_dims = read_u32(&mut file)?;

            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(read_i64(&mut file)? as u64);
            }

            let ggml_type = read_u32(&mut file)?;
            let offset = read_u64(&mut file)?;

            tensors.push(TensorInfo {
                name,
                ggml_type,
                shape,
                offset,
            });
        }

        // --- Tensor data starts after the index, aligned ---
        let pos = file.stream_position().map_err(|e| e.to_string())?;
        let tensor_data_offset = align_up(pos, GGUF_ALIGNMENT);

        Ok(Self {
            file,
            tensors,
            tensor_data_offset,
        })
    }

    /// Find a tensor by name, return its info.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Read a tensor and dequantize it to a flat `Vec<f32>`.
    ///
    /// Supports Q8_0 (dequantized), F32 (copied), and I32 (converted to f32).
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not found or the type is unsupported.
    pub fn read_tensor_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let info = self
            .tensor(name)
            .ok_or_else(|| format!("tensor '{name}' not found"))?;

        let n_elems: usize = info.shape.iter().map(|&d| d as usize).product();

        match info.ggml_type {
            GGML_TYPE_Q8_0 => {
                // Each Q8_0 block represents 32 values in 34 bytes
                let n_blocks = n_elems / Q8_0_VALUES;
                let raw_size = n_blocks * Q8_0_BLOCK_SIZE;
                let raw = self.read_raw_at(info.offset, raw_size)?;

                let mut result = Vec::with_capacity(n_elems);
                for block_idx in 0..n_blocks {
                    let start = block_idx * Q8_0_BLOCK_SIZE;
                    let block_vals = dequantize_q8_0_block(&raw[start..]);
                    result.extend_from_slice(&block_vals);
                }
                Ok(result)
            }
            GGML_TYPE_F32 => {
                let raw_size = n_elems * 4;
                let raw = self.read_raw_at(info.offset, raw_size)?;
                let mut result = Vec::with_capacity(n_elems);
                for i in 0..n_elems {
                    let bytes = &raw[i * 4..(i + 1) * 4];
                    result.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
                }
                Ok(result)
            }
            GGML_TYPE_F16 => {
                let raw_size = n_elems * 2;
                let raw = self.read_raw_at(info.offset, raw_size)?;
                let mut result = Vec::with_capacity(n_elems);
                for i in 0..n_elems {
                    let bytes = &raw[i * 2..(i + 1) * 2];
                    let h = u16::from_le_bytes([bytes[0], bytes[1]]);
                    result.push(f16_bits_to_f32(h));
                }
                Ok(result)
            }
            GGML_TYPE_I32 => {
                let raw_size = n_elems * 4;
                let raw = self.read_raw_at(info.offset, raw_size)?;
                let mut result = Vec::with_capacity(n_elems);
                for i in 0..n_elems {
                    let bytes = &raw[i * 4..(i + 1) * 4];
                    result.push(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32);
                }
                Ok(result)
            }
            other => Err(format!(
                "unsupported tensor type {other} for '{name}' (expected Q8_0={GGML_TYPE_Q8_0}, \
                 F16={GGML_TYPE_F16}, F32={GGML_TYPE_F32}, I32={GGML_TYPE_I32})"
            )),
        }
    }

    /// Read a tensor's raw bytes (for Q8_0 data) without dequantizing.
    ///
    /// Useful for on-the-fly dequant during matmul to save memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not found or the type is not Q8_0.
    pub fn read_tensor_q8_0_raw(&mut self, name: &str) -> Result<Vec<u8>, String> {
        let info = self
            .tensor(name)
            .ok_or_else(|| format!("tensor '{name}' not found"))?;

        if info.ggml_type != GGML_TYPE_Q8_0 {
            return Err(format!(
                "tensor '{name}' is type {} not Q8_0",
                info.ggml_type
            ));
        }

        let n_elems: usize = info.shape.iter().map(|&d| d as usize).product();
        let n_blocks = n_elems / Q8_0_VALUES;
        let raw_size = n_blocks * Q8_0_BLOCK_SIZE;
        self.read_raw_at(info.offset, raw_size)
    }

    // -----------------------------------------------------------------------
    // Private I/O helpers
    // -----------------------------------------------------------------------

    fn read_raw_at(&mut self, tensor_offset: u64, size: usize) -> Result<Vec<u8>, String> {
        let byte_offset = self.tensor_data_offset + tensor_offset;
        self.file
            .seek(SeekFrom::Start(byte_offset))
            .map_err(|e| format!("seek error at offset {byte_offset}: {e}"))?;

        let mut buf = vec![0u8; size];
        self.file
            .read_exact(&mut buf)
            .map_err(|e| format!("read error at offset {byte_offset}: {e}"))?;
        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

fn read_u32(file: &mut File) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(file: &mut File) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(file: &mut File) -> Result<i64, String> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(i64::from_le_bytes(buf))
}

fn read_gguf_string(file: &mut File) -> Result<String, String> {
    let len = read_u64(file)? as usize;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    String::from_utf8(buf).map_err(|e| format!("invalid UTF-8 in GGUF string: {e}"))
}

fn skip_gguf_key_value(file: &mut File) -> Result<(), String> {
    // Key
    let _key = read_gguf_string(file)?;

    // Value type
    let _val_type = read_u32(file)?;

    // Value data — skip based on type (we don't need metadata for codec decode)
    // We just skip everything. Types: 0=uint8, 1=int8, 2=uint16, 3=int16,
    // 4=uint32, 5=int32, 6=float32, 7=bool, 8=string, 9=array, 10=uint64,
    // 11=int64, 12=float64
    //
    // For simplicity, we read based on the type byte.

    // Actually, we'll just read type-by-type to be safe.
    // But to keep it simple: skip remaining by seeking to end of KV.
    // Problem is we don't know the total size upfront. So we must parse.

    // For this codebase, we only read the GGUF to find tensors.
    // Metadata values can be large (tokenizer arrays), so we need to handle all types.

    match _val_type {
        0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 | 11 | 12 => {
            // Fixed-size: 1, 1, 2, 2, 4, 4, 4, 1, 8, 8, 8 bytes respectively
            let sizes: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];
            let size = sizes[_val_type as usize];
            file.seek(SeekFrom::Current(size as i64))
                .map_err(|e| e.to_string())?;
        }
        8 => {
            // String: length prefix (u64) + data
            let len = read_u64(file)? as i64;
            file.seek(SeekFrom::Current(len))
                .map_err(|e| e.to_string())?;
        }
        9 => {
            // Array: type (u32) + length (u64) + elements
            let arr_type = read_u32(file)?;
            let arr_len = read_u64(file)?;
            for _ in 0..arr_len {
                match arr_type {
                    0..=7 | 10 | 11 | 12 => {
                        let sizes: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];
                        let size = sizes[arr_type as usize];
                        file.seek(SeekFrom::Current(size as i64))
                            .map_err(|e| e.to_string())?;
                    }
                    8 => {
                        let len = read_u64(file)? as i64;
                        file.seek(SeekFrom::Current(len))
                            .map_err(|e| e.to_string())?;
                    }
                    _ => {
                        return Err(format!("unknown array element type {arr_type}"));
                    }
                }
            }
        }
        _ => {
            return Err(format!("unknown GGUF value type {_val_type}"));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_basics() {
        // 1.0 in f16
        assert!((f16_bits_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0.5 in f16
        assert!((f16_bits_to_f32(0x3800) - 0.5).abs() < 1e-6);
        // -2.0 in f16
        assert!((f16_bits_to_f32(0xC000) - (-2.0)).abs() < 1e-6);
        // 0.0
        assert!((f16_bits_to_f32(0x0000) - 0.0).abs() < 1e-6);
        // Inf
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn dequantize_q8_0_block_identity() {
        // Scale = 1.0, values = [1, 2, 3, ..., 32]
        let mut block_data = vec![0u8; Q8_0_BLOCK_SIZE];
        // f16 for 1.0 = 0x3C00
        block_data[0] = 0x00;
        block_data[1] = 0x3C;
        for i in 0..32 {
            block_data[2 + i] = (i + 1) as u8;
        }

        let result = dequantize_q8_0_block(&block_data);
        assert_eq!(result.len(), 32);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[15] - 16.0).abs() < 1e-6);
        assert!((result[31] - 32.0).abs() < 1e-6);
    }
}
