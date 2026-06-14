//! Q8_0 quantized operations: dequant, dot product, and matrix multiply.
//!
//! Q8_0 is GGML's 8-bit block quantization format:
//! - Each block of 32 values is stored as 34 bytes:
//!   - 2 bytes: f16 scale
//!   - 32 bytes: int8 quantized values
//! - Dequantized value = quantized_value × scale
//!
//! This module provides:
//! - `dequantize_q8_0_block`: single block → 32 f32
//! - `q8_0_dot`: integer dot product of two i8 blocks
//! - `q8_0_matmul_f32`: Q8_0 × f32 matrix multiply (RowMajor)
//! - `q8_0_dequant_row`: full Q8_0 row → Vec<f32>

use crate::gguf::f16_to_f32;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const Q8_0_BLOCK_SIZE: usize = 34; // 2 (f16 scale) + 32 (i8 values)
pub const Q8_0_VALUES: usize = 32;

// ---------------------------------------------------------------------------
// Block dequant
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Dot product (i8 × i8)
// ---------------------------------------------------------------------------

/// Dot product of two i8 arrays of length 32.
///
/// Returns i32 to avoid overflow (max: 32 × 127 × 127 = 516,128).
#[inline]
pub fn q8_0_dot(a: &[i8; Q8_0_VALUES], b: &[i8; Q8_0_VALUES]) -> i32 {
    let mut sum = 0i32;
    for i in 0..Q8_0_VALUES {
        sum += (a[i] as i32) * (b[i] as i32);
    }
    sum
}

// ---------------------------------------------------------------------------
// Block-level utilities
// ---------------------------------------------------------------------------

/// Dequantize a single Q8_0 block into an f32 slice (length must be 32).
#[inline]
pub fn dequantize_block_into(data: &[u8], out: &mut [f32]) {
    debug_assert!(data.len() >= Q8_0_BLOCK_SIZE);
    debug_assert!(out.len() >= Q8_0_VALUES);

    let scale = f16_to_f32(&data[..2]);
    for i in 0..Q8_0_VALUES {
        out[i] = (data[2 + i] as i8 as f32) * scale;
    }
}

/// Dequantize a full Q8_0 row (K elements, K must be multiple of 32) into Vec<f32>.
pub fn q8_0_dequant_row(data: &[u8], k: usize) -> Vec<f32> {
    debug_assert!(k % Q8_0_VALUES == 0, "Q8_0 row length must be multiple of 32");
    let n_blocks = k / Q8_0_VALUES;
    let mut result = Vec::with_capacity(k);
    for block_idx in 0..n_blocks {
        let start = block_idx * Q8_0_BLOCK_SIZE;
        let block_vals = dequantize_q8_0_block(&data[start..]);
        result.extend_from_slice(&block_vals);
    }
    result
}

// ---------------------------------------------------------------------------
// Matrix multiply: Q8_0 × f32  →  f32
// ---------------------------------------------------------------------------

/// Q8_0 × f32 matrix multiply: C[M, N] = A[M, K] × B[K, N]
///
/// # Arguments
/// * `a_data` — Q8_0 quantized data as flat bytes: M rows, each K/32 blocks of 34 bytes
/// * `a_rows` — M (number of rows in A)
/// * `a_cols` — K (number of columns in A / rows in B, must be multiple of 32)
/// * `b` — B matrix as flat f32 in row-major: [K × N]
/// * `n` — N (number of columns in B)
///
/// # Returns
/// `Vec<f32>` of length M × N in row-major order.
pub fn q8_0_matmul_f32(
    a_data: &[u8],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    n: usize,
) -> Vec<f32> {
    let k = a_cols;
    debug_assert!(k % Q8_0_VALUES == 0, "K must be multiple of 32");
    debug_assert!(a_data.len() >= a_rows * (k / Q8_0_VALUES) * Q8_0_BLOCK_SIZE);
    debug_assert!(b.len() >= k * n);

    let k_blocks = k / Q8_0_VALUES;
    let mut c = vec![0.0f32; a_rows * n];

    for m in 0..a_rows {
        let a_row_start = m * k_blocks * Q8_0_BLOCK_SIZE;

        for n_idx in 0..n {
            let mut sum = 0.0f32;

            for kb in 0..k_blocks {
                let block_start = a_row_start + kb * Q8_0_BLOCK_SIZE;
                let scale = f16_to_f32(&a_data[block_start..block_start + 2]);
                let a_vals = &a_data[block_start + 2..block_start + 2 + Q8_0_VALUES];

                // dequant × f32 dot product for this block
                let mut partial = 0.0f32;
                let b_base = kb * Q8_0_VALUES * n + n_idx;
                for i in 0..Q8_0_VALUES {
                    let av = a_vals[i] as i8 as f32;
                    let bv = b[b_base + i * n];
                    partial += av * bv;
                }
                sum += scale * partial;
            }

            c[m * n + n_idx] = sum;
        }
    }

    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_block_roundtrip() {
        // Scale = 1.0 (f16: 0x3C00), values = [1, 2, 3, ..., 32]
        let mut block = [0u8; Q8_0_BLOCK_SIZE];
        block[0] = 0x00;
        block[1] = 0x3C;
        for i in 0..32 {
            block[2 + i] = (i + 1) as u8;
        }

        let deq = dequantize_q8_0_block(&block);
        assert_eq!(deq.len(), 32);
        assert!((deq[0] - 1.0).abs() < 1e-6);
        assert!((deq[15] - 16.0).abs() < 1e-6);
        assert!((deq[31] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_accuracy() {
        let a: [i8; 32] = [
            1, 2, 3, 4, 5, -1, -2, -3, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let b: [i8; 32] = [
            5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let result = q8_0_dot(&a, &b);
        let expected = 1 * 5 + 2 * 4 + 3 * 3 + 4 * 2 + 5 * 1
            + (-1) * 0 + (-2) * 0 + (-3) * 0;
        assert_eq!(result, expected);
    }

    #[test]
    fn dequant_row_matches_block_by_block() {
        // Build synthetic 2-block Q8_0 row (64 elements)
        // Values must fit in i8 range (-128..127)
        let k = 64;
        let mut raw = vec![0u8; (k / Q8_0_VALUES) * Q8_0_BLOCK_SIZE];

        // Block 0: scale=2.0 (0x4000), values [1, 2, 3, ..., 32]
        raw[0] = 0x00;
        raw[1] = 0x40;
        for i in 0..32 {
            raw[2 + i] = (i + 1) as u8;
        }

        // Block 1: scale=0.5 (0x3800), values [-1, -2, -3, ..., -32]
        let off = Q8_0_BLOCK_SIZE;
        raw[off] = 0x00;
        raw[off + 1] = 0x38;
        for i in 0..32 {
            raw[off + 2 + i] = (-(i as i8 + 1)) as u8;
        }

        let deq_row = q8_0_dequant_row(&raw, k);
        assert_eq!(deq_row.len(), k);

        // Block 0 values should be 2.0 * [1, 2, 3, ..., 32]
        for i in 0..32 {
            let expected = 2.0_f32 * (i + 1) as f32;
            assert!((deq_row[i] - expected).abs() < 1e-4,
                "block 0, i={i}: got {} expected {}", deq_row[i], expected);
        }
        // Block 1 values should be 0.5 * [-1, -2, ..., -32]
        for i in 0..32 {
            let expected = 0.5_f32 * (-(i as f32 + 1.0));
            assert!((deq_row[32 + i] - expected).abs() < 1e-4,
                "block 1, i={i}: got {} expected {}", deq_row[32 + i], expected);
        }
    }

    #[test]
    fn matmul_small_identity() {
        // A = Q8_0[2, 32] with scale=1.0, values = [1,0,0,...,0] (row 0) and [0,1,0,...,0] (row 1)
        let k = 32;
        let a_rows = 2;
        let mut a_data = vec![0u8; a_rows * (k / Q8_0_VALUES) * Q8_0_BLOCK_SIZE];

        // Row 0: val[0] = 1, rest = 0, scale = 1.0
        a_data[0] = 0x00;
        a_data[1] = 0x3C;
        a_data[2] = 1; // first value = 1

        // Row 1: val[1] = 1, rest = 0, scale = 1.0
        let r1_start = Q8_0_BLOCK_SIZE;
        a_data[r1_start] = 0x00;
        a_data[r1_start + 1] = 0x3C;
        a_data[r1_start + 3] = 1; // second value = 1 (index 1)

        // B = eye(32) identity matrix
        let b_rows = k;
        let b_cols = k;
        let mut b = vec![0.0f32; b_rows * b_cols];
        for i in 0..b_rows {
            b[i * b_cols + i] = 1.0;
        }

        // C[2, 32] = A[2, 32] × I[32, 32]
        let c = q8_0_matmul_f32(&a_data, a_rows, k, &b, b_cols);
        assert_eq!(c.len(), a_rows * b_cols);

        // Row 0 should be [1, 0, 0, ..., 0]
        assert!((c[0] - 1.0).abs() < 1e-4, "c[0] = {}", c[0]);
        for j in 1..b_cols {
            assert!(c[j].abs() < 1e-4, "c[{j}] = {}", c[j]);
        }

        // Row 1 should be [0, 1, 0, ..., 0]
        assert!((c[1 * b_cols + 1] - 1.0).abs() < 1e-4,
            "c[1,1] = {}", c[1 * b_cols + 1]);
        for j in 0..b_cols {
            if j != 1 {
                assert!(c[1 * b_cols + j].abs() < 1e-4,
                    "c[1,{j}] = {}", c[1 * b_cols + j]);
            }
        }
    }

    #[test]
    fn matmul_from_real_gguf_weight() {
        // Load a real Q8_0 weight from the GGUF file and verify
        // matmul produces reasonable (finite, non-zero) output
        let path = {
            let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let workspace = manifest_dir.parent().and_then(|p| p.parent()).unwrap();
            workspace.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
        };

        let mut gguf = crate::gguf::GgufFile::open(&path)
            .expect("should open codec GGUF");

        let weight_name = "tok_dec.pre_tfm.input_proj.weight";
        let info = gguf.tensor(weight_name).expect("weight should exist");
        assert_eq!(info.ggml_type, 8, "expected Q8_0 type");

        // Shape from GGUF: [1024, 512] — this is [K, M] in GGML convention
        // In GGML: ne[0]=1024 (innermost = cols), ne[1]=512 (rows)
        // A = weight[M=512, K=1024] as Q8_0
        let k = info.shape[0] as usize;  // 1024
        let m = info.shape[1] as usize;  // 512

        let raw = gguf.read_tensor_q8_0_raw(weight_name)
            .expect("should read Q8_0 raw data");

        // B = random-ish f32[K=1024, N=8]
        let n = 8;
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        let c = q8_0_matmul_f32(&raw, m, k, &b, n);
        assert_eq!(c.len(), m * n);

        // All values should be finite
        assert!(c.iter().all(|v| v.is_finite()), "all values should be finite");

        // At least some values should be non-zero
        let has_nonzero = c.iter().any(|v| v.abs() > 0.001);
        assert!(has_nonzero, "matmul should produce non-zero output for non-zero input");

        let mean = c.iter().sum::<f32>() / c.len() as f32;
        println!("q8_0_matmul_f32({weight_name}) [{m}×{k}]×[{k}×{n}]: mean={mean:.6}");
    }
}
