//! Custom Q8_0 quantized matmul with single-threaded quantization and
//! optional multi-threaded vec_dot via Rayon.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::k_quants::{BlockQ8_0, GgmlType, QK8_0};
use candle_core::Tensor;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Q8Weights — Q8_0 quantized weight matrix
// ---------------------------------------------------------------------------

/// A Q8_0 quantized weight matrix, stored as contiguous blocks.
///
/// Shape: `[n, k]` where `n` = out_features, `k` = in_features.
/// Each row of `k` elements is stored as `ceil(k / 32)` [`BlockQ8_0`] blocks.
pub struct Q8Weights {
    /// Number of output features (= number of rows / output dim).
    n: usize,
    /// Number of input features (= number of columns / input dim).
    k: usize,
    /// Blocks per row: `ceil(k / 32)`.
    blocks_per_row: usize,
    /// Padded input size in floats. Always `blocks_per_row * QK8_0`.
    padded_k: usize,
    /// Contiguous Q8_0 blocks: `[n * blocks_per_row]`.
    data: Vec<BlockQ8_0>,
}

impl Q8Weights {
    /// Load from GGUF tensor data without dequantizing.
    ///
    /// Reads the raw Q8_0 block data directly from the file at the tensor's
    /// offset + `tensor_data_offset`. This is *much* faster than dequantizing
    /// to F32 at load time (~1s vs ~5s for the full model).
    pub fn from_gguf(content: &Content, file: &mut File, name: &str) -> anyhow::Result<Self> {
        let info = content
            .tensor_infos
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        let shape = info.shape.dims();
        anyhow::ensure!(shape.len() >= 2, "tensor {name} has {shape:?}, expected ≥2D");
        let n = shape[0];
        let k = shape[1];
        let blocks_per_row = k.div_ceil(QK8_0);
        let padded_k = blocks_per_row * QK8_0;
        let total_blocks = n * blocks_per_row;
        let total_bytes = total_blocks * std::mem::size_of::<BlockQ8_0>();

        // Allocate uninitialized block buffer — safe because BlockQ8_0 is
        // f16 + [i8; 32] which has no invalid bit patterns.
        let mut data: Vec<BlockQ8_0> = Vec::with_capacity(total_blocks);
        unsafe {
            data.set_len(total_blocks);
        }

        let file_offset = content.tensor_data_offset + info.offset;
        file.seek(SeekFrom::Start(file_offset))
            .map_err(|e| anyhow::anyhow!("seek for {name}: {e}"))?;

        let buf = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, total_bytes)
        };
        file.read_exact(buf)
            .map_err(|e| anyhow::anyhow!("read {name}: {e}"))?;

        Ok(Self {
            n,
            k,
            blocks_per_row,
            padded_k,
            data,
        })
    }

    // ── accessors ─────────────────────────────────────────────────────────

    /// Return `(out_features, in_features)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    /// Number of output features.
    pub fn out_features(&self) -> usize {
        self.n
    }

    /// Number of input features.
    pub fn in_features(&self) -> usize {
        self.k
    }

    /// Raw Q8_0 block slice.
    pub fn blocks(&self) -> &[BlockQ8_0] {
        &self.data
    }

    /// Block view for a specific weight row.
    pub fn row_blocks(&self, row: usize) -> &[BlockQ8_0] {
        let start = row * self.blocks_per_row;
        &self.data[start..start + self.blocks_per_row]
    }

    /// Construct from pre-built Q8_0 data (for benchmarking).
    #[doc(hidden)]
    pub fn from_raw(n: usize, k: usize, bpr: usize, padded_k: usize, data: Vec<BlockQ8_0>) -> Self {
        Self { n, k, blocks_per_row: bpr, padded_k, data }
    }

    /// Construct Q8_0 weights by quantizing f32 data.
    ///
    /// `data`: `[n × k]` f32 in row-major layout.
    /// `n`: number of output features (rows).
    /// `k`: number of input features (columns).
    pub fn from_f32_data(data: &[f32], n: usize, k: usize) -> Self {
        let bpr = k.div_ceil(QK8_0);
        let padded_k = bpr * QK8_0;
        let total_blocks = n * bpr;
        let mut q8 = vec![BlockQ8_0::zeros(); total_blocks];
        for row in 0..n {
            let mut row_data = data[row * k..(row + 1) * k].to_vec();
            row_data.resize(padded_k, 0.0);
            let dst = &mut q8[row * bpr..(row + 1) * bpr];
            <BlockQ8_0 as GgmlType>::from_float(&row_data, dst);
        }
        Self { n, k, blocks_per_row: bpr, padded_k, data: q8 }
    }
}

// ── workspace ─────────────────────────────────────────────────────────

/// Reusable scratch buffers for allocation-free GEMV / matmul.
///
/// Create one of these per thread / per forward pass and pass to
/// [`Q8Weights::gemv`] / [`Q8Weights::matmul`] to avoid repeated
/// heap allocations on the hot path.
pub struct Q8Workspace {
    padded: Vec<f32>,
    x_q: Vec<BlockQ8_0>,
}

impl Q8Workspace {
    /// Create an empty workspace. Buffers grow on first use and are then
    /// reused via `clear`, avoiding further allocations.
    pub fn new() -> Self {
        Self {
            padded: Vec::new(),
            x_q: Vec::new(),
        }
    }

    /// Ensure `x_q` has at least `bpr` blocks (zero-filled).
    pub(crate) fn ensure_xq(&mut self, bpr: usize) {
        if self.x_q.len() < bpr {
            self.x_q.resize(bpr, BlockQ8_0::zeros());
        }
    }

    /// Copy externally-quantized Q8 blocks into the workspace.
    ///
    /// After calling this, use [`Q8Weights::gemv_into_quantized`] to compute
    /// the GEMV without re-quantizing.
    pub fn set_quantized_input(&mut self, blocks: &[BlockQ8_0]) {
        self.ensure_xq(blocks.len());
        self.x_q[..blocks.len()].clone_from_slice(blocks);
    }

    /// Quantize `x` (length `k`) into the workspace's `x_q` buffer.
    ///
    /// After calling this, use [`Q8Weights::gemv_into_quantized`] to compute
    /// the GEMV without re-quantizing.
    pub fn quantize_input(&mut self, x: &[f32], k: usize, bpr: usize, padded_k: usize) {
        assert_eq!(x.len(), k);
        self.ensure_xq(bpr);
        self.padded.clear();
        self.padded.extend_from_slice(x);
        self.padded.resize(padded_k, 0.0);
        <BlockQ8_0 as GgmlType>::from_float(&self.padded, &mut self.x_q[..bpr]);
    }
}

impl Q8Weights {
    /// GEMV (M = 1): `y = W @ x`.
    ///
    /// `x`: `[k]` f32 — input activations.
    /// `ws`: scratch buffers (reused to avoid allocations).
    /// Returns `[n]` f32 — output.
    ///
    /// Single-threaded loop; no Rayon, no task dispatch overhead.
    /// Automatically uses AVX2 `vec_dot` when compiled with `target-cpu=native`.
    pub fn gemv(&self, x: &[f32], ws: &mut Q8Workspace) -> Vec<f32> {
        let mut dst = vec![0.0f32; self.n];
        self.gemv_into(x, ws, &mut dst);
        dst
    }

    /// GEMV (M = 1) into a pre-allocated destination — allocation-free.
    ///
    /// `x`: `[k]` f32 — input activations.
    /// `ws`: scratch buffers (reused to avoid allocations).
    /// `dst`: `[n]` f32 — pre-allocated output buffer. Overwritten on return.
    pub fn gemv_into(&self, x: &[f32], ws: &mut Q8Workspace, dst: &mut [f32]) {
        assert_eq!(x.len(), self.k, "gemv_into: input length {} != k={}", x.len(), self.k);
        assert_eq!(dst.len(), self.n, "gemv_into: dst length {} != n={}", dst.len(), self.n);

        ws.ensure_xq(self.blocks_per_row);

        // 1. Pad input and quantize (single-threaded — fast, <10µs).
        ws.padded.clear();
        ws.padded.extend_from_slice(x);
        ws.padded.resize(self.padded_k, 0.0);
        <BlockQ8_0 as GgmlType>::from_float(&ws.padded, &mut ws.x_q[..self.blocks_per_row]);

        // 2. Parallel vec_dot across output rows, writing directly into dst.
        let x_q_ref: &[BlockQ8_0] = &ws.x_q[..self.blocks_per_row];
        let padded_k = self.padded_k;
        dst.par_iter_mut()
            .enumerate()
            .for_each(|(row, d)| {
                let w_row = self.row_blocks(row);
                *d = <BlockQ8_0 as GgmlType>::vec_dot(padded_k, w_row, x_q_ref);
            });
    }

    /// GEMV using already-quantized input in `ws.x_q`.
    ///
    /// `ws` must have been prepared by calling `ws.quantize_input(x, self.k, self.blocks_per_row, self.padded_k)`.
    ///
    /// `dst`: `[n]` f32 — pre-allocated output buffer.
    pub fn gemv_into_quantized(&self, ws: &Q8Workspace, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.n);
        let x_q_ref: &[BlockQ8_0] = &ws.x_q[..self.blocks_per_row];
        let padded_k = self.padded_k;
        dst.par_iter_mut()
            .enumerate()
            .for_each(|(row, d)| {
                let w_row = self.row_blocks(row);
                *d = <BlockQ8_0 as GgmlType>::vec_dot(padded_k, w_row, x_q_ref);
            });
    }

    /// Quantize `x` once, then compute gemv into multiple output buffers
    /// in a single flat `par_iter` for maximum thread utilization.
    ///
    /// `weights` and `dsts` must have the same length (1-3). All weights
    /// must have the same `k` (input dimension). Each `dsts[i]` must have
    /// length `weights[i].n`.
    pub fn gemv_multi_into(
        weights: &[&Q8Weights],
        dsts: &mut [&mut [f32]],
        x: &[f32],
        ws: &mut Q8Workspace,
    ) {
        assert!(!weights.is_empty(), "gemv_multi_into: empty weights");
        assert_eq!(weights.len(), dsts.len(), "gemv_multi_into: weights.len != dsts.len");
        let bpr = weights[0].blocks_per_row;
        let padded_k = weights[0].padded_k;
        let k = weights[0].k;
        for w in weights {
            assert_eq!(w.k, k, "gemv_multi_into: all weights must have same k");
        }
        for (w, d) in weights.iter().zip(dsts.iter()) {
            assert_eq!(d.len(), w.n, "gemv_multi_into: dst[?].len != w.n");
        }
        ws.quantize_input(x, k, bpr, padded_k);

        let x_q_ref: &[BlockQ8_0] = &ws.x_q[..bpr];

        // Build flat offset table: (weight_idx, row_start_in_flat, row_end_exclusive)
        let mut offsets: Vec<(usize, usize, usize)> = Vec::with_capacity(weights.len());
        let mut flat_ofs = 0usize;
        for (wi, w) in weights.iter().enumerate() {
            offsets.push((wi, flat_ofs, flat_ofs + w.n));
            flat_ofs += w.n;
        }
        let total_rows = flat_ofs;

        // Single flat par_iter across ALL rows of ALL weights.
        let flat_results: Vec<f32> = (0..total_rows)
            .into_par_iter()
            .map(|flat_row| {
                let &(wi, row_start, _row_end) = offsets
                    .iter()
                    .find(|&&(_wi, start, end)| flat_row >= start && flat_row < end)
                    .unwrap();
                let row_in_w = flat_row - row_start;
                let w = weights[wi];
                let w_row = w.row_blocks(row_in_w);
                <BlockQ8_0 as GgmlType>::vec_dot(padded_k, w_row, x_q_ref)
            })
            .collect();

        // Scatter back into per-weight dst buffers.
        for (wi, start, end) in offsets {
            let slice = &flat_results[start..end];
            dsts[wi].copy_from_slice(slice);
        }
    }

    /// Batch matmul (M ≥ 1): `Y = X @ W^T`.
    ///
    /// `x`: `[m, k]` f32 — flattened input activations (row-major).
    /// `m`: number of rows.
    /// `ws`: scratch buffers (reused to avoid allocations).
    /// Returns `[m, n]` f32 — flattened output.
    ///
    /// Batch matmul: `Y = X @ W^T`.
    ///
    /// Parallelizes across input rows. Each thread quantizes its own input
    /// row independently, then computes vec_dot against all weight rows.
    ///
    /// `x`: `[m, k]` f32 flattened (row-major). `m`: number of rows.
    /// Returns `[m, n]` f32 flattened.
    pub fn matmul(&self, x: &[f32], m: usize, _ws: &mut Q8Workspace) -> Vec<f32> {
        assert_eq!(x.len(), m * self.k, "matmul: x.len {} != m*k={}", x.len(), m * self.k);

        let k = self.k;
        let padded_k = self.padded_k;
        let bpr = self.blocks_per_row;
        let n = self.n;

        (0..m)
            .into_par_iter()
            .flat_map(|row_idx| {
                let x_row = &x[row_idx * k..(row_idx + 1) * k];

                // Per-row quantize (small local alloc — unavoidable with parallel rows)
                let mut local_padded = Vec::with_capacity(padded_k);
                local_padded.extend_from_slice(x_row);
                local_padded.resize(padded_k, 0.0);
                let mut local_x_q = vec![BlockQ8_0::zeros(); bpr];
                <BlockQ8_0 as GgmlType>::from_float(&local_padded, &mut local_x_q);

                (0..n)
                    .map(|col_idx| {
                        let w_row = self.row_blocks(col_idx);
                        <BlockQ8_0 as GgmlType>::vec_dot(padded_k, w_row, &local_x_q)
                    })
                    .collect::<Vec<f32>>()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Q8Linear — Tensor wrapper helper
// ---------------------------------------------------------------------------

/// Apply a [`Q8Weights`] linear projection to a candle [`Tensor`].
///
/// Handles arbitrary-rank input tensors: flattens all but the last dim,
/// computes `x @ W^T`, and reshapes the result.
///
/// `ws`: scratch workspace reused across calls to avoid allocations.
/// Uses the single-threaded `gemv` (M=1) or `matmul` (M>1) internally.
pub fn q8_linear(weights: &Q8Weights, x: &Tensor, ws: &mut Q8Workspace) -> anyhow::Result<Tensor> {
    let x_dims = x.dims();
    let rank = x_dims.len();
    anyhow::ensure!(rank >= 1, "q8_linear: input must be at least 1D");
    let bsz: usize = x_dims[..rank - 1].iter().product();
    let in_features = x_dims[rank - 1];
    anyhow::ensure!(
        in_features == weights.k,
        "q8_linear: input last dim {in_features} != weights.k {}",
        weights.k
    );

    // Flatten to 2D and get raw data (flatten_all first because to_vec1
    // requires rank 1).
    let x_2d = x.reshape((bsz, in_features))?;
    let x_slice = x_2d.flatten_all()?.to_vec1::<f32>()?;

    // Compute (reuses ws to avoid allocation on the hot path).
    let y_vec = if bsz == 1 {
        weights.gemv(&x_slice, ws)
    } else {
        weights.matmul(&x_slice, bsz, ws)
    };

    // Reshape back. Use from_slice (accepts &[usize] via Into<Shape>)
    // rather than from_vec (requires ShapeWithOneHole).
    let mut out_dims = x_dims.to_vec();
    out_dims[rank - 1] = weights.n;
    let y = Tensor::from_slice(&y_vec, out_dims.as_slice(), x.device())?;
    Ok(y)
}

// ---------------------------------------------------------------------------
// Q8LinearMulti — fused projections sharing one input quantization
// ---------------------------------------------------------------------------

/// Quantize input **once**, then compute multiple output projections in parallel.
///
/// All weights must share the same `k` (input dimension). The input `x` is
/// quantized a single time, then `vec_dot` runs against **every row of every
/// weight matrix** in a single flat `par_iter` for maximum throughput.
///
/// For M > 1 (batch matmul), falls back to per-weight `q8_linear` calls since
/// each input row requires independent quantization.
pub fn q8_linear_multi(
    weights: &[&Q8Weights],
    x: &Tensor,
    ws: &mut Q8Workspace,
) -> anyhow::Result<Vec<Tensor>> {
    anyhow::ensure!(!weights.is_empty(), "q8_linear_multi: empty weights");
    let first = weights[0];
    for w in weights {
        anyhow::ensure!(
            w.k == first.k,
            "q8_linear_multi: all weights must have same k (got {} and {})",
            w.k,
            first.k,
        );
    }

    let x_dims = x.dims();
    let rank = x_dims.len();
    anyhow::ensure!(rank >= 1, "q8_linear_multi: input must be at least 1D");
    let bsz: usize = x_dims[..rank - 1].iter().product();
    let in_features = x_dims[rank - 1];
    anyhow::ensure!(
        in_features == first.k,
        "q8_linear_multi: input features {in_features} != k={}",
        first.k,
    );

    let device = x.device();

    if bsz == 1 {
        // ── M=1 (GEMV): one quantize, flat parallel vec_dot ────────────
        let x_2d = x.reshape((1, in_features))?;
        let x_slice = x_2d.flatten_all()?.to_vec1::<f32>()?;

        ws.ensure_xq(first.blocks_per_row);
        ws.padded.clear();
        ws.padded.extend_from_slice(&x_slice);
        ws.padded.resize(first.padded_k, 0.0);
        <BlockQ8_0 as GgmlType>::from_float(&ws.padded, &mut ws.x_q[..first.blocks_per_row]);

        let x_q_ref: &[BlockQ8_0] = &ws.x_q[..first.blocks_per_row];
        let padded_k = first.padded_k;

        // Build row-offset table: for each weight, where its rows start in flat index.
        let mut offsets: Vec<usize> = Vec::with_capacity(weights.len() + 1);
        offsets.push(0);
        for w in weights {
            offsets.push(offsets.last().unwrap() + w.n);
        }
        let total_rows = *offsets.last().unwrap();

        // Single flat par_iter across ALL weight rows.
        let flat_results: Vec<f32> = (0..total_rows)
            .into_par_iter()
            .map(|flat_row| {
                // Find which weight and row within that weight.
                // Binary search would be faster for many weights; linear is fine for 2-3.
                let (wi, row_in_w) = offsets[..weights.len()]
                    .binary_search(&flat_row)
                    .map(|i| (i, 0))
                    .unwrap_or_else(|i| (i - 1, flat_row - offsets[i - 1]));
                let w_row = weights[wi].row_blocks(row_in_w);
                <BlockQ8_0 as GgmlType>::vec_dot(padded_k, w_row, x_q_ref)
            })
            .collect();

        // Split flat results back into per-weight tensors.
        let mut outputs = Vec::with_capacity(weights.len());
        for i in 0..weights.len() {
            let start = offsets[i];
            let end = offsets[i + 1];
            let slice = &flat_results[start..end];
            outputs.push(Tensor::from_slice(slice, (1, weights[i].n), device)?);
        }
        Ok(outputs)
    } else {
        // ── M > 1 (batch matmul): fall back to sequential per-weight calls ──
        let mut outputs = Vec::with_capacity(weights.len());
        for w in weights {
            outputs.push(q8_linear(w, x, ws)?);
        }
        Ok(outputs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The block size for Q8_0.
    const BLCK: usize = QK8_0;

    #[test]
    fn test_block_size_consistency() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34, "BlockQ8_0 must be 34 bytes");
        assert_eq!(QK8_0, 32, "QK8_0 must be 32");
    }

    /// Create a trivial Q8Weights (n=2, k=64) and check GEMV vs manual.
    #[test]
    fn test_gemv_small() {
        let k = 64usize;
        let n = 2usize;
        let bpr = k.div_ceil(BLCK);

        // Build weights by quantizing known f32 data.
        let mut f32_weight = vec![0.0f32; n * k];
        for col in 0..k {
            f32_weight[0 * k + col] = 1.0; // row 0: all 1.0
            f32_weight[1 * k + col] = 2.0; // row 1: all 2.0
        }
        let mut data = vec![BlockQ8_0::zeros(); n * bpr];
        // Quantize row by row via from_float (padded to bpr*32).
        let padded_k = bpr * BLCK;
        for row in 0..n {
            let mut row_data = f32_weight[row * k..(row + 1) * k].to_vec();
            row_data.resize(padded_k, 0.0);
            let dst = &mut data[row * bpr..(row + 1) * bpr];
            <BlockQ8_0 as GgmlType>::from_float(&row_data, dst);
        }

        let w = Q8Weights {
            n,
            k,
            blocks_per_row: bpr,
            padded_k,
            data,
        };

        // Input = [1.0; 64]
        let x = vec![1.0f32; k];
        let mut ws = Q8Workspace::new();
        let y = w.gemv(&x, &mut ws);

        assert_eq!(y.len(), n);
        // Row 0: all weight 1.0 × input 1.0 = 64.0
        // Row 1: all weight 2.0 × input 1.0 = 128.0
        // Allow 10% tolerance for quantization error.
        assert!(
            (y[0] - 64.0).abs() < 6.4,
            "row 0: expected ~64, got {}",
            y[0]
        );
        assert!(
            (y[1] - 128.0).abs() < 12.8,
            "row 1: expected ~128, got {}",
            y[1]
        );
    }

    #[test]
    fn test_matmul_vs_gemv_equivalence() {
        // For M=1, matmul and gemv should produce the same result.
        let (n, k) = (8usize, 64usize);
        let bpr = k.div_ceil(BLCK);
        let padded_k = bpr * BLCK;

        // Create random-ish weights.
        let mut f32_w = vec![0.0f32; n * k];
        for i in 0..n * k {
            f32_w[i] = ((i * 7) % 100) as f32 / 10.0;
        }
        let mut data = vec![BlockQ8_0::zeros(); n * bpr];
        for row in 0..n {
            let mut row_data = f32_w[row * k..(row + 1) * k].to_vec();
            row_data.resize(padded_k, 0.0);
            let dst = &mut data[row * bpr..(row + 1) * bpr];
            <BlockQ8_0 as GgmlType>::from_float(&row_data, dst);
        }
        let w = Q8Weights { n, k, blocks_per_row: bpr, padded_k, data };

        // Input.
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let mut ws = Q8Workspace::new();

        let y_gemv = w.gemv(&x, &mut ws);
        let y_matmul = w.matmul(&x, 1, &mut ws);

        assert_eq!(y_gemv.len(), n);
        assert_eq!(y_matmul.len(), n);
        for i in 0..n {
            assert!(
                (y_gemv[i] - y_matmul[i]).abs() < 1e-4,
                "M=1 mismatch at {i}: gemv={} matmul={}",
                y_gemv[i],
                y_matmul[i]
            );
        }

        // Also test M=3 batch matmul matches repeated GEMV.
        let m = 3usize;
        let x_batch: Vec<f32> = (0..m * k).map(|i| ((i * 3) % 50) as f32).collect();
        let y_batch = w.matmul(&x_batch, m, &mut ws);

        for batch_row in 0..m {
            let x_row = &x_batch[batch_row * k..(batch_row + 1) * k];
            let y_single = w.gemv(x_row, &mut ws);
            let y_slice = &y_batch[batch_row * n..(batch_row + 1) * n];
            for j in 0..n {
                assert!(
                    (y_single[j] - y_slice[j]).abs() < 1e-4,
                    "batch row {batch_row} col {j}: single={} batch={}",
                    y_single[j],
                    y_slice[j]
                );
            }
        }
    }

    /// Verify q8_linear produces the same result as an equivalent linear_fwd
    /// (x @ W^T) for a small matrix.
    #[test]
    fn test_q8_linear_versus_f32_linear() {
        use crate::talker::linear_fwd;

        let dev = Device::Cpu;
        let (n, k) = (8usize, 64usize);
        let bpr = k.div_ceil(BLCK);
        let padded_k = bpr * BLCK;

        // Build F32 weight.
        let mut w_f32_data = vec![0.0f32; n * k];
        for i in 0..n * k {
            w_f32_data[i] = ((i * 7) % 100) as f32 / 10.0;
        }
        let w_f32 =
            Tensor::from_vec(w_f32_data.clone(), (n, k), &dev).unwrap();

        // Build Q8_0 weight from the same F32 data.
        let mut data = vec![BlockQ8_0::zeros(); n * bpr];
        for row in 0..n {
            let mut row_data = w_f32_data[row * k..(row + 1) * k].to_vec();
            row_data.resize(padded_k, 0.0);
            let dst = &mut data[row * bpr..(row + 1) * bpr];
            <BlockQ8_0 as GgmlType>::from_float(&row_data, dst);
        }
        let w_q8 = Q8Weights { n, k, blocks_per_row: bpr, padded_k, data };

        // Input tensor [1, k].
        let x_data: Vec<f32> = (0..k).map(|i| ((i * 3) % 50) as f32).collect();
        let x = Tensor::from_vec(x_data.clone(), (1, k), &dev).unwrap();

        // F32 linear_fwd.
        let y_f32 = linear_fwd(&w_f32, &x).unwrap();
        let y_f32_vec: Vec<f32> = y_f32.flatten_all().unwrap().to_vec1().unwrap();

        // Q8 q8_linear.
        let mut ws = Q8Workspace::new();
        let y_q8 = q8_linear(&w_q8, &x, &mut ws).unwrap();
        let y_q8_vec: Vec<f32> = y_q8.flatten_all().unwrap().to_vec1().unwrap();

        // Should match within 5% tolerance (Q8_0 quantization error).
        for j in 0..n {
            let diff = (y_f32_vec[j] - y_q8_vec[j]).abs();
            let rel = diff / y_f32_vec[j].abs().max(1e-6);
            assert!(
                rel < 0.05,
                "col {j}: f32={} q8={} rel_err={}",
                y_f32_vec[j],
                y_q8_vec[j],
                rel
            );
        }
    }

    /// Test that the GEMV runs for realistic sizes (no crash, outputs finite).
    #[test]
    fn test_gemv_realistic_size() {
        let (n, k) = (256usize, 256usize);
        let bpr = k.div_ceil(BLCK);
        let padded_k = bpr * BLCK;
        let mut data = vec![BlockQ8_0::zeros(); n * bpr];
        let mut f32_w = vec![0.0f32; n * k];
        for i in 0..n * k {
            f32_w[i] = (i % 100) as f32 / 10.0;
        }
        for row in 0..n {
            let mut row_data = f32_w[row * k..(row + 1) * k].to_vec();
            row_data.resize(padded_k, 0.0);
            let dst = &mut data[row * bpr..(row + 1) * bpr];
            <BlockQ8_0 as GgmlType>::from_float(&row_data, dst);
        }
        let w = Q8Weights { n, k, blocks_per_row: bpr, padded_k, data };
        let x = vec![0.5f32; k];
        let mut ws = Q8Workspace::new();
        let y = w.gemv(&x, &mut ws);
        assert_eq!(y.len(), n);
        for &v in &y {
            assert!(v.is_finite(), "non-finite value in output: {v}");
        }
    }

    /// Test with k not a multiple of 32 (partial last block).
    #[test]
    fn test_gemv_partial_block() {
        let (n, k) = (3usize, 40usize); // 40 = 32 + 8 (partial block)
        let bpr = k.div_ceil(BLCK); // 2
        let padded_k = bpr * BLCK; // 64
        let mut data = vec![BlockQ8_0::zeros(); n * bpr];
        let mut f32_w = vec![0.0f32; n * k];
        for i in 0..n * k {
            f32_w[i] = (i % 100) as f32 / 10.0;
        }
        for row in 0..n {
            let mut row_data = f32_w[row * k..(row + 1) * k].to_vec();
            row_data.resize(padded_k, 0.0);
            let dst = &mut data[row * bpr..(row + 1) * bpr];
            <BlockQ8_0 as GgmlType>::from_float(&row_data, dst);
        }
        let w = Q8Weights { n, k, blocks_per_row: bpr, padded_k, data };
        let x: Vec<f32> = (0..k).map(|i| ((i * 3) % 50) as f32).collect();
        let mut ws = Q8Workspace::new();
        let y = w.gemv(&x, &mut ws);
        assert_eq!(y.len(), n);
        for &v in &y {
            assert!(v.is_finite(), "non-finite value in output: {v}");
        }
    }

    #[test]
    fn test_gemv_partial_block_accuracy() {
        let (n, k) = (4usize, 43usize); // non-multiple of 32
        let bpr = k.div_ceil(BLCK);
        let padded_k = bpr * BLCK;
        let mut data = vec![BlockQ8_0::zeros(); n * bpr];
        let mut f32_w = vec![0.0f32; n * k];
        for i in 0..n * k {
            f32_w[i] = ((i * 7 + 3) % 100) as f32 / 10.0;
        }
        for row in 0..n {
            let mut row_data = f32_w[row * k..(row + 1) * k].to_vec();
            row_data.resize(padded_k, 0.0);
            let dst = &mut data[row * bpr..(row + 1) * bpr];
            <BlockQ8_0 as GgmlType>::from_float(&row_data, dst);
        }
        let w = Q8Weights { n, k, blocks_per_row: bpr, padded_k, data };

        let x: Vec<f32> = (0..k).map(|i| ((i * 3) % 50) as f32).collect();

        // F32 reference dot product.
        let expected: Vec<f32> = (0..n)
            .map(|row| {
                let w_row = &f32_w[row * k..(row + 1) * k];
                w_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum()
            })
            .collect();

        let mut ws = Q8Workspace::new();
        let y = w.gemv(&x, &mut ws);
        for j in 0..n {
            let diff = (y[j] - expected[j]).abs();
            let rel = diff / expected[j].abs().max(1e-6);
            assert!(
                rel < 0.05,
                "col {j}: expected={} q8={} rel_err={}",
                expected[j],
                y[j],
                rel
            );
        }
    }
}
