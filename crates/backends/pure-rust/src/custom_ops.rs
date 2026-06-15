//! Custom f32-slice ops replacing candle Tensor dispatch for hot-path
//! operations: RMS norm, SiLU, and scaled dot-product attention.
//!
//! These operate on raw `&[f32]` slices and produce `Vec<f32>` outputs,
//! then wrap results back into candle Tensors. This avoids Tensor dispatch
//! overhead (reshape, permute, narrow, etc.) for the hot path.

use candle_core::{Device, Tensor};

// ── RMS Norm ───────────────────────────────────────────────────────────

/// RMS Normalization on a flat f32 slice.
///
/// `x`: `[n]` — input activations.
/// `weight`: `[n]` — learned scale.
/// `eps`: f64 — small constant for numerical stability.
///
/// Returns `[n]` f32: `y[i] = x[i] * w[i] / sqrt(mean(x²) + eps)`
pub fn rms_norm_f32(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let n = x.len();
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv_rms = (1.0 / ((mean_sq as f64 + eps).sqrt())) as f32;
    x.iter()
        .zip(weight.iter())
        .map(|(&xv, &wv)| xv * inv_rms * wv)
        .collect()
}

/// In-place RMS Normalization on a flat f32 slice.
///
/// `x` is modified in-place: `x[i] *= w[i] * inv_rms`.
/// `weight`: `[n]` — learned scale.
pub fn rms_norm_f32_inplace(x: &mut [f32], weight: &[f32], eps: f64) {
    let n = x.len();
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv_rms = (1.0 / ((mean_sq as f64 + eps).sqrt())) as f32;
    for i in 0..n {
        x[i] *= inv_rms * weight[i];
    }
}

/// Tensor wrapper: applies RMS norm without going through candle's RmsNorm.
///
/// `x`: `[..., d_model]` — any rank.
/// `weight`: `[d_model]` — norm weight.
pub fn rms_norm_tensor(x: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let x_slice = x.flatten_all()?.to_vec1::<f32>()?;
    let w_slice = weight.flatten_all()?.to_vec1::<f32>()?;
    let y = rms_norm_f32(&x_slice, &w_slice, eps);
    Tensor::from_slice(&y, x.dims(), x.device())
}

// ── Per-Head RMS Norm ──────────────────────────────────────────────────

/// RMS Normalization applied per-head to a multi-head attention tensor.
///
/// `x`: `[n_heads * head_dim]` — flattened multi-head query or key.
/// `weight`: `[head_dim]` — norm weight (shared across heads).
/// `eps`: f64 — small constant.
/// Returns `[n_heads * head_dim]` f32.
pub fn per_head_rms_norm_f32(x: &[f32], weight: &[f32], n_heads: usize, head_dim: usize, eps: f64) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for h in 0..n_heads {
        let start = h * head_dim;
        let slice = &x[start..start + head_dim];
        let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / head_dim as f32;
        let inv_rms = (1.0 / ((mean_sq as f64 + eps).sqrt())) as f32;
        for d in 0..head_dim {
            out.push(slice[d] * inv_rms * weight[d]);
        }
    }
    out
}

/// Tensor wrapper: applies per-head RMS norm.
///
/// `x`: `[B, n_heads, 1, head_dim]`
/// `weight`: `[head_dim]` norm weight.
pub fn per_head_rms_norm_tensor(x: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let shape = x.dims();
    let n_heads = shape[1];
    let head_dim = shape[3];
    let x_slice = x.flatten_all()?.to_vec1::<f32>()?;
    let w_slice = weight.flatten_all()?.to_vec1::<f32>()?;
    let y = per_head_rms_norm_f32(&x_slice, &w_slice, n_heads, head_dim, eps);
    Tensor::from_slice(&y, shape, x.device())
}

// ── 1D NEOX RoPE ───────────────────────────────────────────────────────

/// Apply 1D NEOX rotary position encoding on a flat f32 slice.
///
/// `x`: `[n_heads * head_dim]` — flattened query/key after reshape.
/// `cos`: `[head_dim]` — cosine table for the current position.
/// `sin`: `[head_dim]` — sine table for the current position.
/// Returns `[n_heads * head_dim]` f32.
pub fn rope_f32(x: &[f32], cos: &[f32], sin: &[f32], n_heads: usize, head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = Vec::with_capacity(x.len());
    for h in 0..n_heads {
        let off = h * head_dim;
        for d in 0..half {
            let x1 = x[off + d];
            let x2 = x[off + d + half];
            out.push(x1 * cos[d] - x2 * sin[d]);
        }
        for d in 0..half {
            let x1 = x[off + d];
            let x2 = x[off + d + half];
            out.push(x2 * cos[d] + x1 * sin[d]);
        }
    }
    out
}

/// Tensor wrapper: applies 1D NEOX RoPE.
///
/// `x`: `[B, n_heads, 1, head_dim]`
/// `cos`: `[1, 1, 1, head_dim]`
/// `sin`: `[1, 1, 1, head_dim]`
pub fn rope_tensor(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let shape = x.dims();
    let n_heads = shape[1];
    let head_dim = shape[3];
    let x_slice = x.flatten_all()?.to_vec1::<f32>()?;
    let cos_slice = cos.flatten_all()?.to_vec1::<f32>()?;
    let sin_slice = sin.flatten_all()?.to_vec1::<f32>()?;
    let y = rope_f32(&x_slice, &cos_slice, &sin_slice, n_heads, head_dim);
    Tensor::from_slice(&y, shape, x.device())
}

// ── SiLU (Sigmoid Linear Unit) ─────────────────────────────────────────

/// SiLU activation on a flat f32 slice.
///
/// `x`: `[n]` — input.
/// Returns `[n]` f32: `y[i] = x[i] * sigmoid(x[i])`
pub fn silu_f32(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| v * (1.0 / (1.0 + (-v as f64).exp())) as f32)
        .collect()
}

/// Tensor wrapper: applies SiLU without going through candle_nn::ops::silu.
pub fn silu_tensor(x: &Tensor) -> candle_core::Result<Tensor> {
    let x_slice = x.flatten_all()?.to_vec1::<f32>()?;
    let y = silu_f32(&x_slice);
    Tensor::from_slice(&y, x.dims(), x.device())
}

// ── Scaled Dot-Product Attention ───────────────────────────────────────

/// Scaled dot-product attention on raw f32 slices (single query, KV cache).
///
/// Single-batch, single-query attention with GQA support.
///
/// # Layout
/// - `q`: `[n_heads, head_dim]` — query.
/// - `k`: `[n_kv_heads, kv_len, head_dim]` — full key cache (f32),
///   with `head_stride` positions between heads (allows pre-allocated flat buffers).
/// - `v`: `[n_kv_heads, kv_len, head_dim]` — full value cache (f32).
/// - `n_heads`: number of query heads.
/// - `n_kv_heads`: number of key/value heads.
/// - `kv_len`: number of cached positions.
/// - `head_dim`: dimension per head.
/// - `head_stride`: number of positions allocated per head in the flat buffer
///   (≥ kv_len). For a compact buffer `head_stride = kv_len`; for a
///   pre-allocated buffer `head_stride = max_seq_len`.
///
/// Returns `[n_heads * head_dim]` f32 — attention output.
pub fn attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    kv_len: usize,
    head_dim: usize,
    head_stride: usize,
) -> Vec<f32> {
    let n_repeat = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; n_heads * head_dim];
    let hd = head_dim;

    for h in 0..n_heads {
        let kv_h = h / n_repeat;
        let q_off = h * hd;

        // scores[h, t] = q[h,:] · k[kv_h, t, :] * scale  for t=0..kv_len
        // head_stride replaces kv_len in the per-head base offset, allowing
        // pre-allocated flat buffers where each head has head_stride slots.
        let k_base = kv_h * head_stride * hd;
        let mut max_s = f32::NEG_INFINITY;
        let mut scores = vec![0.0f32; kv_len];
        for t in 0..kv_len {
            let k_off = k_base + t * hd;
            let dot = (0..hd).fold(0.0f32, |acc, d| acc + q[q_off + d] * k[k_off + d]);
            let s = dot * scale;
            scores[t] = s;
            if s > max_s {
                max_s = s;
            }
        }
        // Softmax: subtract max, exp, normalize
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        let inv_sum = 1.0 / sum;

        // Weighted sum of V
        let v_base = kv_h * head_stride * hd;
        for d in 0..hd {
            let acc = (0..kv_len).fold(0.0f32, |acc, t| {
                acc + scores[t] * v[v_base + t * hd + d]
            });
            output[q_off + d] = acc * inv_sum;
        }
    }
    output
}

/// Tensor wrapper for attention_f32 (GQA-aware, no repeat_kv needed).
///
/// # Inputs
/// - `q`: `[1, n_heads, 1, head_dim]` — query (after per-head norm and RoPE).
/// - `k_cache`: `[1, n_kv_heads, kv_len, head_dim]` — full KV cache (not repeated).
/// - `v_cache`: `[1, n_kv_heads, kv_len, head_dim]` — full KV cache.
///
/// Returns `[1, n_heads, 1, head_dim]` — attention output.
pub fn attention_gqa_tensor(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
) -> candle_core::Result<Tensor> {
    let batch = q.dim(0)?;
    let n_heads = q.dim(1)?;
    let head_dim = q.dim(3)?;
    let kv_len = k_cache.dim(2)?;
    let n_kv_heads = k_cache.dim(1)?;

    let q_slice = q.flatten_all()?.to_vec1::<f32>()?;
    let k_slice = k_cache.flatten_all()?.to_vec1::<f32>()?;
    let v_slice = v_cache.flatten_all()?.to_vec1::<f32>()?;

    let out = attention_f32(
        &q_slice,
        &k_slice,
        &v_slice,
        n_heads,
        n_kv_heads,
        kv_len,
        head_dim,
        kv_len, // head_stride = kv_len (compact layout in concatenated tensors)
    );

    Tensor::from_slice(&out, (batch, n_heads, 1, head_dim), q.device())
}

/// Flat-buffer variant of attention_gqa that takes raw f32 slices for K/V
/// instead of Tensors, avoiding the to_vec1 copy of the entire KV cache.
///
/// # Inputs
/// - `q`: `[1, n_heads, 1, head_dim]` — query tensor.
/// - `k_flat`, `v_flat`: flat f32 buffers with layout `[n_kv_heads, head_stride, head_dim]`.
/// - `kv_len`: actual number of valid positions (≤ head_stride).
/// - `head_stride`: allocated positions per head in the flat buffer (= max_seq_len).
///
/// Returns `[1, n_heads, 1, head_dim]` — attention output.
pub fn attention_gqa_flat(
    q: &Tensor,
    k_flat: &[f32],
    v_flat: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    kv_len: usize,
    head_dim: usize,
    head_stride: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let q_slice = q.flatten_all()?.to_vec1::<f32>()?;

    let out = attention_f32(
        &q_slice,
        k_flat,
        v_flat,
        n_heads,
        n_kv_heads,
        kv_len,
        head_dim,
        head_stride,
    );

    Tensor::from_slice(&out, (1, n_heads, 1, head_dim), device)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_core::Module;

    #[test]
    fn test_rms_norm_versus_candle() {
        let dev = Device::Cpu;
        let n = 64usize;
        let eps = 1e-6;

        let x_data: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 50) as f32 / 10.0).collect();
        let w_data: Vec<f32> = (0..n).map(|i| 1.0 + ((i * 3) % 10) as f32 / 20.0).collect();

        // Our custom
        let y_custom = rms_norm_f32(&x_data, &w_data, eps);

        // Candle
        let x_t = Tensor::from_slice(&x_data, (1, 1, n), &dev).unwrap();
        let w_t = Tensor::from_slice(&w_data, (n,), &dev).unwrap();
        let norm = candle_nn::RmsNorm::new(w_t, eps);
        let y_candle = norm.forward(&x_t).unwrap();
        let y_candle_vec: Vec<f32> = y_candle.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(y_custom.len(), y_candle_vec.len());
        for i in 0..n {
            let diff = (y_custom[i] - y_candle_vec[i]).abs();
            assert!(
                diff < 1e-5,
                "rms_norm mismatch at [{i}]: custom={} candle={} diff={}",
                y_custom[i],
                y_candle_vec[i],
                diff,
            );
        }
    }

    #[test]
    fn test_silu_versus_candle() {
        let dev = Device::Cpu;
        let n = 32usize;
        let x_data: Vec<f32> = (0..n).map(|i| ((i as i32 - 16) * 3) as f32 / 10.0).collect();

        let y_custom = silu_f32(&x_data);
        let x_t = Tensor::from_slice(&x_data, (n,), &dev).unwrap();
        let y_candle = candle_nn::ops::silu(&x_t).unwrap();
        let y_candle_vec: Vec<f32> = y_candle.flatten_all().unwrap().to_vec1().unwrap();

        for i in 0..n {
            let diff = (y_custom[i] - y_candle_vec[i]).abs();
            assert!(
                diff < 1e-5,
                "silu mismatch at [{i}]: custom={} candle={} diff={}",
                y_custom[i],
                y_candle_vec[i],
                diff,
            );
        }
    }
}
