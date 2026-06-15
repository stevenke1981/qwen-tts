//! Custom f32-slice ops replacing candle Tensor dispatch for hot-path
//! operations: RMS norm, SiLU, and scaled dot-product attention.
//!
//! These operate on raw `&[f32]` slices and produce `Vec<f32>` outputs,
//! then wrap results back into candle Tensors. This avoids Tensor dispatch
//! overhead (reshape, permute, narrow, etc.) for the hot path.

use candle_core::Tensor;

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
/// - `k`: `[n_kv_heads, kv_len, head_dim]` — full key cache (f32).
/// - `v`: `[n_kv_heads, kv_len, head_dim]` — full value cache (f32).
/// - `n_heads`: number of query heads.
/// - `n_kv_heads`: number of key/value heads.
/// - `kv_len`: number of cached positions.
/// - `head_dim`: dimension per head.
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
) -> Vec<f32> {
    let n_repeat = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; n_heads * head_dim];
    let hd = head_dim;

    for h in 0..n_heads {
        let kv_h = h / n_repeat;
        let q_off = h * hd;

        // scores[h, t] = q[h,:] · k[kv_h, t, :] * scale  for t=0..kv_len
        let k_base = kv_h * kv_len * hd;
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
        let v_base = kv_h * kv_len * hd;
        for d in 0..hd {
            let acc = (0..kv_len).fold(0.0f32, |acc, t| {
                acc + scores[t] * v[v_base + t * hd + d]
            });
            output[q_off + d] = acc * inv_sum;
        }
    }
    output
}

/// Tensor wrapper for attention_f32.
///
/// Inputs are candle Tensors with standard shapes.
pub fn attention_tensor(
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
    );

    Tensor::from_slice(&out, (batch, n_heads, 1, head_dim), q.device())
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
