//! Pre-transformer: 8-layer Qwen3-style decoder transformer for the 12Hz codec.
//!
//! Architecture (C-first throughout [C, T]):
//! ```text
//! input [1024, T]
//!   → input_proj (Linear 1024 → 512, with bias)
//!   → 8× DecoderLayer:
//!       → RMSNorm → Q/K/V projections → RoPE (Neox)
//!       → Sliding window causal self-attention (window=72)
//!       → LayerScale(attn_scale) + residual
//!       → RMSNorm → SwiGLU MLP (gate/up/down)
//!       → LayerScale(mlp_scale) + residual
//!   → final RMSNorm
//!   → output_proj (Linear 512 → 1024, with bias)
//!   → output [1024, T]
//! ```
//!
//! Hyperparameters:
//! - hidden_size = 512
//! - latent_dim  = 1024
//! - num_layers  = 8
//! - num_attention_heads = 16
//! - num_kv_heads = 16 (MHA, no GQA)
//! - head_dim = 64
//! - intermediate_size (FFN) = 1024
//! - sliding_window = 72
//! - rope_theta = 10000.0
//! - rms_norm_eps = 1e-5
//!
//! All weights are Q8_0 quantized in the GGUF, dequantized to f32 at load time.

use crate::gguf::GgufFile;
use crate::q8_0::q8_0_matmul_f32;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HIDDEN_SIZE: usize = 512;
const LATENT_DIM: usize = 1024;
const NUM_LAYERS: usize = 8;
const NUM_HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const INTERMEDIATE_SIZE: usize = 1024;
const SLIDING_WINDOW: usize = 72;
const ROPE_THETA: f32 = 10000.0;
const RMS_NORM_EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Helper: RMS Norm
// ---------------------------------------------------------------------------

/// Apply RMSNorm to a tensor `[C, T]` in-place.
fn rms_norm_cfirst(data: &mut [f32], c: usize, t: usize, weight: &[f32], eps: f32) {
    for ti in 0..t {
        let mut sum_sq = 0.0f32;
        for ch in 0..c {
            let val = data[ch * t + ti];
            sum_sq += val * val;
        }
        let rms = (sum_sq / c as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for ch in 0..c {
            let idx = ch * t + ti;
            data[idx] = data[idx] * inv_rms * weight[ch];
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: matmul for C-first layout
// ---------------------------------------------------------------------------

/// Compute `y = W @ x + b` where:
/// - `w`: `[C_out, C_in]` row-major
/// - `x`: `[C_in, T]` C-first
/// - `b`: `[C_out]`
/// - Returns: `[C_out, T]` C-first
fn linear_forward(w: &[f32], x: &[f32], c_in: usize, c_out: usize, t: usize, b: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0.0f32; c_out * t];
    for oc in 0..c_out {
        let out_base = oc * t;
        let w_base = oc * c_in;
        for ic in 0..c_in {
            let w_val = w[w_base + ic];
            let in_base = ic * t;
            for ti in 0..t {
                y[out_base + ti] += w_val * x[in_base + ti];
            }
        }
        if let Some(bias) = b {
            let bval = bias[oc];
            for ti in 0..t {
                y[out_base + ti] += bval;
            }
        }
    }
    y
}

// ---------------------------------------------------------------------------
// Helper: sliding window causal mask builder
// ---------------------------------------------------------------------------

/// Build additive causal sliding-window mask of shape `[T, T]`.
/// `mask[q, k] = 0.0` if `0 <= q - k < window`, else `-f32::INFINITY`.
fn build_sliding_window_mask(t: usize, window: usize) -> Vec<f32> {
    let mut mask = vec![-f32::INFINITY; t * t];
    for q in 0..t {
        let k_min = if q >= window { q - window + 1 } else { 0 };
        for k in k_min..=q {
            mask[q * t + k] = 0.0;
        }
    }
    mask
}

// ---------------------------------------------------------------------------
// Helper: RoPE (Neox style)
// ---------------------------------------------------------------------------

/// Apply Neox-style rotary position embeddings to Q/K tensors in-place.
///
/// `qk`: `[hd * n_heads, T]` C-first (combined hd and n_heads in leading dim)
/// `hd`: head dimension
/// `n_heads`: number of heads
/// `t`: time steps
/// `theta`: RoPE theta (10000.0)
fn apply_rope_neox(qk: &mut [f32], hd: usize, n_heads: usize, t: usize, theta: f32) {
    // For each head, position, and half-dim pair
    for h in 0..n_heads {
        let head_base = h * hd;
        for pos in 0..t {
            for i in (0..hd).step_by(2) {
                let dim = head_base + i;
                let freq = 1.0 / theta.powi((i / 2) as i32);
                let cos_v = (pos as f32 * freq).cos();
                let sin_v = (pos as f32 * freq).sin();

                let idx0 = dim * t + pos;
                let idx1 = (dim + 1) * t + pos;

                let x0 = qk[idx0];
                let x1 = if i + 1 < hd { qk[idx1] } else { 0.0 };

                qk[idx0] = x0 * cos_v - x1 * sin_v;
                if i + 1 < hd {
                    qk[idx1] = x1 * cos_v + x0 * sin_v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Silu activation
// ---------------------------------------------------------------------------

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// DecoderLayer
// ---------------------------------------------------------------------------

/// One decoder transformer layer.
///
/// Attention:
/// - RMSNorm → Q/K/V proj (Q8_0) → RoPE → sliding window causal attn → O proj → LayerScale
///
/// MLP:
/// - RMSNorm → SwiGLU: SiLU(Gate @ x) * (Up @ x) → Down proj → LayerScale
pub(crate) struct DecoderLayer {
    // Attention
    attn_norm_weight: Vec<f32>,    // [512]
    q_proj_w: Vec<u8>,             // Q8_0 raw bytes [1024, 512]
    k_proj_w: Vec<u8>,             // Q8_0 raw bytes [1024, 512]
    v_proj_w: Vec<u8>,             // Q8_0 raw bytes [1024, 512]
    o_proj_w: Vec<u8>,             // Q8_0 raw bytes [512, 1024]
    attn_scale: Vec<f32>,          // [512] LayerScale
    // MLP
    ffn_norm_weight: Vec<f32>,     // [512]
    ffn_gate_w: Vec<u8>,           // Q8_0 raw bytes [1024, 512]
    ffn_up_w: Vec<u8>,             // Q8_0 raw bytes [1024, 512]
    ffn_down_w: Vec<u8>,           // Q8_0 raw bytes [512, 1024]
    mlp_scale: Vec<f32>,           // [512] LayerScale

    // Dimensions (stored for convenience)
    hidden: usize,
    n_heads: usize,
    hd: usize,
    ffn_dim: usize,
}

impl DecoderLayer {
    fn load_from_gguf(gguf: &mut GgufFile, layer_idx: usize) -> Result<Self, String> {
        let prefix = format!("tok_dec.pre_tfm.blk.{layer_idx}");
        let hidden = HIDDEN_SIZE;
        let _qk_dim = NUM_HEADS * HEAD_DIM; // 1024
        let ffn_dim = INTERMEDIATE_SIZE;

        // Attention norm (RMSNorm weight, f32)
        let attn_norm_weight = gguf.read_tensor_f32(&format!("{prefix}.attn_norm.weight"))?;
        assert_eq!(attn_norm_weight.len(), hidden, "{prefix}.attn_norm.weight size");

        // Q projections (Q8_0, raw)
        let q_proj_w = gguf.read_tensor_q8_0_raw(&format!("{prefix}.attn_q.weight"))?;
        let k_proj_w = gguf.read_tensor_q8_0_raw(&format!("{prefix}.attn_k.weight"))?;
        let v_proj_w = gguf.read_tensor_q8_0_raw(&format!("{prefix}.attn_v.weight"))?;
        let o_proj_w = gguf.read_tensor_q8_0_raw(&format!("{prefix}.attn_output.weight"))?;

        // attn_scale (LayerScale, f32)
        let attn_scale = gguf.read_tensor_f32(&format!("{prefix}.attn_scale"))?;
        assert_eq!(attn_scale.len(), hidden, "{prefix}.attn_scale size");

        // FFN norm
        let ffn_norm_weight = gguf.read_tensor_f32(&format!("{prefix}.ffn_norm.weight"))?;
        assert_eq!(ffn_norm_weight.len(), hidden, "{prefix}.ffn_norm.weight size");

        // FFN projections (Q8_0, raw)
        let ffn_gate_w = gguf.read_tensor_q8_0_raw(&format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up_w = gguf.read_tensor_q8_0_raw(&format!("{prefix}.ffn_up.weight"))?;
        let ffn_down_w = gguf.read_tensor_q8_0_raw(&format!("{prefix}.ffn_down.weight"))?;

        // mlp_scale (LayerScale, f32)
        let mlp_scale = gguf.read_tensor_f32(&format!("{prefix}.ffn_scale"))?;
        assert_eq!(mlp_scale.len(), hidden, "{prefix}.ffn_scale size");

        Ok(Self {
            attn_norm_weight,
            q_proj_w,
            k_proj_w,
            v_proj_w,
            o_proj_w,
            attn_scale,
            ffn_norm_weight,
            ffn_gate_w,
            ffn_up_w,
            ffn_down_w,
            mlp_scale,
            hidden,
            n_heads: NUM_HEADS,
            hd: HEAD_DIM,
            ffn_dim,
        })
    }

    /// Forward pass for one layer.
    ///
    /// Input: `[hidden, T]` C-first
    /// Output: `[hidden, T]` C-first
    fn forward(&self, input: &[f32], t: usize, mask: &[f32]) -> Vec<f32> {
        let hidden = self.hidden;
        let qk_dim = self.n_heads * self.hd;
        let ffn_dim = self.ffn_dim;

        // ------------------------------------------------------------------
        // Attention block
        // ------------------------------------------------------------------

        // 1. RMSNorm
        let mut ln = input.to_vec();
        rms_norm_cfirst(&mut ln, hidden, t, &self.attn_norm_weight, RMS_NORM_EPS);

        // 2. Q/K/V projections (Q8_0 matmul)
        // Q: [1024, T], K: [1024, T], V: [1024, T]
        let q = q8_0_matmul_f32(&self.q_proj_w, qk_dim, hidden, &ln, t);
        let k = q8_0_matmul_f32(&self.k_proj_w, qk_dim, hidden, &ln, t);
        let v = q8_0_matmul_f32(&self.v_proj_w, qk_dim, hidden, &ln, t);

        // 3. Apply RoPE to Q and K in-place
        let mut q = q;
        let mut k = k;
        apply_rope_neox(&mut q, self.hd, self.n_heads, t, ROPE_THETA);
        apply_rope_neox(&mut k, self.hd, self.n_heads, t, ROPE_THETA);

        // 4. Sliding window causal self-attention
        // For each head:
        //   score[t_q, t_k] = sum over hd of Q_h[hd, t_q] * K_h[hd, t_k]
        //   attn_h[hd, t_q] = sum over t_k of V_h[hd, t_k] * softmax_score[t_q, t_k]
        let mut attn = vec![0.0f32; qk_dim * t]; // [1024, T]

        // Pre-compute scale factor
        let scale = 1.0 / (self.hd as f32).sqrt();

        for h in 0..self.n_heads {
            let head_base = h * self.hd;

            // Compute attention scores for this head: score[t_q, t_k]
            // Using sliding window mask
            let mut scores = vec![-f32::INFINITY; t * t];

            for tq in 0..t {
                let k_min = if tq >= SLIDING_WINDOW { tq - SLIDING_WINDOW + 1 } else { 0 };
                for tk in k_min..=tq {
                    let mut s = 0.0f32;
                    for d in 0..self.hd {
                        let idx = head_base + d;
                        s += q[idx * t + tq] * k[idx * t + tk];
                    }
                    scores[tq * t + tk] = s * scale + mask[tq * t + tk];
                }
            }

            // Softmax over tk dim for each tq
            // Use max-subtraction for numerical stability
            for tq in 0..t {
                let k_min = if tq >= SLIDING_WINDOW { tq - SLIDING_WINDOW + 1 } else { 0 };
                let row_start = tq * t;
                // Find max
                let mut max_val = -f32::INFINITY;
                for tk in k_min..=tq {
                    let v = scores[row_start + tk];
                    if v > max_val {
                        max_val = v;
                    }
                }
                // Exp sum
                let mut sum_exp = 0.0f32;
                for tk in k_min..=tq {
                    scores[row_start + tk] = (scores[row_start + tk] - max_val).exp();
                    sum_exp += scores[row_start + tk];
                }
                let inv_sum = 1.0 / sum_exp;
                for tk in k_min..=tq {
                    scores[row_start + tk] *= inv_sum;
                }
            }

            // Attention output: attn_h[hd, t_q] = sum over tk of V_h[hd, tk] * score[t_q, tk]
            for d in 0..self.hd {
                let dim_idx = head_base + d;
                for tq in 0..t {
                    let mut s = 0.0f32;
                    let k_min = if tq >= SLIDING_WINDOW { tq - SLIDING_WINDOW + 1 } else { 0 };
                    for tk in k_min..=tq {
                        s += v[dim_idx * t + tk] * scores[tq * t + tk];
                    }
                    attn[dim_idx * t + tq] = s;
                }
            }
        }

        // 5. Output projection (Q8_0): [512, T] = W_o @ [1024, T]
        let attn_out = q8_0_matmul_f32(&self.o_proj_w, hidden, qk_dim, &attn, t);

        // 6. LayerScale + residual
        let mut x = vec![0.0f32; hidden * t];
        for ch in 0..hidden {
            let out_base = ch * t;
            let scale_val = self.attn_scale[ch];
            for ti in 0..t {
                x[out_base + ti] = input[out_base + ti] + attn_out[out_base + ti] * scale_val;
            }
        }

        // ------------------------------------------------------------------
        // MLP block
        // ------------------------------------------------------------------

        // 7. RMSNorm
        let mut ln2 = x.clone();
        rms_norm_cfirst(&mut ln2, hidden, t, &self.ffn_norm_weight, RMS_NORM_EPS);

        // 8. SwiGLU: SiLU(Gate @ x) * (Up @ x), then Down
        // Gate: [1024, T], Up: [1024, T], Down: [512, 1024] × result
        let gate = q8_0_matmul_f32(&self.ffn_gate_w, ffn_dim, hidden, &ln2, t);
        let up = q8_0_matmul_f32(&self.ffn_up_w, ffn_dim, hidden, &ln2, t);

        // SiLU(gate) * up
        let mut act = gate;
        for i in 0..ffn_dim * t {
            act[i] = silu(act[i]) * up[i];
        }

        // Down projection: [512, T] = W_down @ [1024, T]
        let mlp_out = q8_0_matmul_f32(&self.ffn_down_w, hidden, ffn_dim, &act, t);

        // 9. LayerScale + residual
        for ch in 0..hidden {
            let out_base = ch * t;
            let scale_val = self.mlp_scale[ch];
            for ti in 0..t {
                x[out_base + ti] += mlp_out[out_base + ti] * scale_val;
            }
        }

        x
    }
}

// ---------------------------------------------------------------------------
// PreTransformer
// ---------------------------------------------------------------------------

/// 8-layer pre-transformer: input_proj → 8× DecoderLayer → final norm → output_proj.
pub struct PreTransformer {
    pub input_proj_w: Vec<f32>,      // [512, 1024] f32
    pub input_proj_b: Vec<f32>,      // [512] f32
    pub(crate) layers: Vec<DecoderLayer>,    // 8 layers
    pub final_norm_w: Vec<f32>,      // [512] f32 (RMSNorm)
    pub output_proj_w: Vec<f32>,     // [1024, 512] f32
    pub output_proj_b: Vec<f32>,     // [1024] f32
    pub hidden: usize,
    pub latent: usize,
}

impl PreTransformer {
    /// Load all transformer weights from the GGUF file.
    pub fn from_gguf(gguf: &mut GgufFile) -> Result<Self, String> {
        // Input projection (f32)
        let input_proj_w = gguf.read_tensor_f32("tok_dec.pre_tfm.input_proj.weight")?;
        let input_proj_b = gguf.read_tensor_f32("tok_dec.pre_tfm.input_proj.bias")?;
        assert_eq!(input_proj_w.len(), HIDDEN_SIZE * LATENT_DIM, "input_proj.weight size");
        assert_eq!(input_proj_b.len(), HIDDEN_SIZE, "input_proj.bias size");

        // 8 layers
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            layers.push(DecoderLayer::load_from_gguf(gguf, i)?);
        }

        // Final norm (f32)
        let final_norm_w = gguf.read_tensor_f32("tok_dec.pre_tfm.norm.weight")?;
        assert_eq!(final_norm_w.len(), HIDDEN_SIZE, "final_norm.weight size");

        // Output projection (f32)
        let output_proj_w = gguf.read_tensor_f32("tok_dec.pre_tfm.output_proj.weight")?;
        let output_proj_b = gguf.read_tensor_f32("tok_dec.pre_tfm.output_proj.bias")?;
        assert_eq!(output_proj_w.len(), LATENT_DIM * HIDDEN_SIZE, "output_proj.weight size");
        assert_eq!(output_proj_b.len(), LATENT_DIM, "output_proj.bias size");

        Ok(Self {
            input_proj_w,
            input_proj_b,
            layers,
            final_norm_w,
            output_proj_w,
            output_proj_b,
            hidden: HIDDEN_SIZE,
            latent: LATENT_DIM,
        })
    }

    /// Full transformer forward pass.
    ///
    /// Input:  `[latent_dim=1024, T]` C-first
    /// Output: `[latent_dim=1024, T]` C-first
    pub fn forward(&self, input: &[f32], t: usize) -> Vec<f32> {
        // 1. input_proj: [1024, T] → [512, T]
        let mask = build_sliding_window_mask(t, SLIDING_WINDOW);

        let mut x = linear_forward(
            &self.input_proj_w, input,
            self.latent, self.hidden, t,
            Some(&self.input_proj_b),
        );

        // 2. 8 layers
        for layer in &self.layers {
            x = layer.forward(&x, t, &mask);
        }

        // 3. Final RMSNorm
        rms_norm_cfirst(&mut x, self.hidden, t, &self.final_norm_w, RMS_NORM_EPS);

        // 4. output_proj: [512, T] → [1024, T]
        linear_forward(
            &self.output_proj_w, &x,
            self.hidden, self.latent, t,
            Some(&self.output_proj_b),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use std::path::PathBuf;

    fn codec_gguf_path() -> PathBuf {
        let md = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let ws = md.parent().and_then(|p| p.parent()).unwrap();
        ws.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
    }

    #[test]
    fn pre_transformer_load_weights() {
        let path = codec_gguf_path();
        assert!(path.exists(), "GGUF not found");
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let tr = PreTransformer::from_gguf(&mut gguf).expect("load transformer");

        assert_eq!(tr.layers.len(), NUM_LAYERS);

        // Verify input_proj
        assert_eq!(tr.input_proj_w.len(), HIDDEN_SIZE * LATENT_DIM);
        assert_eq!(tr.input_proj_b.len(), HIDDEN_SIZE);

        // Verify first layer weights exist
        let l0 = &tr.layers[0];
        assert_eq!(l0.attn_norm_weight.len(), HIDDEN_SIZE);
        assert_eq!(l0.attn_scale.len(), HIDDEN_SIZE);
        assert_eq!(l0.ffn_norm_weight.len(), HIDDEN_SIZE);
        assert_eq!(l0.mlp_scale.len(), HIDDEN_SIZE);
        assert_eq!(l0.q_proj_w.len(), NUM_HEADS * HEAD_DIM * HIDDEN_SIZE / 32 * 34); // Q8_0 raw size
        assert_eq!(l0.ffn_gate_w.len(), INTERMEDIATE_SIZE * HIDDEN_SIZE / 32 * 34);

        // Final norm and output proj
        assert_eq!(tr.final_norm_w.len(), HIDDEN_SIZE);
        assert_eq!(tr.output_proj_w.len(), LATENT_DIM * HIDDEN_SIZE);
        assert_eq!(tr.output_proj_b.len(), LATENT_DIM);

        println!("PreTransformer loaded: {} layers, hidden {}, latent {}",
            NUM_LAYERS, HIDDEN_SIZE, LATENT_DIM);
    }

    #[test]
    fn pre_transformer_forward_shape() {
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let tr = PreTransformer::from_gguf(&mut gguf).expect("load transformer");

        for t in [1, 2, 5] {
            let input = vec![0.01f32; LATENT_DIM * t];
            let output = tr.forward(&input, t);
            assert_eq!(output.len(), LATENT_DIM * t,
                "t={t} output shape should be [{}, {t}]", LATENT_DIM);
            assert!(output.iter().all(|v| v.is_finite()), "t={t} non-finite");
            let has_signal = output.iter().any(|&v| v.abs() > 1e-6);
            assert!(has_signal, "t={t} output all zero");
            println!("Transformer t={t}: [{}, {t}] → [{}, {t}] mean={:.6}",
                LATENT_DIM, LATENT_DIM,
                output.iter().sum::<f32>() / output.len() as f32
            );
        }
    }

    #[test]
    fn rms_norm_works() {
        let c = 4;
        let t = 2;
        // C-first layout [C, T] = [4, 2]:
        //   data[ch * t + ti]
        //   ti=0: [ch0=1, ch1=3, ch2=5, ch3=7]  at indices [0,2,4,6]
        //   ti=1: [ch0=2, ch1=4, ch2=6, ch3=8]  at indices [1,3,5,7]
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0; c];

        rms_norm_cfirst(&mut data, c, t, &weight, 1e-5);

        // RMS for ti=0: sqrt((1+9+25+49)/4) = sqrt(21)
        let rms0 = ((1.0f32 + 9.0 + 25.0 + 49.0) / 4.0).sqrt();
        assert!((data[0] - 1.0 / rms0).abs() < 1e-5);
        assert!((data[6] - 7.0 / rms0).abs() < 1e-5);

        // RMS for ti=1: sqrt((4+16+36+64)/4) = sqrt(30)
        let rms1 = ((4.0f32 + 16.0 + 36.0 + 64.0) / 4.0).sqrt();
        assert!((data[1] - 2.0 / rms1).abs() < 1e-5);
        assert!((data[7] - 8.0 / rms1).abs() < 1e-5);
    }

    #[test]
    fn silu_activation() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7310586).abs() < 1e-5);
        assert!((silu(-1.0) - (-0.2689414)).abs() < 1e-5);
    }

    #[test]
    fn sliding_window_mask() {
        let window = 3;
        let t = 6;
        let mask = build_sliding_window_mask(t, window);

        // q=0: k_min=0, mask[0,0] = 0
        assert_eq!(mask[0 * t + 0], 0.0);
        // q=2: k_min=0, mask[2,0]=0, mask[2,1]=0, mask[2,2]=0
        assert_eq!(mask[2 * t + 0], 0.0);
        assert_eq!(mask[2 * t + 1], 0.0);
        assert_eq!(mask[2 * t + 2], 0.0);
        // q=5: k_min=3 (window-1 from q), mask[5,2] = -inf
        assert_eq!(mask[5 * t + 2], -f32::INFINITY);
        assert_eq!(mask[5 * t + 3], 0.0);
        // q < k should always be -inf (causal)
        assert_eq!(mask[1 * t + 2], -f32::INFINITY);
    }

    #[test]
    fn rope_neox_rotates() {
        // Small test: 2 heads, hd=4, T=3
        let hd = 4;
        let n_heads = 2;
        let t = 3;
        let mut q = vec![0.0f32; hd * n_heads * t];
        // Fill with ones
        for v in q.iter_mut() {
            *v = 1.0;
        }

        apply_rope_neox(&mut q, hd, n_heads, t, 10000.0);

        // After RoPE, norms should still be sqrt(4) = 2.0 for each head-position
        for h in 0..n_heads {
            for pos in 0..t {
                let base = h * hd;
                let mut sq_sum = 0.0f32;
                for d in 0..hd {
                    let v = q[(base + d) * t + pos];
                    sq_sum += v * v;
                }
                let norm = sq_sum.sqrt();
                assert!((norm - (hd as f32).sqrt()).abs() < 1e-5,
                    "head {h} pos {pos}: norm {norm}");
            }
        }
    }

    #[test]
    fn linear_forward_basic() {
        let c_in = 3;
        let c_out = 2;
        let t = 2;
        // Input [3, 2]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Weight [2, 3]
        let w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b = vec![10.0, 20.0];

        let y = linear_forward(&w, &x, c_in, c_out, t, Some(&b));
        assert_eq!(y.len(), c_out * t);

        // y[oc=0, t=0] = 0.1*1 + 0.2*3 + 0.3*5 + 10 = 0.1+0.6+1.5+10 = 12.2
        assert!((y[0] - 12.2).abs() < 1e-5);
        // y[oc=0, t=1] = 0.1*2 + 0.2*4 + 0.3*6 + 10 = 0.2+0.8+1.8+10 = 12.8
        assert!((y[1] - 12.8).abs() < 1e-5);
        // y[oc=1, t=0] = 0.4*1 + 0.5*3 + 0.6*5 + 20 = 0.4+1.5+3.0+20 = 24.9
        assert!((y[c_out * 0 + 2] - 24.9).abs() < 1e-5, "got {}", y[2]);
        // y[oc=1, t=1] = 0.4*2 + 0.5*4 + 0.6*6 + 20 = 0.8+2.0+3.6+20 = 26.4
        assert!((y[3] - 26.4).abs() < 1e-5);
    }
}
