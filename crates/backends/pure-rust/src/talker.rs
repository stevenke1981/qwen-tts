//! Qwen2 talker transformer — loaded from a GGUF file.
//!
//! All weights are dequantized to F32 at load time for simplicity.
//! Uses raw candle `Tensor` matmul for all projections (no QTensor).

use std::fs::File;
use std::path::Path;

use candle_core::quantized::gguf_file::Content;
use candle_core::{Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::RmsNorm;

use crate::config::ModelConfig;
use crate::code_predictor::CodePredictor;

const FN_TOKEN_EMBD: &str = "talker.text_embd.weight";
const FN_OUTPUT: &str = "talker.output.weight";
const FN_OUTPUT_NORM: &str = "talker.output_norm.weight";
const FN_ATTN_NORM: &str = "attn_norm.weight";
const FN_ATTN_Q: &str = "attn_q.weight";
const FN_ATTN_K: &str = "attn_k.weight";
const FN_ATTN_V: &str = "attn_v.weight";
const FN_ATTN_O: &str = "attn_output.weight";
const FN_FFN_NORM: &str = "ffn_norm.weight";
const FN_FFN_GATE: &str = "ffn_gate.weight";
const FN_FFN_UP: &str = "ffn_up.weight";
const FN_FFN_DOWN: &str = "ffn_down.weight";

/// Prefix for talker transformer layer tensors.
const BLK_PREFIX: &str = "talker.blk";

/// Qwen2 talker transformer.
pub struct Talker {
    config: ModelConfig,
    token_embd: Tensor,
    output: Option<Tensor>,
    output_norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    device: Device,
}

struct DecoderLayer {
    attn_norm: RmsNorm,
    attn_q: Tensor,
    attn_k: Tensor,
    attn_v: Tensor,
    attn_o: Tensor,
    ffn_norm: RmsNorm,
    ffn_gate: Tensor,
    ffn_up: Tensor,
    ffn_down: Tensor,
}

/// Load a single tensor from GGUF, dequantized to F32.
fn load_f32_tensor(content: &Content, file: &mut File, name: &str, device: &Device) -> anyhow::Result<Tensor> {
    let qt = content
        .tensor(file, name, device)
        .map_err(|e| anyhow::anyhow!("missing tensor {name}: {e}"))?;
    qt.dequantize(device)
        .map_err(|e| anyhow::anyhow!("dequantize {name}: {e}"))
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

impl Talker {
    /// Load from a GGUF file.
    pub fn from_gguf(path: &Path, device: &Device) -> anyhow::Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| anyhow::anyhow!("failed to open GGUF {path:?}: {e}"))?;

        let content = Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("bad GGUF header: {e}"))?;

        Self::from_content(&content, &mut file, device)
    }

    /// Load from a GGUF file, also extracting the code predictor weights
    /// from the same GGUF file.
    pub fn load_with_predictor(path: &Path, device: &Device) -> anyhow::Result<(Self, CodePredictor)> {
        let mut file = File::open(path)
            .map_err(|e| anyhow::anyhow!("failed to open GGUF {path:?}: {e}"))?;

        let content = Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("bad GGUF header: {e}"))?;

        let talker = Self::from_content(&content, &mut file, device)?;
        let code_predictor = CodePredictor::from_gguf(&content, &mut file, device)?;

        Ok((talker, code_predictor))
    }

    fn from_content(content: &Content, mut file: &mut File, device: &Device) -> anyhow::Result<Self> {
        let cfg = ModelConfig::from_gguf(&content.metadata);

        // Helper bound to content & file (takes device only)
        let mut load = |name: &str| load_f32_tensor(&content, &mut file, name, device);

        let token_embd = load(FN_TOKEN_EMBD)?;

        let output = if content.tensor_infos.contains_key(FN_OUTPUT) {
            Some(load(FN_OUTPUT)?)
        } else {
            None
        };

        let output_norm = RmsNorm::new(load(FN_OUTPUT_NORM)?, cfg.norm_eps);

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let blk = |n: &str| format!("{BLK_PREFIX}.{i}.{n}");
            layers.push(DecoderLayer {
                attn_norm: RmsNorm::new(load(&blk(FN_ATTN_NORM))?, cfg.norm_eps),
                attn_q: load(&blk(FN_ATTN_Q))?,
                attn_k: load(&blk(FN_ATTN_K))?,
                attn_v: load(&blk(FN_ATTN_V))?,
                attn_o: load(&blk(FN_ATTN_O))?,
                ffn_norm: RmsNorm::new(load(&blk(FN_FFN_NORM))?, cfg.norm_eps),
                ffn_gate: load(&blk(FN_FFN_GATE))?,
                ffn_up: load(&blk(FN_FFN_UP))?,
                ffn_down: load(&blk(FN_FFN_DOWN))?,
            });
        }

        Ok(Self {
            config: cfg,
            token_embd,
            output,
            output_norm,
            layers,
            device: device.clone(),
        })
    }

    /// Forward: token IDs → logits for the last position.
    ///
    /// `input_ids`: `[batch, seq_len]`, dtype `u32`.
    pub fn forward(&self, input_ids: &Tensor) -> anyhow::Result<Tensor> {
        let dev = &self.device;
        let cfg = &self.config;
        let batch = input_ids.dims()[0];
        let seq_len = input_ids.dims()[1];

        // --- Embedding lookup (manual) ---
        // token_embd shape: [d_model, vocab_size] or [vocab_size, d_model]
        // We gather embeddings by index.
        let emb_w = &self.token_embd;
        let vocab_size = cfg.vocab_size;
        let d_model = cfg.d_model;

        // Determine layout: [d_model, vocab_size] (transposed) or [vocab_size, d_model]
        let emb_weights = if emb_w.dims()[0] == d_model && emb_w.dims()[1] == vocab_size {
            // Transposed: [d_model, vocab_size] — need to take() first
            emb_w.t()?
        } else {
            emb_w.clone()
        };
        // Now emb_weights: [vocab_size, d_model], use gather
        let mut hidden = emb_weights.gather(input_ids, 0)?;
        // hidden: [batch, seq_len, d_model]
        // Make contiguous for matmul
        hidden = hidden.contiguous()?;

        // --- RoPE precompute ---
        let (cos, sin) = precompute_cos_sin(cfg, seq_len, dev)?;

        // --- Decoder layers ---
        for layer in &self.layers {
            // Pre-attention norm
            let residual = hidden.clone();
            hidden = layer.attn_norm.forward(&hidden)?;

            // Self-attention with GQA
            let n_heads = cfg.n_heads;
            let n_kv = cfg.n_kv_heads;
            let hd = cfg.head_dim();

            // QKV projections: [batch, seq_len, d_model] → [batch, seq_len, dim]
            let q = layer.attn_q.matmul(&hidden)?;
            let k = layer.attn_k.matmul(&hidden)?;
            let v = layer.attn_v.matmul(&hidden)?;

            // Reshape to multi-head: [batch, seq_len, n_heads, hd] → [batch, n_heads, seq_len, hd]
            let q = q.reshape((batch, seq_len, n_heads, hd))?.permute((0, 2, 1, 3))?;
            let k = k.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;
            let v = v.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;

            let q = apply_rope(&q, &cos, &sin)?;
            let k = apply_rope(&k, &cos, &sin)?;

            // GQA: repeat K,V heads
            let n_repeat = n_heads / n_kv;
            let k = if n_repeat > 1 { repeat_kv(&k, n_repeat)? } else { k };
            let v = if n_repeat > 1 { repeat_kv(&v, n_repeat)? } else { v };

            // Scaled dot-product attention
            let scale = (hd as f64).sqrt().recip();
            let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
            let attn = (attn * scale)?;
            let mask = build_causal_mask(seq_len, dev)?;
            let attn = attn.broadcast_add(&mask)?;
            let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
            let attn_out = attn.matmul(&v)?;

            // Reshape back
            let attn_out = attn_out.permute((0, 2, 1, 3))?.reshape((batch, seq_len, d_model))?;
            let attn_out = layer.attn_o.matmul(&attn_out)?;

            hidden = (residual + attn_out)?;

            // SwiGLU FFN
            let residual = hidden.clone();
            hidden = layer.ffn_norm.forward(&hidden)?;
            let gate = candle_nn::ops::silu(&layer.ffn_gate.matmul(&hidden)?)?;
            let up = layer.ffn_up.matmul(&hidden)?;
            hidden = layer.ffn_down.matmul(&(gate * up)?)?;
            hidden = (hidden + residual)?;
        }

        // Final norm + output projection
        hidden = self.output_norm.forward(&hidden)?;
        let out_w = self.output.as_ref().unwrap_or(&self.token_embd);
        // out_w shape: typically [d_model, vocab_size] (transposed)
        // hidden: [batch, seq_len, d_model]
        // We want: logits = hidden @ out_w^T → [batch, seq_len, vocab_size]
        let logits = hidden.matmul(&out_w.t()?)?;

        // Take last position
        let last = logits.i((.., seq_len.saturating_sub(1), ..))?;
        Ok(last)
    }

    /// Forward pass returning both logits and the final hidden state
    /// (post-norm, before LM head). The hidden state is used by the
    /// code predictor for acoustic code generation.
    pub fn forward_hidden(&self, input_ids: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let dev = &self.device;
        let cfg = &self.config;
        let batch = input_ids.dims()[0];
        let seq_len = input_ids.dims()[1];

        // --- Embedding lookup ---
        let emb_w = &self.token_embd;
        let d_model = cfg.d_model;
        let vocab_size = cfg.vocab_size;

        let emb_weights = if emb_w.dims()[0] == d_model && emb_w.dims()[1] == vocab_size {
            emb_w.t()?
        } else {
            emb_w.clone()
        };
        let mut hidden = emb_weights.gather(input_ids, 0)?;
        hidden = hidden.contiguous()?;

        // --- RoPE precompute ---
        let (cos, sin) = precompute_cos_sin(cfg, seq_len, dev)?;

        // --- Decoder layers ---
        for layer in &self.layers {
            let residual = hidden.clone();
            hidden = layer.attn_norm.forward(&hidden)?;

            let n_heads = cfg.n_heads;
            let n_kv = cfg.n_kv_heads;
            let hd = cfg.head_dim();

            let q = layer.attn_q.matmul(&hidden)?;
            let k = layer.attn_k.matmul(&hidden)?;
            let v = layer.attn_v.matmul(&hidden)?;

            let q = q.reshape((batch, seq_len, n_heads, hd))?.permute((0, 2, 1, 3))?;
            let k = k.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;
            let v = v.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;

            let q = apply_rope(&q, &cos, &sin)?;
            let k = apply_rope(&k, &cos, &sin)?;

            let n_repeat = n_heads / n_kv;
            let k = if n_repeat > 1 { repeat_kv(&k, n_repeat)? } else { k };
            let v = if n_repeat > 1 { repeat_kv(&v, n_repeat)? } else { v };

            let scale = (hd as f64).sqrt().recip();
            let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
            let attn = (attn * scale)?;
            let mask = build_causal_mask(seq_len, dev)?;
            let attn = attn.broadcast_add(&mask)?;
            let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
            let attn_out = attn.matmul(&v)?;

            let attn_out = attn_out.permute((0, 2, 1, 3))?.reshape((batch, seq_len, d_model))?;
            let attn_out = layer.attn_o.matmul(&attn_out)?;
            hidden = (residual + attn_out)?;

            let residual = hidden.clone();
            hidden = layer.ffn_norm.forward(&hidden)?;
            let gate = candle_nn::ops::silu(&layer.ffn_gate.matmul(&hidden)?)?;
            let up = layer.ffn_up.matmul(&hidden)?;
            hidden = layer.ffn_down.matmul(&(gate * up)?)?;
            hidden = (hidden + residual)?;
        }

        // Final norm
        let hidden_normed = self.output_norm.forward(&hidden)?;

        // LM head (output projection) — reuse token_embd if no separate output weight
        let out_w = self.output.as_ref().unwrap_or(&self.token_embd);
        let logits = hidden_normed.matmul(&out_w.t()?)?;

        // Return both logits and the normed hidden state
        let last_logits = logits.i((.., seq_len.saturating_sub(1), ..))?;
        Ok((last_logits, hidden_normed))
    }

    pub fn device(&self) -> &Device { &self.device }
    pub fn config(&self) -> &ModelConfig { &self.config }
}

// -----------------------------------------------------------------------
// RoPE helpers
// -----------------------------------------------------------------------

fn precompute_cos_sin(cfg: &ModelConfig, max_s: usize, dev: &Device) -> anyhow::Result<(Tensor, Tensor)> {
    let hd = cfg.head_dim();
    let inv_freq: Vec<f32> = (0..hd).step_by(2)
        .map(|i| (1.0_f64 / cfg.rope_theta.powf(i as f64 / hd as f64)) as f32)
        .collect();
    let n = inv_freq.len();
    let inv_freq = Tensor::from_slice(&inv_freq, (n,), dev)?;
    let pos: Vec<f32> = (0..max_s).map(|i| i as f32).collect();
    let pos = Tensor::from_slice(&pos, (max_s,), dev)?;
    // freqs: [max_s, n]
    let freqs = pos.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    // Interleave pairs → [max_s, hd]
    let cos = interleave(&cos, 2)?;
    let sin = interleave(&sin, 2)?;
    // Add batch+head dims → [1, 1, max_s, hd]
    Ok((cos.unsqueeze(0)?.unsqueeze(0)?, sin.unsqueeze(0)?.unsqueeze(0)?))
}

fn interleave(x: &Tensor, n: usize) -> Result<Tensor> {
    let s = x.dims();
    let last = s[s.len() - 1];
    let x = x.unsqueeze(s.len())?;
    let mut shape = s.to_vec();
    shape.push(n);
    let x = x.expand(shape.as_slice())?;
    let mut out_shape = s.to_vec();
    out_shape[s.len() - 1] = last * n;
    x.reshape(out_shape.as_slice())
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [batch, n_heads, seq_len, head_dim]
    let hd = x.dims()[3];
    let half = hd / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[x2.neg()?, x1], D::Minus1)?;
    x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)
}

fn repeat_kv(x: &Tensor, r: usize) -> Result<Tensor> {
    if r == 1 { return Ok(x.clone()); }
    let s = x.dims();
    let x = x.unsqueeze(2)?;
    let expanded = x.expand(&[s[0], s[1], r, s[2], s[3]])?;
    expanded.reshape(&[s[0], s[1] * r, s[2], s[3]])
}

fn build_causal_mask(n: usize, dev: &Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            data[i * n + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_slice(&data, (n, n), dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_rope_identity() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (1, 1, 1, 4), &dev).unwrap();
        let cos = Tensor::ones((1, 1, 1, 4), candle_core::DType::F32, &dev).unwrap();
        let sin = Tensor::zeros((1, 1, 1, 4), candle_core::DType::F32, &dev).unwrap();
        let r = apply_rope(&x, &cos, &sin).unwrap();
        let diff = (&r - &x).unwrap().abs().unwrap().sum_all().unwrap().to_vec0::<f32>().unwrap();
        assert!(diff < 1e-5);
    }

    #[test]
    fn test_repeat_kv_doubles() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32], (1, 1, 1, 1), &dev).unwrap();
        assert_eq!(repeat_kv(&x, 2).unwrap().dims(), &[1, 2, 1, 1]);
    }

    #[test]
    fn test_causal_mask_diagonal() {
        let dev = Device::Cpu;
        let m = build_causal_mask(4, &dev).unwrap();
        let v: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v[0].is_finite());   // (0,0)
        assert!(v[5].is_finite());   // (1,1)
        assert!(v[6].is_infinite()); // (1,2) future
        assert!(v[15].is_finite());  // (3,3)
    }
}
