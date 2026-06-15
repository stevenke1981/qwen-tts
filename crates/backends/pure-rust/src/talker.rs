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
/// Codebook 0 embedding and LM head (for autoregressive TTS).
const FN_CODEC_EMBD: &str = "talker.codec_embd.weight";
const FN_CODEC_HEAD: &str = "talker.codec_head.weight";

/// Text projection MLP tensor names.
const FN_TEXT_PROJ_FC1_W: &str = "talker.text_proj.fc1.weight";
const FN_TEXT_PROJ_FC1_B: &str = "talker.text_proj.fc1.bias";
const FN_TEXT_PROJ_FC2_W: &str = "talker.text_proj.fc2.weight";
const FN_TEXT_PROJ_FC2_B: &str = "talker.text_proj.fc2.bias";

/// Prefix for talker transformer layer tensors.
const BLK_PREFIX: &str = "talker.blk";

/// Qwen2 talker transformer.
pub struct Talker {
    config: ModelConfig,
    token_embd: Tensor,
    output: Option<Tensor>,
    output_norm: RmsNorm,
    /// Embedding for codebook 0 tokens (autoregressive TTS).
    codec_embd: Option<Tensor>,
    /// LM head for codebook 0 prediction.
    codec_head: Option<Tensor>,
    /// Text projection MLP: fc1 → silu → fc2
    text_proj_fc1_w: Tensor,
    text_proj_fc1_b: Tensor,
    text_proj_fc2_w: Tensor,
    text_proj_fc2_b: Tensor,
    layers: Vec<DecoderLayer>,
    device: Device,
}

pub(crate) struct DecoderLayer {
    pub(crate) attn_norm: RmsNorm,
    pub(crate) attn_q: Tensor,
    pub(crate) attn_k: Tensor,
    pub(crate) attn_v: Tensor,
    pub(crate) attn_o: Tensor,
    /// Per-head QK-norm (applied after reshaping to multi-head, before RoPE).
    pub(crate) attn_q_norm: RmsNorm,
    pub(crate) attn_k_norm: RmsNorm,
    pub(crate) ffn_norm: RmsNorm,
    pub(crate) ffn_gate: Tensor,
    pub(crate) ffn_up: Tensor,
    pub(crate) ffn_down: Tensor,
}

// -----------------------------------------------------------------------
// KV Cache
// -----------------------------------------------------------------------

/// Per-layer key/value cache for incremental decoding.
///
/// Stores the K and V tensors for each decoder layer as they are computed,
/// growing one position at a time. Used with `Talker::forward_step`.
pub struct KvCache {
    k_caches: Vec<Option<Tensor>>,
    v_caches: Vec<Option<Tensor>>,
}

impl KvCache {
    /// Create a new cache for `n_layers` layers (all empty).
    pub fn new(n_layers: usize) -> Self {
        Self {
            k_caches: vec![None; n_layers],
            v_caches: vec![None; n_layers],
        }
    }

    /// Append K,V for one token at layer `layer`.
    ///
    /// K/V shapes: `[batch, n_kv_heads, 1, head_dim]`.
    pub fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let new_k = match &self.k_caches[layer] {
            Some(cache) => Tensor::cat(&[cache, k], 2)?,
            None => k.clone(),
        };
        let new_v = match &self.v_caches[layer] {
            Some(cache) => Tensor::cat(&[cache, v], 2)?,
            None => v.clone(),
        };
        self.k_caches[layer] = Some(new_k);
        self.v_caches[layer] = Some(new_v);
        Ok(())
    }

    /// Number of cached positions.
    pub fn current_len(&self) -> usize {
        self.k_caches[0]
            .as_ref()
            .map(|t| t.dims()[2])
            .unwrap_or(0)
    }

    /// Borrow the K cache at `layer` (returns `Some` after at least one append).
    pub fn k(&self, layer: usize) -> Option<&Tensor> {
        self.k_caches[layer].as_ref()
    }

    /// Borrow the V cache at `layer` (returns `Some` after at least one append).
    pub fn v(&self, layer: usize) -> Option<&Tensor> {
        self.v_caches[layer].as_ref()
    }
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

        // Optional: codebook 0 embedding + LM head (for autoregressive TTS)
        let codec_embd = if content.tensor_infos.contains_key(FN_CODEC_EMBD) {
            Some(load(FN_CODEC_EMBD)?)
        } else {
            None
        };
        let codec_head = if content.tensor_infos.contains_key(FN_CODEC_HEAD) {
            Some(load(FN_CODEC_HEAD)?)
        } else {
            None
        };

        // Text projection MLP (required for correct text embedding)
        let text_proj_fc1_w = load(FN_TEXT_PROJ_FC1_W)?;
        let text_proj_fc1_b = load(FN_TEXT_PROJ_FC1_B)?;
        let text_proj_fc2_w = load(FN_TEXT_PROJ_FC2_W)?;
        let text_proj_fc2_b = load(FN_TEXT_PROJ_FC2_B)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let blk = |n: &str| format!("{BLK_PREFIX}.{i}.{n}");
            let q_norm_w = load(&format!("{BLK_PREFIX}.{i}.attn_q_norm.weight"));
            let k_norm_w = load(&format!("{BLK_PREFIX}.{i}.attn_k_norm.weight"));
            // If QK-norm weights exist, use them; otherwise use identity-norm
            // (standard RmsNorm with weight=ones is fine since the default
            // gain of 1.0 matches the identity after norm).
            let attn_q_norm = match q_norm_w {
                Ok(w) => RmsNorm::new(w, cfg.norm_eps),
                Err(_) => {
                    let ones = Tensor::ones(cfg.head_dim(), candle_core::DType::F32, device)?;
                    RmsNorm::new(ones, cfg.norm_eps)
                }
            };
            let attn_k_norm = match k_norm_w {
                Ok(w) => RmsNorm::new(w, cfg.norm_eps),
                Err(_) => {
                    let ones = Tensor::ones(cfg.head_dim(), candle_core::DType::F32, device)?;
                    RmsNorm::new(ones, cfg.norm_eps)
                }
            };
            layers.push(DecoderLayer {
                attn_norm: RmsNorm::new(load(&blk(FN_ATTN_NORM))?, cfg.norm_eps),
                attn_q: load(&blk(FN_ATTN_Q))?,
                attn_k: load(&blk(FN_ATTN_K))?,
                attn_v: load(&blk(FN_ATTN_V))?,
                attn_o: load(&blk(FN_ATTN_O))?,
                attn_q_norm,
                attn_k_norm,
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
            codec_embd,
            codec_head,
            text_proj_fc1_w,
            text_proj_fc1_b,
            text_proj_fc2_w,
            text_proj_fc2_b,
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
        let d_model = cfg.d_model;
        let vocab_size = cfg.vocab_size;

        // --- Embedding lookup (manual) ---
        let emb_w = &self.token_embd;
        let emb_weights = if emb_w.dims()[0] == d_model && emb_w.dims()[1] == vocab_size {
            emb_w.t()?
        } else {
            emb_w.clone()
        };
        // index_select requires 1D indices
        let ids_flat = input_ids.flatten_all()?;
        let hidden_flat = emb_weights.index_select(&ids_flat, 0)?;
        let mut hidden = hidden_flat.reshape((batch, seq_len, d_model))?;
        hidden = hidden.contiguous()?;

        // --- RoPE precompute ---
        let (cos, sin) = precompute_cos_sin(cfg, seq_len, dev)?;

        // --- Decoder layers ---
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim();
        let head_dim_sum = n_heads * hd;
        
        for layer in &self.layers {
            let residual = hidden.clone();
            hidden = layer.attn_norm.forward(&hidden)?;

            // Flatten batch*seq_len for 2D matmul (candle requires matching ranks)
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;

            // QKV projections: h_2d [B*T, d_model] @ W^T
            let q = h_2d.matmul(&layer.attn_q.t()?)?; // [B*T, head_dim_sum]
            let k = h_2d.matmul(&layer.attn_k.t()?)?; // [B*T, n_kv*hd]
            let v = h_2d.matmul(&layer.attn_v.t()?)?; // [B*T, n_kv*hd]

            // Reshape to multi-head: [B*T, dim] → [B, T, n_heads, hd] → [B, n_heads, T, hd]
            let q = q.reshape((batch, seq_len, n_heads, hd))?.permute((0, 2, 1, 3))?;
            let k = k.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;
            let v = v.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;

            // Per-head QK-norm (after reshape, before RoPE)
            let q = apply_per_head_norm(&q, &layer.attn_q_norm)?;
            let k = apply_per_head_norm(&k, &layer.attn_k_norm)?;

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

            // Output projection: flatten to 2D, apply, reshape back
            let attn_flat = attn_out
                .permute((0, 2, 1, 3))?
                .reshape((batch * seq_len, head_dim_sum))?;
            let attn_proj = attn_flat.matmul(&layer.attn_o.t()?)?; // [B*T, d_model]
            hidden = (residual + attn_proj.reshape((batch, seq_len, d_model))?)?;

            // SwiGLU FFN
            let residual = hidden.clone();
            hidden = layer.ffn_norm.forward(&hidden)?;
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;
            let gate = candle_nn::ops::silu(&h_2d.matmul(&layer.ffn_gate.t()?)?)?;
            let up = h_2d.matmul(&layer.ffn_up.t()?)?;
            let hid_2d = (gate * up)?;
            let hid_out = hid_2d.matmul(&layer.ffn_down.t()?)?; // [B*T, d_model]
            hidden = (residual + hid_out.reshape((batch, seq_len, d_model))?)?;
        }

        // Final norm + output projection
        hidden = self.output_norm.forward(&hidden)?;
        let out_w = self.output.as_ref().unwrap_or(&self.token_embd);
        let logits = linear_fwd(out_w, &hidden)?;

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
        let d_model = cfg.d_model;
        let vocab_size = cfg.vocab_size;

        // --- Embedding lookup ---
        let emb_w = &self.token_embd;
        let emb_weights = if emb_w.dims()[0] == d_model && emb_w.dims()[1] == vocab_size {
            emb_w.t()?
        } else {
            emb_w.clone()
        };
        // Flatten input_ids to 1D first (index_select requires 1D indices)
        let ids_flat = input_ids.flatten_all()?;
        let hidden_flat_res = emb_weights.index_select(&ids_flat, 0)?;
        let mut hidden = hidden_flat_res.reshape((batch, seq_len, d_model))?;
        hidden = hidden.contiguous()?;

        // --- RoPE precompute ---
        let (cos, sin) = precompute_cos_sin(cfg, seq_len, dev)?;

        // --- Decoder layers ---
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim();
        let head_dim_sum = n_heads * hd;
        
        for layer in &self.layers {
            let residual = hidden.clone();
            hidden = layer.attn_norm.forward(&hidden)?;

            // Flatten batch*seq_len for 2D matmul (candle requires matching ranks)
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;

            // QKV projections: h_2d [B*T, d_model] @ W^T
            let q = h_2d.matmul(&layer.attn_q.t()?)?; // [B*T, head_dim_sum]
            let k = h_2d.matmul(&layer.attn_k.t()?)?; // [B*T, n_kv*hd]
            let v = h_2d.matmul(&layer.attn_v.t()?)?; // [B*T, n_kv*hd]

            // Reshape to multi-head: [B*T, dim] → [B, T, n_heads, hd] → [B, n_heads, T, hd]
            let q = q.reshape((batch, seq_len, n_heads, hd))?.permute((0, 2, 1, 3))?;
            let k = k.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;
            let v = v.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;

            // Per-head QK-norm (after reshape, before RoPE)
            let q = apply_per_head_norm(&q, &layer.attn_q_norm)?;
            let k = apply_per_head_norm(&k, &layer.attn_k_norm)?;

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

            // Output projection: flatten to 2D, apply, reshape back
            let attn_flat = attn_out
                .permute((0, 2, 1, 3))?
                .reshape((batch * seq_len, head_dim_sum))?;
            let attn_proj = attn_flat.matmul(&layer.attn_o.t()?)?; // [B*T, d_model]
            hidden = (residual + attn_proj.reshape((batch, seq_len, d_model))?)?;

            // SwiGLU FFN
            let residual = hidden.clone();
            hidden = layer.ffn_norm.forward(&hidden)?;
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;
            let gate = candle_nn::ops::silu(&h_2d.matmul(&layer.ffn_gate.t()?)?)?;
            let up = h_2d.matmul(&layer.ffn_up.t()?)?;
            let hid_2d = (gate * up)?;
            let hid_out = hid_2d.matmul(&layer.ffn_down.t()?)?; // [B*T, d_model]
            hidden = (residual + hid_out.reshape((batch, seq_len, d_model))?)?;
        }

        // Final norm
        let hidden_normed = self.output_norm.forward(&hidden)?;

        // LM head (output projection) — reuse token_embd if no separate output weight
        let out_w = self.output.as_ref().unwrap_or(&self.token_embd);
        // out_w: [vocab_size, d_model], hidden_normed: [B, T, d_model]
        // Use linear_fwd which handles flatten→matmul→reshape
        let logits = linear_fwd(out_w, &hidden_normed)?;

        // Return both logits and the normed hidden state
        let last_logits = logits.i((.., seq_len.saturating_sub(1), ..))?;
        Ok((last_logits, hidden_normed))
    }

    /// Forward with pre-computed embeddings (skips embedding lookup).
    ///
    /// `embeddings`: `[batch, seq_len, d_model]` — pre-mixed text + code embeddings.
    /// Returns the output-normed hidden state: `[batch, seq_len, d_model]`.
    pub fn forward_embeddings(&self, embeddings: &Tensor) -> anyhow::Result<Tensor> {
        let dev = &self.device;
        let cfg = &self.config;
        let (batch, seq_len, d_model) = embeddings.dims3()?;
        assert_eq!(d_model, cfg.d_model, "embedding dim mismatch");

        let mut hidden = embeddings.clone();

        // --- RoPE precompute ---
        let (cos, sin) = precompute_cos_sin(cfg, seq_len, dev)?;

        // --- Decoder layers ---
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim();
        let head_dim_sum = n_heads * hd;
        
        for layer in &self.layers {
            let residual = hidden.clone();
            hidden = layer.attn_norm.forward(&hidden)?;

            // Flatten batch*seq_len for 2D matmul (candle requires matching ranks)
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;

            // QKV projections: h_2d [B*T, d_model] @ W^T
            let q = h_2d.matmul(&layer.attn_q.t()?)?; // [B*T, head_dim_sum]
            let k = h_2d.matmul(&layer.attn_k.t()?)?; // [B*T, n_kv*hd]
            let v = h_2d.matmul(&layer.attn_v.t()?)?; // [B*T, n_kv*hd]

            // Reshape to multi-head: [B*T, dim] → [B, T, n_heads, hd] → [B, n_heads, T, hd]
            let q = q.reshape((batch, seq_len, n_heads, hd))?.permute((0, 2, 1, 3))?;
            let k = k.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;
            let v = v.reshape((batch, seq_len, n_kv, hd))?.permute((0, 2, 1, 3))?;

            // Per-head QK-norm (after reshape, before RoPE)
            let q = apply_per_head_norm(&q, &layer.attn_q_norm)?;
            let k = apply_per_head_norm(&k, &layer.attn_k_norm)?;

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

            // Output projection: flatten to 2D, apply, reshape back
            let attn_flat = attn_out
                .permute((0, 2, 1, 3))?
                .reshape((batch * seq_len, head_dim_sum))?;
            let attn_proj = attn_flat.matmul(&layer.attn_o.t()?)?; // [B*T, d_model]
            hidden = (residual + attn_proj.reshape((batch, seq_len, d_model))?)?;

            // SwiGLU FFN
            let residual = hidden.clone();
            hidden = layer.ffn_norm.forward(&hidden)?;
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;
            let gate = candle_nn::ops::silu(&h_2d.matmul(&layer.ffn_gate.t()?)?)?;
            let up = h_2d.matmul(&layer.ffn_up.t()?)?;
            let hid_2d = (gate * up)?;
            let hid_out = hid_2d.matmul(&layer.ffn_down.t()?)?; // [B*T, d_model]
            hidden = (residual + hid_out.reshape((batch, seq_len, d_model))?)?;
        }

        // Final norm
        let hidden_normed = self.output_norm.forward(&hidden)?;
        Ok(hidden_normed)
    }

    /// Predict codebook 0 token logits from the final hidden state.
    ///
    /// `hidden`: `[batch, 1, d_model]` — the last position's hidden state.
    /// Returns logits: `[batch, 1, codebook0_vocab]`.
    pub fn predict_codebook0(&self, hidden: &Tensor) -> anyhow::Result<Tensor> {
        let head = self.codec_head.as_ref()
            .ok_or_else(|| anyhow::anyhow!("codec_head not loaded (no codec_head.weight in GGUF)"))?;
        // codec_head shape: [vocab_size_cb0, d_model] = [3072, 2048]
        // hidden shape: [batch, 1, d_model]
        // logits = hidden @ head^T = linear_fwd(head, hidden)
        let logits = linear_fwd(head, hidden)?;
        Ok(logits)
    }

    /// Embed a codebook 0 token ID into `[batch, 1, d_model]`.
    pub fn embed_codebook0(&self, token_id: u32) -> anyhow::Result<Tensor> {
        let emb = self.codec_embd.as_ref()
            .ok_or_else(|| anyhow::anyhow!("codec_embd not loaded"))?;
        let d_model = self.config.d_model;
        // Determine layout: [d_model, vocab_size] (transposed) or [vocab_size, d_model]
        let emb_w = if emb.dims()[0] == d_model {
            // Transposed: [d_model, vocab_size] — transpose to [vocab_size, d_model]
            emb.t()?
        } else {
            emb.clone()
        };
        let ids = Tensor::from_slice(&[token_id], (1, 1), &self.device)?;
        // emb_w: [vocab_size, d_model]
        // index_select requires 1D indices
        let ids_flat = ids.flatten_all()?;
        let result_flat = emb_w.index_select(&ids_flat, 0)?;
        // Reshape back to [1, 1, d_model]
        let result = result_flat.reshape((1, 1, d_model))?;
        Ok(result.contiguous()?)
    }

    /// Process one token through all decoder layers with KV cache (incremental decode).
    ///
    /// `x`: `[batch, 1, d_model]` — single token embedding.
    /// `cache`: mutable per-layer KV cache (will be appended to).
    /// `cos_full`, `sin_full`: `[1, 1, max_seq, head_dim]` — precomputed RoPE
    ///   for all positions; the current position is `cache.current_len()`.
    ///
    /// Returns output-normed hidden state: `[batch, 1, d_model]`.
    pub fn forward_step(
        &self,
        x: &Tensor,
        cache: &mut KvCache,
        cos_full: &Tensor,
        sin_full: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let cfg = &self.config;
        let (batch, _one, d_model) = x.dims3()?;
        assert_eq!(d_model, cfg.d_model, "forward_step dim mismatch");

        let mut hidden = x.clone();
        let pos = cache.current_len();
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim();
        let head_dim_sum = n_heads * hd;
        let n_repeat = n_heads / n_kv;

        for (i, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            hidden = layer.attn_norm.forward(&hidden)?;

            // QKV projections for a single token: [B, d_model] → [B, ...]
            let h_2d = hidden.reshape((batch, d_model))?;
            let q = h_2d.matmul(&layer.attn_q.t()?)?; // [B, n_heads*hd]
            let k = h_2d.matmul(&layer.attn_k.t()?)?; // [B, n_kv*hd]
            let v = h_2d.matmul(&layer.attn_v.t()?)?; // [B, n_kv*hd]

            // Reshape to multi-head
            let q = q.reshape((batch, n_heads, 1, hd))?;
            let k = k.reshape((batch, n_kv, 1, hd))?;
            let v = v.reshape((batch, n_kv, 1, hd))?;

            // Per-head QK-norm (after reshape, before RoPE)
            let q = apply_per_head_norm(&q, &layer.attn_q_norm)?;
            let k = apply_per_head_norm(&k, &layer.attn_k_norm)?;

            // RoPE at the current position
            let cos = cos_full.narrow(D::Minus2, pos, 1)?;
            let sin = sin_full.narrow(D::Minus2, pos, 1)?;
            let q = apply_rope(&q, &cos, &sin)?;
            let k = apply_rope(&k, &cos, &sin)?;

            // Append new K,V to cache
            cache.append(i, &k, &v)?;

            // Get full K/V from cache
            let k_cache = cache
                .k(i)
                .ok_or_else(|| anyhow::anyhow!("K cache empty after append"))?;
            let v_cache = cache
                .v(i)
                .ok_or_else(|| anyhow::anyhow!("V cache empty after append"))?;

            // GQA: repeat K,V heads to match n_heads
            let k_full = if n_repeat > 1 {
                repeat_kv(k_cache, n_repeat)?
            } else {
                k_cache.clone()
            };
            let v_full = if n_repeat > 1 {
                repeat_kv(v_cache, n_repeat)?
            } else {
                v_cache.clone()
            };

            // Scaled dot-product attention (single query, full KV from cache)
            let scale = (hd as f64).sqrt().recip();
            let attn = q.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?;
            let attn = (attn * scale)?;
            // No causal mask: the query is at the last position, it attends to
            // all cached positions (which are all ≤ current position).
            let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
            let attn_out = attn.matmul(&v_full)?;

            // Output projection
            let attn_out = attn_out
                .permute((0, 2, 1, 3))?
                .reshape((batch, head_dim_sum))?;
            let attn_proj = attn_out.matmul(&layer.attn_o.t()?)?;
            hidden = (residual + attn_proj.reshape((batch, 1, d_model))?)?;

            // SwiGLU FFN
            let residual = hidden.clone();
            hidden = layer.ffn_norm.forward(&hidden)?;
            let h_2d = hidden.reshape((batch, d_model))?;
            let gate = candle_nn::ops::silu(&h_2d.matmul(&layer.ffn_gate.t()?)?)?;
            let up = h_2d.matmul(&layer.ffn_up.t()?)?;
            let hid_2d = (gate * up)?;
            let hid_out = hid_2d.matmul(&layer.ffn_down.t()?)?;
            hidden = (residual + hid_out.reshape((batch, 1, d_model))?)?;
        }

        // Final output norm
        let hidden = self.output_norm.forward(&hidden)?;
        Ok(hidden)
    }

    /// Apply the text projection MLP: fc1 → silu → fc2 + bias.
    ///
    /// `x`: `[batch, seq_len, d_model]` — text embeddings.
    /// Returns: `[batch, seq_len, d_model]` — projected embeddings.
    pub fn apply_text_proj(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let fc1 = linear_fwd(&self.text_proj_fc1_w, x)?.broadcast_add(&self.text_proj_fc1_b)?;
        let act = candle_nn::ops::silu(&fc1)?;
        let fc2 = linear_fwd(&self.text_proj_fc2_w, &act)?.broadcast_add(&self.text_proj_fc2_b)?;
        Ok(fc2)
    }

    /// Embed text token IDs into `[batch, seq_len, d_model]` using the
    /// text embedding table (`text_embd.weight`) followed by text_proj MLP.
    pub fn embed_text(&self, input_ids: &Tensor) -> anyhow::Result<Tensor> {
        let cfg = &self.config;
        let emb_w = &self.token_embd;
        let d_model = cfg.d_model;
        let vocab_size = cfg.vocab_size;
        let batch = input_ids.dims()[0];
        let seq_len = input_ids.dims()[1];
        let emb_weights = if emb_w.dims()[0] == d_model && emb_w.dims()[1] == vocab_size {
            emb_w.t()?
        } else {
            emb_w.clone()
        };
        // index_select requires 1D indices
        let ids_flat = input_ids.flatten_all()?;
        let hidden_flat = emb_weights.index_select(&ids_flat, 0)?;
        let hidden = hidden_flat.reshape((batch, seq_len, d_model))?;
        let hidden = hidden.contiguous()?;
        // Apply text projection MLP
        self.apply_text_proj(&hidden)
    }

    pub fn device(&self) -> &Device { &self.device }
    pub fn config(&self) -> &ModelConfig { &self.config }
}

// -----------------------------------------------------------------------
// Linear projection helper
// -----------------------------------------------------------------------

/// Apply a linear projection `weight @ x` with correct rank handling.
///
/// `weight`: `[out_features, in_features]` (GGUF convention).
/// `x`: any-rank tensor with last dim = `in_features`.
///
/// Flattens all batch dims to 2D, applies `x @ W^T`, reshapes back.
pub(crate) fn linear_fwd(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    let x_dims = x.dims();
    let rank = x_dims.len();
    let bsz: usize = x_dims[..rank - 1].iter().product();
    let in_features = x_dims[rank - 1];
    let out_features = weight.dims()[0];
    let x_2d = x.reshape((bsz, in_features))?;
    let w_t = weight.t()?; // [in_features, out_features]
    let y_2d = x_2d.matmul(&w_t)?;
    let mut out_dims = x_dims.to_vec();
    out_dims[rank - 1] = out_features;
    y_2d.reshape(out_dims)
}

// -----------------------------------------------------------------------
// RoPE helpers
// -----------------------------------------------------------------------

pub(crate) fn precompute_cos_sin(cfg: &ModelConfig, max_s: usize, dev: &Device) -> anyhow::Result<(Tensor, Tensor)> {
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

pub(crate) fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [batch, n_heads, seq_len, head_dim]
    let hd = x.dims()[3];
    let half = hd / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[x2.neg()?, x1], D::Minus1)?;
    x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)
}

pub(crate) fn repeat_kv(x: &Tensor, r: usize) -> Result<Tensor> {
    if r == 1 { return Ok(x.clone()); }
    let s = x.dims();
    let x = x.unsqueeze(2)?;
    let expanded = x.expand(&[s[0], s[1], r, s[2], s[3]])?;
    expanded.reshape(&[s[0], s[1] * r, s[2], s[3]])
}

pub(crate) fn build_causal_mask(n: usize, dev: &Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            data[i * n + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_slice(&data, (n, n), dev)
}

/// Apply a per-head RMS norm to a multi-head attention tensor.
///
/// `x`: `[batch, n_heads, seq_len, head_dim]`
/// `norm`: RmsNorm with weight `[head_dim]`
///
/// Returns: `[batch, n_heads, seq_len, head_dim]`
pub(crate) fn apply_per_head_norm(x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
    let shape = x.dims();
    let n_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];
    // Flatten batch*n_heads*seq_len into one dim, apply norm over head_dim
    let x_flat = x.reshape((shape[0] * n_heads * seq_len, head_dim))?;
    let x_normed = norm.forward(&x_flat)?;
    x_normed.reshape(shape)
}

/// General-purpose embedding lookup via index_select.
///
/// `weight`: `[vocab_size, d_model]` or `[d_model, vocab_size]` (transposed).
/// Returns `[1, 1, d_model]`.
pub(crate) fn embed_token(weight: &Tensor, token: u32, d_model: usize, device: &Device) -> Result<Tensor> {
    // Normalize layout to [vocab_size, d_model] and ensure contiguous
    // (candle's index_select requires contiguous tensors).
    let emb_w = if weight.dims()[0] == d_model && weight.dims()[1] != d_model {
        weight.t()?.contiguous()?
    } else {
        // Already [vocab_size, d_model]; make contiguous if needed
        weight.clone()
    };
    let ids = Tensor::from_slice(&[token], (1,), device)?;
    let result = emb_w.index_select(&ids, 0)?;
    result.reshape((1, 1, d_model))
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
