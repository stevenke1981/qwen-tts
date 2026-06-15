//! Qwen2 talker transformer — loaded from a GGUF file.
//!
//! Linear projection weights are stored as Q8_0 quantized blocks and use a
//! custom single-threaded GEMV (see [`crate::qgemv`]) which avoids the Rayon
//! overhead of candle's `k_quants::matmul`. RMS norm weights, embeddings, and
//! biases remain F32.

use std::fs::File;
use std::path::Path;

use candle_core::quantized::gguf_file::Content;
use candle_core::{Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::RmsNorm;
use rayon::prelude::*;

use crate::config::ModelConfig;
use crate::code_predictor::CodePredictor;
use crate::custom_ops::{
    attention_f32_par, attention_gqa_flat,
    per_head_rms_norm_f32_par,
    rms_norm_f32, rms_norm_f32_inplace, rms_norm_tensor,
    rope_f32_par, silu_f32_par,
};
use crate::qgemv::{q8_linear, q8_linear_multi, Q8Weights, Q8Workspace};

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
    /// LM head for codebook 0 prediction (F32).
    codec_head: Option<Tensor>,
    /// Text projection MLP: fc1 → silu → fc2
    text_proj_fc1_w: Tensor,
    text_proj_fc1_b: Tensor,
    text_proj_fc2_w: Tensor,
    text_proj_fc2_b: Tensor,
    layers: Vec<DecoderLayer>,
    device: Device,
    /// Persistent Q8 workspace (avoids re-allocation on the hot path).
    pub(crate) q8_ws: Q8Workspace,
    /// F32 view of output_norm weight (for fused forward path).
    pub(crate) output_norm_w: Vec<f32>,
    /// Pre-allocated scratch buffers for the fused forward path.
    pub(crate) scratch: FusedScratch,
}

/// Pre-allocated scratch buffers for `forward_step_fused`.
///
/// These are sized once during `from_content` and reused across all
/// fused forward steps, avoiding repeated heap allocation on the hot path.
pub(crate) struct FusedScratch {
    /// Saved pre-norm hidden state for residual connections (d_model).
    pub(crate) residual: Vec<f32>,
    /// Output of RMS norm (d_model). Used as input to QKV and FFN.
    pub(crate) normed: Vec<f32>,
    /// FFN intermediate: SiLU(gate) * up (ffn_dim).
    pub(crate) ffn_mid: Vec<f32>,

    // ── Pre-allocated GEMV output buffers (allocation-free hot path) ──
    /// Q projection output [attn_dim] → reused for attn_o, ffn_down.
    pub(crate) q_buf: Vec<f32>,
    /// K projection output [kv_dim].
    pub(crate) k_buf: Vec<f32>,
    /// V projection output [kv_dim].
    pub(crate) v_buf: Vec<f32>,
    /// FFN gate projection output [ffn_dim].
    pub(crate) gate_buf: Vec<f32>,
    /// FFN up projection output [ffn_dim].
    pub(crate) up_buf: Vec<f32>,
}

pub(crate) struct DecoderLayer {
    pub(crate) attn_norm: RmsNorm,
    /// F32 view of attn_norm weight for the fused forward path.
    pub(crate) attn_norm_w: Vec<f32>,
    pub(crate) attn_q: Q8Weights,
    pub(crate) attn_k: Q8Weights,
    pub(crate) attn_v: Q8Weights,
    pub(crate) attn_o: Q8Weights,
    /// Per-head QK-norm (applied after reshaping to multi-head, before RoPE).
    pub(crate) attn_q_norm: RmsNorm,
    pub(crate) attn_k_norm: RmsNorm,
    /// F32 view of QK-norm weights (for fused forward path).
    pub(crate) attn_q_norm_w: Vec<f32>,
    pub(crate) attn_k_norm_w: Vec<f32>,
    pub(crate) ffn_norm: RmsNorm,
    /// F32 view of ffn_norm weight for the fused forward path.
    pub(crate) ffn_norm_w: Vec<f32>,
    pub(crate) ffn_gate: Q8Weights,
    pub(crate) ffn_up: Q8Weights,
    pub(crate) ffn_down: Q8Weights,
}

// -----------------------------------------------------------------------
// KV Cache (flat pre-allocated buffers)
// -----------------------------------------------------------------------

/// Per-layer key/value cache using pre-allocated flat f32 buffers.
///
/// Each layer allocates an f32 buffer of size `n_kv_heads * max_seq_len * head_dim`.
/// On each append, the new K,V values are memcpy'd into the buffer at position
/// `pos`. Attention reads directly from the flat buffer (zero-copy slice reference).
///
/// This replaces the old `KvCache` which used `Tensor::cat` — a growing
/// allocation pattern that copies O(n²) data over the full decode run.
pub struct KvCacheFlat {
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    k_bufs: Vec<Vec<f32>>,
    v_bufs: Vec<Vec<f32>>,
    pos: usize,
}

impl KvCacheFlat {
    /// The head stride (max_seq) used when allocating this cache.
    /// This is the number of positions between consecutive heads in the flat buffer,
    /// needed by `attention_f32` to correctly compute head offsets.
    pub fn head_stride(&self) -> usize {
        self.max_seq
    }

    /// Create a new cache with pre-allocated buffers of `max_seq` per head.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let per_layer = n_kv_heads * max_seq * head_dim;
        Self {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq,
            k_bufs: (0..n_layers).map(|_| vec![0.0f32; per_layer]).collect(),
            v_bufs: (0..n_layers).map(|_| vec![0.0f32; per_layer]).collect(),
            pos: 0,
        }
    }

    /// Number of cached positions.
    pub fn current_len(&self) -> usize {
        self.pos
    }

    /// Append K,V for one token at layer `layer`.
    ///
    /// `k`, `v`: flat f32 slices in `[n_kv_heads, 1, head_dim]` row-major
    /// (from the output of q8_linear after reshape). These are copied into
    /// the pre-allocated buffer at position `pos`.
    ///
    /// Buffer layout: `[n_kv_heads, max_seq, head_dim]` row-major.
    /// For head `h` at position `t`, dim `d`: offset = h * (max_seq * hd) + t * hd + d
    ///
    /// NOTE: This does NOT increment `self.pos`. Call `advance_pos()` once
    /// after all layers have appended for the current forward step.
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let p = self.pos;
        let max_hd = self.max_seq * hd;
        for h in 0..n_kv {
            let k_src = &k[h * hd..(h + 1) * hd];
            let k_dst = &mut self.k_bufs[layer][h * max_hd + p * hd..h * max_hd + (p + 1) * hd];
            k_dst.copy_from_slice(k_src);

            let v_src = &v[h * hd..(h + 1) * hd];
            let v_dst = &mut self.v_bufs[layer][h * max_hd + p * hd..h * max_hd + (p + 1) * hd];
            v_dst.copy_from_slice(v_src);
        }
    }

    /// Call once after all layers have appended for the current step.
    pub fn advance_pos(&mut self) {
        self.pos += 1;
    }

    /// Reference to the pre-allocated K buffer for `layer`.
    ///
    /// The buffer always has length `n_kv_heads * max_seq * head_dim`, but only
    /// the first `current_len()` positions of each head contain valid data.
    /// Use `head_stride = max_seq` when passing to `attention_f32`.
    pub fn k_slice(&self, layer: usize) -> &[f32] {
        &self.k_bufs[layer]
    }

    /// Reference to the pre-allocated V buffer for `layer`.
    pub fn v_slice(&self, layer: usize) -> &[f32] {
        &self.v_bufs[layer]
    }
}

impl FusedScratch {
    /// Allocate scratch buffers sized for the fused forward path.
    ///
    /// `d_model`: hidden dimension.
    /// `ffn_dim`: intermediate FFN dimension (for SiLU(gate) * up).
    /// `attn_dim`: Q projection output size (= n_heads × head_dim).
    /// `kv_dim`: K/V projection output size (= n_kv_heads × head_dim).
    pub fn new(d_model: usize, ffn_dim: usize, attn_dim: usize, kv_dim: usize) -> Self {
        Self {
            residual: vec![0.0f32; d_model],
            normed: vec![0.0f32; d_model],
            ffn_mid: vec![0.0f32; ffn_dim],
            q_buf: vec![0.0f32; attn_dim],
            k_buf: vec![0.0f32; kv_dim],
            v_buf: vec![0.0f32; kv_dim],
            gate_buf: vec![0.0f32; ffn_dim],
            up_buf: vec![0.0f32; ffn_dim],
        }
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

/// Load a single Q8_0 quantized weight from GGUF (no dequantization).
fn load_q8_weight(content: &Content, file: &mut File, name: &str) -> anyhow::Result<Q8Weights> {
    Q8Weights::from_gguf(content, file, name)
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

        // Helper: load F32 tensor (pass `&mut file` to reborrow per call)
        let load_f32 = |name: &str, f: &mut File| load_f32_tensor(&content, f, name, device);

        let token_embd = load_f32(FN_TOKEN_EMBD, &mut file)?;

        let output = if content.tensor_infos.contains_key(FN_OUTPUT) {
            Some(load_f32(FN_OUTPUT, &mut file)?)
        } else {
            None
        };

        let output_norm = RmsNorm::new(load_f32(FN_OUTPUT_NORM, &mut file)?, cfg.norm_eps);
        let output_norm_w = output_norm.weight().to_vec1::<f32>()?;

        // Optional: codebook 0 embedding + LM head (for autoregressive TTS)
        let codec_embd = if content.tensor_infos.contains_key(FN_CODEC_EMBD) {
            Some(load_f32(FN_CODEC_EMBD, &mut file)?)
        } else {
            None
        };
        let codec_head = if content.tensor_infos.contains_key(FN_CODEC_HEAD) {
            Some(load_f32(FN_CODEC_HEAD, &mut file)?)
        } else {
            None
        };

        // Text projection MLP (required for correct text embedding)
        let text_proj_fc1_w = load_f32(FN_TEXT_PROJ_FC1_W, &mut file)?;
        let text_proj_fc1_b = load_f32(FN_TEXT_PROJ_FC1_B, &mut file)?;
        let text_proj_fc2_w = load_f32(FN_TEXT_PROJ_FC2_W, &mut file)?;
        let text_proj_fc2_b = load_f32(FN_TEXT_PROJ_FC2_B, &mut file)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let blk = |n: &str| format!("{BLK_PREFIX}.{i}.{n}");
            let q_norm_w = load_f32(&format!("{BLK_PREFIX}.{i}.attn_q_norm.weight"), &mut file);
            let k_norm_w = load_f32(&format!("{BLK_PREFIX}.{i}.attn_k_norm.weight"), &mut file);
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
            // F32 norm weights for fused forward path.
            let attn_norm = RmsNorm::new(load_f32(&blk(FN_ATTN_NORM), &mut file)?, cfg.norm_eps);
            let attn_norm_w = attn_norm.weight().to_vec1::<f32>()?;
            let attn_q_norm_w = attn_q_norm.weight().to_vec1::<f32>()?;
            let attn_k_norm_w = attn_k_norm.weight().to_vec1::<f32>()?;
            let ffn_norm = RmsNorm::new(load_f32(&blk(FN_FFN_NORM), &mut file)?, cfg.norm_eps);
            let ffn_norm_w = ffn_norm.weight().to_vec1::<f32>()?;

            // Helper: load Q8_0 quantized weight directly from GGUF.
            let load_q8 = |name: &str, f: &mut File| load_q8_weight(&content, f, name);
            layers.push(DecoderLayer {
                attn_norm,
                attn_norm_w,
                attn_q: load_q8(&blk(FN_ATTN_Q), &mut file)?,
                attn_k: load_q8(&blk(FN_ATTN_K), &mut file)?,
                attn_v: load_q8(&blk(FN_ATTN_V), &mut file)?,
                attn_o: load_q8(&blk(FN_ATTN_O), &mut file)?,
                attn_q_norm,
                attn_k_norm,
                attn_q_norm_w,
                attn_k_norm_w,
                ffn_norm,
                ffn_norm_w,
                ffn_gate: load_q8(&blk(FN_FFN_GATE), &mut file)?,
                ffn_up: load_q8(&blk(FN_FFN_UP), &mut file)?,
                ffn_down: load_q8(&blk(FN_FFN_DOWN), &mut file)?,
            });
        }

        // Size scratch buffers for the fused forward path.
        let ffn_dim = layers[0].ffn_gate.out_features();
        let attn_dim = layers[0].attn_q.out_features();
        let kv_dim = layers[0].attn_k.out_features();
        let scratch = FusedScratch::new(cfg.d_model, ffn_dim, attn_dim, kv_dim);

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
            q8_ws: Q8Workspace::new(),
            output_norm_w,
            scratch,
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
        let mut ws = Q8Workspace::new();
        
        for layer in &self.layers {
            let residual = hidden.clone();
            hidden = rms_norm_tensor(&hidden, layer.attn_norm.weight(), layer.attn_norm.eps())?;

            // Flatten batch*seq_len for 2D matmul
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;

            // QKV projections using Q8_0 quantized matmul
            let q = q8_linear(&layer.attn_q, &h_2d, &mut ws)?; // [B*T, head_dim_sum]
            let k = q8_linear(&layer.attn_k, &h_2d, &mut ws)?; // [B*T, n_kv*hd]
            let v = q8_linear(&layer.attn_v, &h_2d, &mut ws)?; // [B*T, n_kv*hd]

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

            // Output projection
            let attn_flat = attn_out
                .permute((0, 2, 1, 3))?
                .reshape((batch * seq_len, head_dim_sum))?;
            let attn_proj = q8_linear(&layer.attn_o, &attn_flat, &mut ws)?; // [B*T, d_model]
            hidden = (residual + attn_proj.reshape((batch, seq_len, d_model))?)?;

            // SwiGLU FFN (fused gate+up quantize)
            let residual = hidden.clone();
            hidden = rms_norm_tensor(&hidden, layer.ffn_norm.weight(), layer.ffn_norm.eps())?;
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;
            let gu = q8_linear_multi(&[&layer.ffn_gate, &layer.ffn_up], &h_2d, &mut ws)?;
            let gate = candle_nn::ops::silu(&gu[0])?;
            let up = gu[1].clone();
            let hid_2d = (gate * up)?;
            let hid_out = q8_linear(&layer.ffn_down, &hid_2d, &mut ws)?; // [B*T, d_model]
            hidden = (residual + hid_out.reshape((batch, seq_len, d_model))?)?;
        }

        // Final norm + output projection
        hidden = rms_norm_tensor(&hidden, self.output_norm.weight(), self.output_norm.eps())?;
        let logits = match &self.output {
            Some(qm) => linear_fwd(qm, &hidden)?,
            None => linear_fwd(&self.token_embd, &hidden)?,
        };

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
        let mut ws = Q8Workspace::new();
        
        for layer in &self.layers {
            let residual = hidden.clone();
            hidden = rms_norm_tensor(&hidden, layer.attn_norm.weight(), layer.attn_norm.eps())?;

            // Flatten batch*seq_len for 2D matmul
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;

            // QKV projections (fused quantize)
            let qkv = q8_linear_multi(
                &[&layer.attn_q, &layer.attn_k, &layer.attn_v],
                &h_2d, &mut ws,
            )?;
            let q = &qkv[0]; let k = &qkv[1]; let v = &qkv[2];

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

            // Output projection
            let attn_flat = attn_out
                .permute((0, 2, 1, 3))?
                .reshape((batch * seq_len, head_dim_sum))?;
            let attn_proj = q8_linear(&layer.attn_o, &attn_flat, &mut ws)?; // [B*T, d_model]
            hidden = (residual + attn_proj.reshape((batch, seq_len, d_model))?)?;

            // SwiGLU FFN (fused gate+up quantize)
            let residual = hidden.clone();
            hidden = rms_norm_tensor(&hidden, layer.ffn_norm.weight(), layer.ffn_norm.eps())?;
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;
            let gu = q8_linear_multi(&[&layer.ffn_gate, &layer.ffn_up], &h_2d, &mut ws)?;
            let gate = candle_nn::ops::silu(&gu[0])?;
            let up = gu[1].clone();
            let hid_2d = (gate * up)?;
            let hid_out = q8_linear(&layer.ffn_down, &hid_2d, &mut ws)?; // [B*T, d_model]
            hidden = (residual + hid_out.reshape((batch, seq_len, d_model))?)?;
        }

        // Final norm
        let hidden_normed = rms_norm_tensor(&hidden, self.output_norm.weight(), self.output_norm.eps())?;

        // LM head (output projection) — reuse token_embd if no separate output weight
        let logits = match &self.output {
            Some(qm) => linear_fwd(qm, &hidden_normed)?,
            None => linear_fwd(&self.token_embd, &hidden_normed)?,
        };

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
        let mut ws = Q8Workspace::new();
        
        for layer in &self.layers {
            let residual = hidden.clone();
            hidden = rms_norm_tensor(&hidden, layer.attn_norm.weight(), layer.attn_norm.eps())?;

            // Flatten batch*seq_len for 2D matmul
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;

            // QKV projections (fused quantize)
            let qkv = q8_linear_multi(
                &[&layer.attn_q, &layer.attn_k, &layer.attn_v],
                &h_2d, &mut ws,
            )?;
            let q = &qkv[0]; let k = &qkv[1]; let v = &qkv[2];

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

            // Output projection
            let attn_flat = attn_out
                .permute((0, 2, 1, 3))?
                .reshape((batch * seq_len, head_dim_sum))?;
            let attn_proj = q8_linear(&layer.attn_o, &attn_flat, &mut ws)?; // [B*T, d_model]
            hidden = (residual + attn_proj.reshape((batch, seq_len, d_model))?)?;

            // SwiGLU FFN (fused gate+up quantize)
            let residual = hidden.clone();
            hidden = rms_norm_tensor(&hidden, layer.ffn_norm.weight(), layer.ffn_norm.eps())?;
            let h_2d = hidden.reshape((batch * seq_len, d_model))?;
            let gu = q8_linear_multi(&[&layer.ffn_gate, &layer.ffn_up], &h_2d, &mut ws)?;
            let gate = candle_nn::ops::silu(&gu[0])?;
            let up = gu[1].clone();
            let hid_2d = (gate * up)?;
            let hid_out = q8_linear(&layer.ffn_down, &hid_2d, &mut ws)?; // [B*T, d_model]
            hidden = (residual + hid_out.reshape((batch, seq_len, d_model))?)?;
        }

        // Final norm + output projection
        hidden = rms_norm_tensor(&hidden, self.output_norm.weight(), self.output_norm.eps())?;
        Ok(hidden)
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
        let logits = linear_fwd(head, hidden)?;
        Ok(logits)
    }

    /// Embed a codebook 0 token ID into `[d_model]` (f32, no Tensor).
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

    /// Embed a codebook 0 token ID into `[d_model]` (f32, no Tensor).
    ///
    /// This flattens the codec_embd to row-major on every call. For a
    /// zero-alloc version, cache the flattened table.
    pub fn embed_codebook0_f32(&self, token_id: u32) -> anyhow::Result<Vec<f32>> {
        let emb = self.codec_embd.as_ref()
            .ok_or_else(|| anyhow::anyhow!("codec_embd not loaded"))?;
        let d_model = self.config.d_model;
        let flat = if emb.dims()[0] == d_model && emb.dims().len() >= 2 && emb.dims()[1] != d_model {
            emb.t()?.contiguous()?.flatten_all()?.to_vec1()?
        } else {
            emb.flatten_all()?.to_vec1()?
        };
        Ok(embed_row_f32(&flat, token_id, d_model))
    }

    /// Process one token through all decoder layers with flat KV cache (incremental decode).
    ///
    /// `x`: `[batch, 1, d_model]` — single token embedding.
    /// `cache`: mutable per-layer flat KV cache (pre-allocated f32 buffers).
    /// `cos_full`, `sin_full`: `[1, 1, max_seq, head_dim]` — precomputed RoPE
    ///   for all positions; the current position is `cache.current_len()`.
    ///
    /// Returns output-normed hidden state: `[batch, 1, d_model]`.
    pub fn forward_step(
        &mut self,
        x: &Tensor,
        cache: &mut KvCacheFlat,
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
        let device = self.device().clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            hidden = rms_norm_tensor(&hidden, layer.attn_norm.weight(), layer.attn_norm.eps())?;

            // QKV projections (fused quantize): [B, d_model] → [B, ...]
            let h_2d = hidden.reshape((batch, d_model))?;
            let qkv = q8_linear_multi(
                &[&layer.attn_q, &layer.attn_k, &layer.attn_v],
                &h_2d, &mut self.q8_ws,
            )?;
            let q = &qkv[0]; let k = &qkv[1]; let v = &qkv[2];

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

            // Extract K,V as f32 slices and append to flat cache
            let k_flat = k.flatten_all()?.to_vec1::<f32>()?;
            let v_flat = v.flatten_all()?.to_vec1::<f32>()?;
            cache.append(i, &k_flat, &v_flat);

            // GQA attention using flat buffer (no Tensor::to_vec1 on the cache)
            let attn_out = attention_gqa_flat(
                &q,
                cache.k_slice(i),
                cache.v_slice(i),
                n_heads, n_kv, pos + 1, hd, cfg.max_seq_len,
                &device,
            )?;

            // Output projection
            let attn_out = attn_out
                .permute((0, 2, 1, 3))?
                .reshape((batch, head_dim_sum))?;
            let attn_proj = q8_linear(&layer.attn_o, &attn_out, &mut self.q8_ws)?;
            hidden = (residual + attn_proj.reshape((batch, 1, d_model))?)?;

            // SwiGLU FFN (fused gate+up quantize)
            let residual = hidden.clone();
            hidden = rms_norm_tensor(&hidden, layer.ffn_norm.weight(), layer.ffn_norm.eps())?;
            let h_2d = hidden.reshape((batch, d_model))?;
            let gu = q8_linear_multi(&[&layer.ffn_gate, &layer.ffn_up], &h_2d, &mut self.q8_ws)?;
            let gate = candle_nn::ops::silu(&gu[0])?;
            let up = gu[1].clone();
            let hid_2d = (gate * up)?;
            let hid_out = q8_linear(&layer.ffn_down, &hid_2d, &mut self.q8_ws)?;
            hidden = (residual + hid_out.reshape((batch, 1, d_model))?)?;
        }

        // Final output norm
        let hidden = rms_norm_tensor(&hidden, self.output_norm.weight(), self.output_norm.eps())?;
        cache.advance_pos();
        Ok(hidden)
    }

    /// Fused single-token forward pass — zero Tensor round-trips on the hot path.
    ///
    /// Pure f32 operations on pre-allocated scratch buffers. Bypasses Tensor
    /// creation, reshape, permute, narrow, and to_vec1 for every per-layer
    /// operation. Only the final result Vec can be wrapped into a Tensor
    /// by the caller.
    ///
    /// `x`: `[d_model]` — single-token hidden state.
    /// `cache`: per-layer flat KV cache.
    /// `pos`: current sequence position (0-based).
    /// `cos`, `sin`: `[head_dim]` — RoPE cos/sin at position `pos`.
    ///
    /// Returns `[d_model]` — output-normed hidden state.
    pub fn forward_step_fused(
        &mut self,
        x: &[f32],
        cache: &mut KvCacheFlat,
        pos: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Vec<f32> {
        let cfg = &self.config;
        let d_model = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim();
        let eps = cfg.norm_eps;
        let qk_eps = eps; // QK-norm eps matches norm eps

        // Copy input into the main state buffer `h`.
        // `h` is recycled from `scratch.normed` (first use, content overwritten).
        let scratch = &mut self.scratch;
        let h = &mut scratch.normed;
        h.copy_from_slice(x);

        for (i, layer) in self.layers.iter().enumerate() {
            // ── Save residual ────────────────────────────────────────────
            scratch.residual.copy_from_slice(h);

            // ── Pre-attention RMS norm: h → scratch.normed ──────────────
            // (h already lives in scratch.normed from the previous iteration
            //  or the initial copy; compute norm in-place.)
            rms_norm_f32_inplace(h, &layer.attn_norm_w, eps);

            // ── QKV projections (quantize-once multi gemv) ────────────────
            Q8Weights::gemv_multi_into(
                &[&layer.attn_q, &layer.attn_k, &layer.attn_v],
                &mut [&mut scratch.q_buf, &mut scratch.k_buf, &mut scratch.v_buf],
                h,
                &mut self.q8_ws,
            );

            // ── Per-head QK-norm (parallel) ──────────────────────────────
            let q = per_head_rms_norm_f32_par(&scratch.q_buf, &layer.attn_q_norm_w, n_heads, hd, qk_eps);
            let k = per_head_rms_norm_f32_par(&scratch.k_buf, &layer.attn_k_norm_w, n_kv, hd, qk_eps);

            // ── RoPE (parallel) ──────────────────────────────────────────
            let q = rope_f32_par(&q, cos, sin, n_heads, hd);
            let k = rope_f32_par(&k, cos, sin, n_kv, hd);

            // ── KV cache append ──────────────────────────────────────────
            cache.append(i, &k, &scratch.v_buf);

            // ── Attention (parallel) ─────────────────────────────────────
            let attn_out_raw = attention_f32_par(
                &q,
                cache.k_slice(i),
                cache.v_slice(i),
                n_heads,
                n_kv,
                pos + 1, // kv_len = positions written so far
                hd,
                cache.head_stride(), // stride between heads in the flat buffer
            );

            // ── Output projection into q_buf (reuse — attn_dim == d_model) ─
            layer.attn_o.gemv_into(&attn_out_raw, &mut self.q8_ws, &mut scratch.q_buf);

            // ── Residual: h = residual + q_buf (parallel) ────────────────
            h.par_iter_mut()
                .zip(scratch.residual.par_iter())
                .zip(scratch.q_buf.par_iter())
                .for_each(|((h_j, &r), &q)| {
                    *h_j = r + q;
                });

            // ── Save residual again for FFN ──────────────────────────────
            scratch.residual.copy_from_slice(h);

            // ── Pre-FFN RMS norm: h → scratch.normed (in-place) ─────────
            rms_norm_f32_inplace(h, &layer.ffn_norm_w, eps);

            // ── FFN gate + up (quantize-once multi gemv) ──────────────────
            Q8Weights::gemv_multi_into(
                &[&layer.ffn_gate, &layer.ffn_up],
                &mut [&mut scratch.gate_buf, &mut scratch.up_buf],
                h,
                &mut self.q8_ws,
            );

            // ── SiLU(gate) * up → scratch.ffn_mid (parallel) ─────────────
            let ffn_dim = scratch.gate_buf.len();
            let scratch_ffn = &mut scratch.ffn_mid[..ffn_dim];
            let silu_gate = silu_f32_par(&scratch.gate_buf);
            scratch_ffn.par_iter_mut()
                .zip(silu_gate.par_iter())
                .zip(scratch.up_buf.par_iter())
                .for_each(|((dst, &g_act), &u)| {
                    *dst = g_act * u;
                });

            // ── Down projection into q_buf (reuse — d_model) ─────────────
            layer.ffn_down.gemv_into(scratch_ffn, &mut self.q8_ws, &mut scratch.q_buf);

            // ── Residual: h = residual + q_buf (parallel) ────────────────
            h.par_iter_mut()
                .zip(scratch.residual.par_iter())
                .zip(scratch.q_buf.par_iter())
                .for_each(|((h_j, &r), &q)| {
                    *h_j = r + q;
                });
        }

        // ── Final output norm ────────────────────────────────────────────
        // h currently holds the post-FFN hidden state. Compute final norm
        // into a newly allocated Vec (the only allocation on the fused path).
        rms_norm_f32(h, &self.output_norm_w, eps)
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

    /// Access the codec embedding table for codebook 0.
    pub fn codec_embd(&self) -> Option<&Tensor> {
        self.codec_embd.as_ref()
    }

    /// Look up a single row in the codec embedding table, return a flat `[hidden]` f32 vec.
    pub fn lookup_codec_row(&self, row_id: u32) -> anyhow::Result<Vec<f32>> {
        let emb = self.codec_embd.as_ref()
            .ok_or_else(|| anyhow::anyhow!("codec_embd not loaded"))?;
        let d_model = self.config.d_model;
        let emb_w = if emb.dims()[0] == d_model {
            emb.t()?
        } else {
            emb.clone()
        };
        let ids = Tensor::from_slice(&[row_id], (1,), &self.device)?;
        let row = emb_w.index_select(&ids, 0)?; // [1, d_model]
        Ok(row.flatten_all()?.to_vec1()?)
    }

    /// Embed text token IDs and return a flat `[N * d_model]` f32 vec.
    /// Convenience wrapper over `embed_text` that skips tensor-to-vec conversion.
    pub fn embed_text_to_vec(&self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let t = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
        let emb = self.embed_text(&t)?; // [1, N, d_model]
        Ok(emb.flatten_all()?.to_vec1()?)
    }
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

/// Pre-compute RoPE cos/sin as flat f32 arrays.
///
/// Returns `(cos, sin)` each of length `max_s * head_dim` with layout
/// `[pos, d]` where `cos[pos * hd + d]` = cos(pos * inv_freq[d % half])
/// (interleaved pairs for 1D NEOX RoPE).
pub fn precompute_cos_sin_flat(cfg: &ModelConfig, max_s: usize) -> (Vec<f32>, Vec<f32>) {
    let hd = cfg.head_dim();
    let half = hd / 2;
    let inv_freq: Vec<f64> = (0..hd)
        .step_by(2)
        .map(|i| 1.0_f64 / cfg.rope_theta.powf(i as f64 / hd as f64))
        .collect();
    let mut cos = vec![0.0f32; max_s * hd];
    let mut sin = vec![0.0f32; max_s * hd];
    for pos in 0..max_s {
        let base = pos * hd;
        for d in 0..half {
            let angle = pos as f64 * inv_freq[d];
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            cos[base + d] = c;
            cos[base + d + half] = c;
            sin[base + d] = s;
            sin[base + d + half] = s;
        }
    }
    (cos, sin)
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

/// Extract a single row from a flat row-major weight matrix `[rows * cols]`.
///
/// Returns `cols`-length Vec.
pub(crate) fn embed_row_f32(weight: &[f32], token: u32, cols: usize) -> Vec<f32> {
    let offset = (token as usize) * cols;
    weight[offset..offset + cols].to_vec()
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
