//! Full MTP code predictor with transformer layers and sequential causal prediction.
//!
//! Architecture (matching C++ `code-predictor-forward.h`):
//!   - 5 transformer layers, pre-norm, GQA (16 Q heads / 8 KV heads), QK-norm,
//!     1D NEOX RoPE (theta=1e6), SwiGLU MLP
//!   - Operates at `pred_hidden` dimension (1024 for 1.7B model)
//!   - Receives input at `talker_hidden` dimension (2048) via `mtp_proj`
//!   - Per-frame sequential prediction:
//!     - Prefill 2 positions: talker hidden state (pos 0), c0 embedding (pos 1)
//!     - Decode 14 positions: one per acoustic codebook (pos 2..15)
//!     - lm_head[g] on position (g+1) output → sample acoustic codebook (g+1)
//!     - codec_embd[g-1] embeds the predicted code as input for next position
//!
//! Tensor naming (from the talker GGUF):
//!   code_pred.output_norm.weight
//!   code_pred.mtp_proj.weight / .bias
//!   code_pred.codec_embd.{0..13}.weight        (14 tables for c1..c14)
//!   code_pred.lm_head.{0..14}.weight            (15 heads for c1..c15)
//!   code_pred.blk.{i}.attn_norm.weight
//!   code_pred.blk.{i}.ffn_norm.weight
//!   code_pred.blk.{i}.attn_q.weight
//!   code_pred.blk.{i}.attn_k.weight
//!   code_pred.blk.{i}.attn_v.weight
//!   code_pred.blk.{i}.attn_output.weight
//!   code_pred.blk.{i}.attn_q_norm.weight        (QK-norm)
//!   code_pred.blk.{i}.attn_k_norm.weight        (QK-norm)
//!   code_pred.blk.{i}.ffn_gate.weight           (SwiGLU)
//!   code_pred.blk.{i}.ffn_up.weight
//!   code_pred.blk.{i}.ffn_down.weight

use std::fs::File;

use candle_core::quantized::gguf_file::Content;
use candle_core::{Device, Tensor, D};
use candle_nn::RmsNorm;
use rand::SeedableRng;

use crate::sampling;
use crate::custom_ops::{
    attention_f32, per_head_rms_norm_f32, rms_norm_f32, rope_f32, silu_f32,
};
use crate::qgemv::{q8_linear, q8_linear_multi, Q8Weights, Q8Workspace};
use crate::talker::{
    embed_token, linear_fwd, DecoderLayer,
};

/// Type alias: a frame of acoustic code token IDs (one per codebook level).
pub type CodeFrame = Vec<u32>;

/// Full code predictor with transformer layers and per-frame KV cache.
///
/// Predicts acoustic codebooks 1..N (codebook 0 is handled by the talker).
pub struct CodePredictor {
    // ── config ────────────────────────────────────────────────────────────
    /// Number of acoustic codebooks (= total_code_groups - 1, typically 15).
    num_acoustic: usize,
    /// Predictor hidden dimension (e.g., 1024 for 1.7B model).
    pred_hidden: usize,
    /// Talker hidden dimension (e.g., 2048 for 1.7B model).
    talker_hidden: usize,
    /// Number of query heads (16 for 1.7B).
    n_q_heads: usize,
    /// Number of key/value heads (8 for 1.7B).
    n_kv_heads: usize,
    /// Head dimension (128).
    head_dim: usize,
    /// Vocabulary size for code tokens (e.g., 2048).
    vocab_size: usize,
    /// Number of transformer layers (5 for 1.7B).
    n_layers: usize,
    /// Maximum sequence length per frame (num_acoustic + 1 = 16).
    max_seq_len: usize,
    /// FFN intermediate dimension (e.g., 3072 for 1.7B model).
    ffn_dim: usize,

    // ── weights ───────────────────────────────────────────────────────────
    /// Transformer decoder layers (linear weights as Q8_0 quantized).
    layers: Vec<DecoderLayer>,
    /// Per-codebook embedding tables for acoustic codebooks 1..14.
    /// codec_embd[g-1] embeds the predicted code of book g → talker_hidden.
    /// Shape: [vocab_size, talker_hidden] = [2048, talker_hidden].
    codec_embd: Vec<Tensor>,
    /// Linear heads, one per acoustic codebook (15 heads for c1..c15).
    /// lm_head[g] maps pred_hidden → vocab_size logits for codebook g+1.
    /// Shape: [vocab_size, pred_hidden] = [2048, 1024].
    lm_heads: Vec<Tensor>,
    /// MTP projection: talker_hidden → pred_hidden.
    /// Shape: [pred_hidden, talker_hidden] = [1024, 2048].
    mtp_proj_w: Option<Tensor>,
    mtp_proj_b: Option<Tensor>,
    /// Output norm weight + eps (applied after last transformer layer).
    output_norm_w: Option<Tensor>,
    output_norm_eps: f64,

    // ── precomputed RoPE ──────────────────────────────────────────────────
    /// Cosine table. Shape: [1, 1, max_seq_len, head_dim].
    cos: Tensor,
    /// Sine table. Shape: [1, 1, max_seq_len, head_dim].
    sin: Tensor,

    // ── mutable per-frame state ───────────────────────────────────────────
    /// Flat f32 KV cache: per-layer vectors of flattened [n_kv_heads, kv_len, head_dim].
    /// Avoids Tensor::cat overhead and keeps data as f32 slices for direct use
    /// by attention_f32 (no to_vec1 copy needed).
    k_cache_data: Vec<Vec<f32>>,
    v_cache_data: Vec<Vec<f32>>,

    /// Device handle.
    device: Device,
}

impl CodePredictor {
    /// Load code predictor weights from the talker GGUF content.
    ///
    /// # Arguments
    /// * `content` — parsed GGUF content (from the talker file).
    /// * `file`    — the open GGUF file (for tensor reading).
    /// * `device`  — target device.
    pub fn from_gguf(
        content: &Content,
        file: &mut File,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let metadata = &content.metadata;

        // ── read metadata ────────────────────────────────────────────────
        let num_code_groups = metadata
            .get("qwen3-tts.num_code_groups")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(16) as usize;
        let num_acoustic = num_code_groups.saturating_sub(1);
        if num_acoustic == 0 {
            anyhow::bail!("num_code_groups <= 1, no acoustic codebooks to predict");
        }

        let pred_hidden = metadata
            .get("qwen3-tts.code_pred.embedding_length")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(1024);

        let talker_hidden = metadata
            .get("qwen3-tts.talker.embedding_length")
            .and_then(|v| v.to_u32().ok())
            .or_else(|| {
                metadata
                    .get("llama.embedding_length")
                    .and_then(|v| v.to_u32().ok())
            })
            .unwrap_or(2048) as usize;

        let n_layers = metadata
            .get("qwen3-tts.code_pred.block_count")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(5) as usize;

        let n_q_heads = metadata
            .get("qwen3-tts.code_pred.attention.head_count")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(16) as usize;

        let n_kv_heads = metadata
            .get("qwen3-tts.code_pred.attention.head_count_kv")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(8) as usize;

        let head_dim = metadata
            .get("qwen3-tts.code_pred.attention.key_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(128) as usize;

        let vocab_size = metadata
            .get("qwen3-tts.code_pred.vocab_size")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(2048) as usize;

        let rope_theta = metadata
            .get("qwen3-tts.code_pred.rope.freq_base")
            .and_then(|v| v.to_f64().ok())
            .unwrap_or(1_000_000.0);

        let norm_eps = metadata
            .get("qwen3-tts.code_pred.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6) as f64;

        // Sequence length per frame = num_acoustic + 1 (prefill 2 + decode N-1)
        let max_seq_len = num_acoustic + 1; // 16 for num_acoustic=15

        // ── helpers: load tensors (pass `f: &mut File` to reborrow per call) ──
        let load_f32 = |name: &str, f: &mut File| -> anyhow::Result<Tensor> {
            let qt = content.tensor(f, name, device).map_err(|e| {
                anyhow::anyhow!("missing code_pred tensor {name}: {e}")
            })?;
            qt.dequantize(&device)
                .map_err(|e| anyhow::anyhow!("dequantize {name}: {e}"))
        };

        // ── transformer layers ───────────────────────────────────────────
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let blk = |name: &str| -> String { format!("code_pred.blk.{i}.{name}") };
            // Helper: load Q8_0 quantized weight directly from GGUF.
            let load_q8 = |name: &str, f: &mut File| -> anyhow::Result<Q8Weights> {
                Q8Weights::from_gguf(content, f, name)
            };
            layers.push(DecoderLayer {
                attn_norm: RmsNorm::new(load_f32(&blk("attn_norm.weight"), &mut *file)?, norm_eps),
                attn_q: load_q8(&blk("attn_q.weight"), &mut *file)?,
                attn_k: load_q8(&blk("attn_k.weight"), &mut *file)?,
                attn_v: load_q8(&blk("attn_v.weight"), &mut *file)?,
                attn_o: load_q8(&blk("attn_output.weight"), &mut *file)?,
                attn_q_norm: RmsNorm::new(load_f32(&blk("attn_q_norm.weight"), &mut *file)?, norm_eps),
                attn_k_norm: RmsNorm::new(load_f32(&blk("attn_k_norm.weight"), &mut *file)?, norm_eps),
                ffn_norm: RmsNorm::new(load_f32(&blk("ffn_norm.weight"), &mut *file)?, norm_eps),
                ffn_gate: load_q8(&blk("ffn_gate.weight"), &mut *file)?,
                ffn_up: load_q8(&blk("ffn_up.weight"), &mut *file)?,
                ffn_down: load_q8(&blk("ffn_down.weight"), &mut *file)?,
            });
        }

        // ── MTP projection ───────────────────────────────────────────────
        let mtp_proj_w =
            if content.tensor_infos.contains_key("code_pred.mtp_proj.weight") {
                Some(load_f32("code_pred.mtp_proj.weight", &mut *file)?)
            } else {
                None
            };
        let mtp_proj_b =
            if content.tensor_infos.contains_key("code_pred.mtp_proj.bias") {
                Some(load_f32("code_pred.mtp_proj.bias", &mut *file)?)
            } else {
                None
            };

        // ── output norm ──────────────────────────────────────────────────
        let (output_norm_w, output_norm_eps_val) =
            if content.tensor_infos.contains_key("code_pred.output_norm.weight") {
                (Some(load_f32("code_pred.output_norm.weight", &mut *file)?), norm_eps)
            } else {
                (None, norm_eps)
            };

        // ── per-codebook embedding tables ────────────────────────────────
        // Load up to (num_acoustic - 1) tables for codes 1..14.
        let num_embd = num_acoustic.saturating_sub(1); // 14
        let mut codec_embd = Vec::with_capacity(num_embd);
        for g in 0..num_embd {
            let name = format!("code_pred.codec_embd.{g}.weight");
            if content.tensor_infos.contains_key(&name) {
                codec_embd.push(load_f32(&name, &mut *file)?);
            } else {
                anyhow::bail!("missing {name} — expected {num_embd} embedding tables");
            }
        }

        // ── linear heads ─────────────────────────────────────────────────
        let mut lm_heads = Vec::with_capacity(num_acoustic);
        for g in 0..num_acoustic {
            let name = format!("code_pred.lm_head.{g}.weight");
            if !content.tensor_infos.contains_key(&name) {
                anyhow::bail!(
                    "missing {name} — expected {num_acoustic} lm heads"
                );
            }
            lm_heads.push(load_f32(&name, &mut *file)?);
        }

        // ── precompute RoPE cos/sin ──────────────────────────────────────
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| (1.0_f64 / rope_theta.powf(i as f64 / head_dim as f64)) as f32)
            .collect();
        let n_freq = inv_freq.len();
        let inv_freq_t = Tensor::from_slice(&inv_freq, (n_freq,), &device)?;
        let pos: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let pos_t = Tensor::from_slice(&pos, (max_seq_len,), &device)?;
        let freqs = pos_t.unsqueeze(1)?.matmul(&inv_freq_t.unsqueeze(0)?)?;
        let mut cos_v = freqs.cos()?;
        let mut sin_v = freqs.sin()?;
        // Interleave pairs → [max_seq_len, head_dim]
        cos_v = interleave(&cos_v, 2)?;
        sin_v = interleave(&sin_v, 2)?;
        // Add batch + head dims → [1, 1, max_seq_len, head_dim]
        let cos = cos_v.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin_v.unsqueeze(0)?.unsqueeze(0)?;

        // FFN intermediate dimension from first layer's ffn_gate (output rows).
        let ffn_dim = layers
            .first()
            .map(|l| l.ffn_gate.out_features())
            .unwrap_or(pred_hidden * 3);

        // Pre-allocate flat f32 KV caches (per-layer, empty initially).
        let k_cache_data = vec![Vec::new(); n_layers];
        let v_cache_data = vec![Vec::new(); n_layers];

        Ok(Self {
            num_acoustic,
            pred_hidden,
            talker_hidden,
            n_q_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            n_layers,
            max_seq_len,
            ffn_dim,
            layers,
            codec_embd,
            lm_heads,
            mtp_proj_w,
            mtp_proj_b,
            output_norm_w,
            output_norm_eps: output_norm_eps_val,
            cos,
            sin,
            k_cache_data,
            v_cache_data,
            device: device.clone(),
        })
    }

    // ── private helpers ─────────────────────────────────────────────────

    /// Project `[batch, seq, talker_hidden]` → `[batch, seq, pred_hidden]` via `mtp_proj`.
    fn project(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mut h = x.clone();
        if let Some(ref w) = self.mtp_proj_w {
            h = linear_fwd(w, &h)?;
            if let Some(ref b) = self.mtp_proj_b {
                h = h.broadcast_add(b)?;
            }
        }
        Ok(h)
    }

    /// Apply lm_head `g` to hidden state at `pred_hidden` dimension.
    ///
    /// `h`: `[1, 1, pred_hidden]`.
    /// Returns `Vec<f32>` logits of length `vocab_size`.
    fn apply_lm_head(&self, g: usize, h: &Tensor) -> anyhow::Result<Vec<f32>> {
        let head = &self.lm_heads[g]; // QMatMul [vocab_size, pred_hidden]
        let logits_t = linear_fwd(head, h)?; // [1, 1, vocab_size]
        Ok(logits_t.flatten_all()?.to_vec1()?)
    }

    /// Forward one position through all transformer layers, updating the KV cache.
    ///
    /// All intermediate operations use raw f32 slices — no Tensor dispatch
    /// for norm, RoPE, or attention. Only the I/O boundaries (input Tensor,
    /// q8_linear projections, output Tensor) create tensors.
    ///
    /// `pos`: absolute position in the sequence (0-based within this frame).
    /// `pred_input`: `[1, 1, pred_hidden]` — already projected.
    fn forward_at_pos(&mut self, pos: usize, pred_input: &Tensor) -> anyhow::Result<Tensor> {
        let mut x = pred_input.clone();
        let (batch, _one, _d_model) = x.dims3()?;

        // Derive head dimensions from the actual Q8 weight shapes (more
        // reliable than metadata, which may differ per model variant).
        let attn_dim = self.layers[0].attn_q.out_features(); // n_q_heads * head_dim
        let kv_dim = self.layers[0].attn_k.out_features();   // n_kv_heads * head_dim
        let n_qh = self.n_q_heads;
        let n_kvh = self.n_kv_heads;
        let hd = attn_dim / n_qh;
        debug_assert_eq!(hd * n_kvh, kv_dim, "predictor KV dim mismatch");

        // RoPE cos/sin at this position — extract once for all layers
        let cos_t = self.cos.narrow(D::Minus2, pos, 1)?; // [1, 1, 1, hd]
        let sin_t = self.sin.narrow(D::Minus2, pos, 1)?;
        let cos_f32: Vec<f32> = cos_t.flatten_all()?.to_vec1()?;
        let sin_f32: Vec<f32> = sin_t.flatten_all()?.to_vec1()?;

        let mut ws = Q8Workspace::new();
        let eps = 1e-6_f64; // norm eps (same as talker default)

        for i in 0..self.n_layers {
            let layer = &self.layers[i];
            let residual = x.clone();

            // ── Attn norm (custom f32) ───────────────────────────────────
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let w_an = layer.attn_norm.weight().flatten_all()?.to_vec1::<f32>()?;
            let x_normed = rms_norm_f32(&x_flat, &w_an, eps);
            let x_t = Tensor::from_slice(&x_normed, (batch, 1, self.pred_hidden), &self.device)?;

            // ── QKV projections (fused quantize) ──────────────────────────
            let h_2d = x_t.reshape((batch, self.pred_hidden))?;
            let qkv = q8_linear_multi(
                &[&layer.attn_q, &layer.attn_k, &layer.attn_v],
                &h_2d, &mut ws,
            )?;

            // Extract f32 data — we'll operate directly on these slices
            let q_f32: Vec<f32> = qkv[0].flatten_all()?.to_vec1()?; // [attn_dim]
            let k_f32: Vec<f32> = qkv[1].flatten_all()?.to_vec1()?; // [kv_dim]
            let v_f32: Vec<f32> = qkv[2].flatten_all()?.to_vec1()?; // [kv_dim]

            // ── Per-head QK-norm (f32 slice) ──────────────────────────────
            let w_qn = layer.attn_q_norm.weight().flatten_all()?.to_vec1::<f32>()?;
            let w_kn = layer.attn_k_norm.weight().flatten_all()?.to_vec1::<f32>()?;
            let q_normed = per_head_rms_norm_f32(&q_f32, &w_qn, n_qh, hd, eps);
            let k_normed = per_head_rms_norm_f32(&k_f32, &w_kn, n_kvh, hd, eps);

            // ── 1D NEOX RoPE (f32 slice) ─────────────────────────────────
            let q_rope = rope_f32(&q_normed, &cos_f32, &sin_f32, n_qh, hd);
            let k_rope = rope_f32(&k_normed, &cos_f32, &sin_f32, n_kvh, hd);

            // ── KV cache append (flat Vec, no Tensor::cat) ───────────────
            self.k_cache_data[i].extend_from_slice(&k_rope);
            self.v_cache_data[i].extend_from_slice(&v_f32); // V is not normed/ROPEd

            // ── GQA attention (f32 slice, no Tensor wrapper) ──────────────
            let kv_len = self.k_cache_data[i].len() / kv_dim;
            let attn_out_f32 = attention_f32(
                &q_rope, &self.k_cache_data[i], &self.v_cache_data[i],
                n_qh, n_kvh, kv_len, hd,
            );
            let attn_out = Tensor::from_slice(&attn_out_f32, (batch, 1, attn_dim), &self.device)?;

            // ── Output projection ────────────────────────────────────────
            let attn_proj = q8_linear(&layer.attn_o, &attn_out, &mut ws)?;
            x = (residual + attn_proj.reshape((batch, 1, self.pred_hidden))?)?;

            // ── SwiGLU FFN (fused gate+up quantize) ─────────────────────
            let residual = x.clone();
            let x2_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let w_fn = layer.ffn_norm.weight().flatten_all()?.to_vec1::<f32>()?;
            let x2_normed = rms_norm_f32(&x2_flat, &w_fn, eps);
            let x2_t = Tensor::from_slice(&x2_normed, (batch, 1, self.pred_hidden), &self.device)?;
            let h2_2d = x2_t.reshape((batch, self.pred_hidden))?;

            let gu = q8_linear_multi(&[&layer.ffn_gate, &layer.ffn_up], &h2_2d, &mut ws)?;
            let gate_f32: Vec<f32> = gu[0].flatten_all()?.to_vec1()?;
            let up_f32: Vec<f32> = gu[1].flatten_all()?.to_vec1()?;
            let gate_act = silu_f32(&gate_f32);
            let hid_f32: Vec<f32> = gate_act.iter().zip(&up_f32).map(|(g, u)| g * u).collect();
            let hid_t = Tensor::from_slice(&hid_f32, (batch, 1, self.ffn_dim), &self.device)?;

            let hid_out = q8_linear(&layer.ffn_down, &hid_t, &mut ws)?;
            x = (residual + hid_out.reshape((batch, 1, self.pred_hidden))?)?;
        }

        // Final output norm
        if let Some(ref w) = self.output_norm_w {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let w_o = w.flatten_all()?.to_vec1::<f32>()?;
            let y = rms_norm_f32(&x_flat, &w_o, self.output_norm_eps);
            x = Tensor::from_slice(&y, (batch, 1, self.pred_hidden), &self.device)?;
        }

        Ok(x)
    }

    // ── public API ─────────────────────────────────────────────────────

    /// Predict a single audio code frame using temperature/top-k/top-p sampling.
    ///
    /// Per-frame architecture:
    ///   - **Prefill 2 positions**: talker hidden state (pos 0), c0 embedding (pos 1)
    ///   - **lm_head[0]** on position 1 output → sample acoustic codebook 1 (c1)
    ///   - **Decode 14 positions**: for g=1..14, embed c_g from `codec_embd[g-1]`,
    ///     forward at position g+1, apply `lm_head[g]` → sample c_{g+1}
    ///
    /// # Arguments
    /// * `talker_hidden` — last talker hidden state `[1, 1, talker_hidden]`
    /// * `c0_embed` — talker's embedding of codebook 0 token `[1, 1, talker_hidden]`
    /// * `temperature` — sampling temperature (0.0 = argmax)
    /// * `top_k` — optional top-k filter
    /// * `top_p` — optional top-p (nucleus) filter
    /// * `rng` — mutable RNG handle
    ///
    /// Returns a `CodeFrame` of `num_acoustic` code token IDs (c1..cN).
    pub fn predict_one_frame_sampled(
        &mut self,
        talker_hidden: &Tensor,
        c0_embed: &Tensor,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        rng: &mut impl rand::Rng,
    ) -> anyhow::Result<CodeFrame> {
        // Reset KV cache (fresh per frame) — just clear the Vecs, no alloc.
        for kc in &mut self.k_cache_data { kc.clear(); }
        for vc in &mut self.v_cache_data { vc.clear(); }

        // ── Prefill position 0 ───────────────────────────────────────────
        // Input: talker hidden state → project → forward (no lm_head here)
        let proj_0 = self.project(talker_hidden)?; // [1, 1, pred_hidden]
        let _ = self.forward_at_pos(0, &proj_0)?;

        // ── Prefill position 1 ───────────────────────────────────────────
        // Input: c0_embed → project → forward → lm_head[0] → sample c1
        let proj_1 = self.project(c0_embed)?; // [1, 1, pred_hidden]
        let h1 = self.forward_at_pos(1, &proj_1)?;
        let logits_0 = self.apply_lm_head(0, &h1)?;
        let (c1, _prob) =
            sampling::sample_token(&logits_0, temperature, top_k, top_p, rng);
        let mut codes: Vec<u32> = vec![c1];

        // ── Decode positions 2..(num_acoustic+1) ─────────────────────────
        // For g = 1..(num_acoustic-1):
        //   position = g + 1
        //   embed codes[g-1] (the just-predicted token) via codec_embd[g-1]
        //   project → forward → lm_head[g] → sample → push
        for g in 1..self.num_acoustic {
            let prev_token = codes[g - 1];
            // Embed the predicted code using the predictor's embedding table
            let emb = embed_token(
                &self.codec_embd[g - 1],
                prev_token,
                self.talker_hidden,
                &self.device,
            )?; // [1, 1, talker_hidden]
            let proj = self.project(&emb)?; // [1, 1, pred_hidden]
            let pos = g + 1; // positions 2..15
            let h = self.forward_at_pos(pos, &proj)?;
            let logits = self.apply_lm_head(g, &h)?;
            let (code, _prob) =
                sampling::sample_token(&logits, temperature, top_k, top_p, rng);
            codes.push(code);
        }

        debug_assert_eq!(codes.len(), self.num_acoustic);
        Ok(codes)
    }

    /// Predict a single audio code frame with argmax (fully deterministic).
    ///
    /// Convenience wrapper: calls `predict_one_frame_sampled` with `temperature=0.0`.
    pub fn predict_one_frame_argmax(
        &mut self,
        talker_hidden: &Tensor,
        c0_embed: &Tensor,
    ) -> anyhow::Result<CodeFrame> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        self.predict_one_frame_sampled(talker_hidden, c0_embed, 0.0, None, None, &mut rng)
    }

    /// Return the number of acoustic codebooks this predictor handles.
    #[must_use]
    pub fn num_acoustic(&self) -> usize {
        self.num_acoustic
    }

    /// Return the predictor's hidden dimension.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.pred_hidden
    }
}

// ── helpers ─────────────────────────────────────────────────────────────

/// Interleave adjacent pairs in the last dimension (for RoPE).
fn interleave(x: &Tensor, n: usize) -> anyhow::Result<Tensor> {
    let s = x.dims();
    let last = s[s.len() - 1];
    let x = x.unsqueeze(s.len())?;
    let mut shape = s.to_vec();
    shape.push(n);
    let x = x.expand(shape.as_slice())?;
    let mut out_shape = s.to_vec();
    out_shape[s.len() - 1] = last * n;
    Ok(x.reshape(out_shape.as_slice())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_frame_type() {
        let frame: CodeFrame = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(frame.len(), 15);
    }

    #[test]
    fn test_interleave_doubles_length() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev).unwrap();
        let r = interleave(&x, 2).unwrap();
        assert_eq!(r.dims(), &[2, 4]); // last dim doubled
    }
}
