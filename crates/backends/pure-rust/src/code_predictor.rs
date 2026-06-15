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
use candle_core::quantized::k_quants::{BlockQ8_0, GgmlType, QK8_0};
use candle_core::{Device, Tensor};
use candle_nn::RmsNorm;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::sampling;
use crate::custom_ops::{
    attention_f32_par, per_head_rms_norm_f32_par,
    rms_norm_f32, rms_norm_f32_inplace,
    rope_f32_par, silu_f32_par,
};
use crate::qgemv::{Q8Weights, Q8Workspace};
use crate::talker::{
    DecoderLayer, Talker,
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
    #[allow(dead_code)]
    vocab_size: usize,
    /// Number of transformer layers (5 for 1.7B).
    n_layers: usize,
    /// Maximum sequence length per frame (num_acoustic + 1 = 16).
    #[allow(dead_code)]
    max_seq_len: usize,
    /// FFN intermediate dimension (e.g., 3072 for 1.7B model).
    #[allow(dead_code)]
    ffn_dim: usize,

    // ── weights ───────────────────────────────────────────────────────────
    /// Transformer decoder layers (linear weights as Q8_0 quantized).
    layers: Vec<DecoderLayer>,
    /// Q8 quantized codec_embd tables — each [vocab_size, talker_hidden] row-major.
    codec_embd_q8: Vec<Q8Weights>,
    /// Q8 quantized lm_heads — each [vocab_size, pred_hidden] row-major.
    lm_heads_q8: Vec<Q8Weights>,
    /// Q8 quantized MTP projection weight — [pred_hidden, talker_hidden].
    mtp_proj_q8: Option<Q8Weights>,
    /// Flat MTP projection bias — [pred_hidden] or empty.
    mtp_proj_b_f32: Vec<f32>,
    /// Output norm eps (applied after last transformer layer).
    output_norm_eps: f64,

    // ── flat RoPE for fused path ──────────────────────────────────────────
    /// Flat cos table: [max_seq_len * head_dim].
    cos_f32: Vec<f32>,
    /// Flat sin table: [max_seq_len * head_dim].
    sin_f32: Vec<f32>,

    /// F32 view of output_norm weight (None if no separate output norm).
    output_norm_w_f32: Option<Vec<f32>>,

    // ── mutable per-frame state ───────────────────────────────────────────
    /// Flat f32 KV cache: per-layer vectors of flattened [n_kv_heads, kv_len, head_dim].
    /// Avoids Tensor::cat overhead and keeps data as f32 slices for direct use
    /// by attention_f32 (no to_vec1 copy needed).
    k_cache_data: Vec<Vec<f32>>,
    v_cache_data: Vec<Vec<f32>>,

    /// Persistent Q8 workspace (avoids re-allocation on the hot path).
    q8_ws: Q8Workspace,

    /// Pre-allocated scratch buffers for fused forward path.
    pred_scratch: PredScratch,

    // ── batched KV cache (for predict_n_frames_batched) ──────────────
    k_cache_batched: Vec<Vec<Vec<f32>>>, // [n_layers][m_max] Vec<f32>
    v_cache_batched: Vec<Vec<Vec<f32>>>,
    batch_scratch: BatchScratch,
}

/// Pre-allocated scratch buffers for CodePredictor's fused forward path.
struct PredScratch {
    /// Saved pre-norm hidden state for residual connection (pred_hidden).
    residual: Vec<f32>,
    /// In/out main state buffer (pred_hidden). Recycled as norm input/output.
    h: Vec<f32>,
    /// FFN intermediate: SiLU(gate) * up (ffn_dim).
    ffn_mid: Vec<f32>,

    // ── Pre-allocated GEMV output buffers (allocation-free hot path) ──
    /// Q projection output [attn_dim] (separate from q_buf because sizes differ).
    attn_q_buf: Vec<f32>,
    /// attn_o/ffn_down projection output [pred_hidden].
    q_buf: Vec<f32>,
    /// K projection output [kv_dim].
    k_buf: Vec<f32>,
    /// V projection output [kv_dim].
    v_buf: Vec<f32>,
    /// FFN gate projection output [ffn_dim].
    gate_buf: Vec<f32>,
    /// FFN up projection output [ffn_dim].
    up_buf: Vec<f32>,
}

impl PredScratch {
    fn new(pred_hidden: usize, attn_dim: usize, ffn_dim: usize, kv_dim: usize) -> Self {
        Self {
            residual: vec![0.0f32; pred_hidden],
            h: vec![0.0f32; pred_hidden],
            ffn_mid: vec![0.0f32; ffn_dim],
            attn_q_buf: vec![0.0f32; attn_dim],
            q_buf: vec![0.0f32; pred_hidden],
            k_buf: vec![0.0f32; kv_dim],
            v_buf: vec![0.0f32; kv_dim],
            gate_buf: vec![0.0f32; ffn_dim],
            up_buf: vec![0.0f32; ffn_dim],
        }
    }
}

/// Pre-allocated scratch buffers for batched (M>1) forward passes.
struct BatchScratch {
    m_max: usize,
    h: Vec<f32>,             // [m_max, pred_hidden]
    residual: Vec<f32>,      // [m_max, pred_hidden]
    q_buf: Vec<f32>,         // [m_max, pred_hidden] (O/down output reused here)
    attn_q_buf: Vec<f32>,    // [m_max, attn_dim]
    k_buf: Vec<f32>,         // [m_max, kv_dim]
    v_buf: Vec<f32>,         // [m_max, kv_dim]
    gate_buf: Vec<f32>,      // [m_max, ffn_dim]
    up_buf: Vec<f32>,        // [m_max, ffn_dim]
    ffn_mid: Vec<f32>,       // [m_max, ffn_dim]
}

impl BatchScratch {
    fn new(m_max: usize, pred_hidden: usize, attn_dim: usize, kv_dim: usize, ffn_dim: usize) -> Self {
        Self {
            m_max,
            h: vec![0.0; m_max * pred_hidden],
            residual: vec![0.0; m_max * pred_hidden],
            q_buf: vec![0.0; m_max * pred_hidden],
            attn_q_buf: vec![0.0; m_max * attn_dim],
            k_buf: vec![0.0; m_max * kv_dim],
            v_buf: vec![0.0; m_max * kv_dim],
            gate_buf: vec![0.0; m_max * ffn_dim],
            up_buf: vec![0.0; m_max * ffn_dim],
            ffn_mid: vec![0.0; m_max * ffn_dim],
        }
    }
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
            // Load norm tensors and extract f32 weight views.
            let attn_norm_t = load_f32(&blk("attn_norm.weight"), &mut *file)?;
            let attn_norm_w = attn_norm_t.to_vec1::<f32>()?;
            let attn_norm = RmsNorm::new(attn_norm_t, norm_eps);

            let q_norm_t = load_f32(&blk("attn_q_norm.weight"), &mut *file)?;
            let attn_q_norm_w = q_norm_t.to_vec1::<f32>()?;
            let attn_q_norm = RmsNorm::new(q_norm_t, norm_eps);

            let k_norm_t = load_f32(&blk("attn_k_norm.weight"), &mut *file)?;
            let attn_k_norm_w = k_norm_t.to_vec1::<f32>()?;
            let attn_k_norm = RmsNorm::new(k_norm_t, norm_eps);

            let ffn_norm_t = load_f32(&blk("ffn_norm.weight"), &mut *file)?;
            let ffn_norm_w = ffn_norm_t.to_vec1::<f32>()?;
            let ffn_norm = RmsNorm::new(ffn_norm_t, norm_eps);

            layers.push(DecoderLayer {
                attn_norm,
                attn_norm_w,
                attn_q: load_q8(&blk("attn_q.weight"), &mut *file)?,
                attn_k: load_q8(&blk("attn_k.weight"), &mut *file)?,
                attn_v: load_q8(&blk("attn_v.weight"), &mut *file)?,
                attn_o: load_q8(&blk("attn_output.weight"), &mut *file)?,
                attn_q_norm,
                attn_k_norm,
                attn_q_norm_w,
                attn_k_norm_w,
                ffn_norm,
                ffn_norm_w,
                ffn_gate: load_q8(&blk("ffn_gate.weight"), &mut *file)?,
                ffn_up: load_q8(&blk("ffn_up.weight"), &mut *file)?,
                ffn_down: load_q8(&blk("ffn_down.weight"), &mut *file)?,
            });
        }

        // ── MTP projection (loaded directly as Q8_0 from GGUF) ───────────
        let mtp_proj_q8 = content.tensor_infos.contains_key("code_pred.mtp_proj.weight")
            .then(|| Q8Weights::from_gguf(content, file, "code_pred.mtp_proj.weight").ok())
            .flatten();
        let mtp_proj_b_f32 = if content.tensor_infos.contains_key("code_pred.mtp_proj.bias") {
            load_f32("code_pred.mtp_proj.bias", &mut *file)?.to_vec1()?
        } else {
            Vec::new()
        };

        // ── output norm (f32 weight for rms_norm_f32) ────────────────────
        let output_norm_eps_val = norm_eps;
        let output_norm_w_f32 = if content.tensor_infos.contains_key("code_pred.output_norm.weight") {
            Some(load_f32("code_pred.output_norm.weight", &mut *file)?.flatten_all()?.to_vec1()?)
        } else {
            None
        };

        // ── codec_embd tables (Q8 direct from GGUF, no transpose needed) ─
        let num_embd = num_acoustic.saturating_sub(1); // 14
        let mut codec_embd_q8 = Vec::with_capacity(num_embd);
        for g in 0..num_embd {
            let name = format!("code_pred.codec_embd.{g}.weight");
            if content.tensor_infos.contains_key(&name) {
                codec_embd_q8.push(Q8Weights::from_gguf(content, file, &name)?);
            } else {
                anyhow::bail!("missing {name} — expected {num_embd} embedding tables");
            }
        }

        // ── lm_heads (Q8 direct from GGUF) ───────────────────────────────
        let mut lm_heads_q8 = Vec::with_capacity(num_acoustic);
        for g in 0..num_acoustic {
            let name = format!("code_pred.lm_head.{g}.weight");
            if !content.tensor_infos.contains_key(&name) {
                anyhow::bail!("missing {name} — expected {num_acoustic} lm heads");
            }
            lm_heads_q8.push(Q8Weights::from_gguf(content, file, &name)?);
        }

        // ── flat RoPE cos/sin (f64 precision, no Tensor) ─────────────────
        let half = head_dim / 2;
        let inv_freq_f64: Vec<f64> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0_f64 / rope_theta.powf(i as f64 / head_dim as f64))
            .collect();
        let mut cos_f32 = vec![0.0f32; max_seq_len * head_dim];
        let mut sin_f32 = vec![0.0f32; max_seq_len * head_dim];
        for p in 0..max_seq_len {
            let base = p * head_dim;
            for d in 0..half {
                let angle = p as f64 * inv_freq_f64[d];
                let c = angle.cos() as f32;
                let s = angle.sin() as f32;
                cos_f32[base + d] = c;
                cos_f32[base + d + half] = c;
                sin_f32[base + d] = s;
                sin_f32[base + d + half] = s;
            }
        }

        // FFN intermediate dimension from first layer's ffn_gate (output rows).
        let ffn_dim = layers
            .first()
            .map(|l| l.ffn_gate.out_features())
            .unwrap_or(pred_hidden * 3);

        // Pre-allocate flat f32 KV caches (per-layer, empty initially).
        let k_cache_data = vec![Vec::new(); n_layers];
        let v_cache_data = vec![Vec::new(); n_layers];

        // Pre-allocated scratch buffers + Q8 workspace for fused path.
        let kv_dim = layers
            .first()
            .map(|l| l.attn_k.out_features())
            .unwrap_or(pred_hidden);
        let attn_dim = n_q_heads * head_dim;
        let pred_scratch = PredScratch::new(pred_hidden, attn_dim, ffn_dim, kv_dim);
        let q8_ws = Q8Workspace::new();

        // ── batched KV cache + scratch ──────────────────────────────────
        let batch_m_max = 128;
        let k_cache_batched = vec![vec![Vec::new(); batch_m_max]; n_layers];
        let v_cache_batched = vec![vec![Vec::new(); batch_m_max]; n_layers];
        let batch_scratch = BatchScratch::new(batch_m_max, pred_hidden, attn_dim, kv_dim, ffn_dim);

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
            codec_embd_q8,
            lm_heads_q8,
            mtp_proj_q8,
            mtp_proj_b_f32,
            output_norm_eps: output_norm_eps_val,
            cos_f32,
            sin_f32,
            output_norm_w_f32,
            k_cache_data,
            v_cache_data,
            q8_ws,
            pred_scratch,
            k_cache_batched,
            v_cache_batched,
            batch_scratch,
        })
    }

    // ── f32 hot-path helpers (zero Tensor round-trips) ─────────────────

    /// Project `[talker_hidden]` → `[pred_hidden]` via mtp_proj (Q8 gemv).
    fn project_f32(&mut self, x: &[f32]) -> Vec<f32> {
        let n_out = self.pred_hidden;
        let mut dst = vec![0.0f32; n_out];
        if let Some(ref proj_q8) = self.mtp_proj_q8 {
            proj_q8.gemv_into(x, &mut self.q8_ws, &mut dst);
        }
        // Add bias
        if self.mtp_proj_b_f32.len() == n_out {
            for (d, b) in dst.iter_mut().zip(self.mtp_proj_b_f32.iter()) {
                *d += b;
            }
        }
        dst
    }

    /// Apply lm_head `g` to `[pred_hidden]` → `[vocab_size]` logits (Q8 gemv).
    fn apply_lm_head_f32(&mut self, g: usize, h: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.vocab_size];
        self.lm_heads_q8[g].gemv_into(h, &mut self.q8_ws, &mut logits);
        logits
    }

    /// Look up token from `codec_embd_q8[g]` and project → `[pred_hidden]`.
    ///
    /// Uses Q8 row lookup + `gemv_into_quantized` to avoid f32 dequantize
    /// of the codec_embd row and the subsequent f32 matmul — one fused step.
    fn embed_codec_f32(&mut self, g: usize, token_id: u32) -> Vec<f32> {
        let row_blocks = self.codec_embd_q8[g].row_blocks(token_id as usize);
        // Inject pre-quantized input (skips f32→Q8 quantize step)
        self.q8_ws.set_quantized_input(row_blocks);
        let n_out = self.pred_hidden;
        let mut dst = vec![0.0f32; n_out];
        if let Some(ref proj_q8) = self.mtp_proj_q8 {
            proj_q8.gemv_into_quantized(&self.q8_ws, &mut dst);
        }
        // Add bias
        if self.mtp_proj_b_f32.len() == n_out {
            for (d, b) in dst.iter_mut().zip(self.mtp_proj_b_f32.iter()) {
                *d += b;
            }
        }
        dst
    }

    /// Fused f32 forward for one position — zero Tensor round-trips on the hot path.
    ///
    /// All intermediate operations use raw f32 slices — no Tensor dispatch
    /// for norm, RoPE, or attention. Only the I/O boundaries (input Tensor,
    /// Q8 gemv projections, output Tensor) create tensors.
    ///
    /// `pos`: absolute position in the sequence (0-based within this frame).
    /// `pred_input`: `[pred_hidden]` — already projected via mtp_proj.
    /// Returns `[pred_hidden]` — output-normed hidden state.
    fn forward_at_pos_fused(&mut self, pos: usize, pred_input: &[f32]) -> Vec<f32> {
        let n_qh = self.n_q_heads;
        let n_kvh = self.n_kv_heads;
        let hd = self.head_dim;
        let eps = 1e-6_f64;
        let pred_h = self.pred_hidden;
        let kv_dim = self.layers[0].attn_k.out_features();

        // Flat RoPE cos/sin at this position
        let cos = &self.cos_f32[pos * hd..(pos + 1) * hd];
        let sin = &self.sin_f32[pos * hd..(pos + 1) * hd];

        let scr = &mut self.pred_scratch;
        scr.h.copy_from_slice(pred_input);

        for i in 0..self.n_layers {
            let layer = &self.layers[i];

            // ── Residual save + attn norm (in-place) ─────────────────────
            scr.residual.copy_from_slice(&scr.h);
            rms_norm_f32_inplace(&mut scr.h, &layer.attn_norm_w, eps);

            // ── QKV gemv (quantize-once multi gemv) ───────────────────────
            Q8Weights::gemv_multi_into(
                &[&layer.attn_q, &layer.attn_k, &layer.attn_v],
                &mut [&mut scr.attn_q_buf, &mut scr.k_buf, &mut scr.v_buf],
                &scr.h,
                &mut self.q8_ws,
            );

            // ── Per-head QK-norm (parallel) ──────────────────────────────
            let q = per_head_rms_norm_f32_par(&scr.attn_q_buf, &layer.attn_q_norm_w, n_qh, hd, eps);
            let k = per_head_rms_norm_f32_par(&scr.k_buf, &layer.attn_k_norm_w, n_kvh, hd, eps);

            // ── RoPE (parallel) ──────────────────────────────────────────
            let q = rope_f32_par(&q, cos, sin, n_qh, hd);
            let k = rope_f32_par(&k, cos, sin, n_kvh, hd);

            // ── KV cache append ──────────────────────────────────────────
            self.k_cache_data[i].extend_from_slice(&k);
            self.v_cache_data[i].extend_from_slice(&scr.v_buf); // V not normed/ROPEd

            // ── GQA attention (parallel) ─────────────────────────────────
            let kv_len = self.k_cache_data[i].len() / kv_dim;
            let attn = attention_f32_par(
                &q, &self.k_cache_data[i], &self.v_cache_data[i],
                n_qh, n_kvh, kv_len, hd, kv_len,
            );

            // ── Output projection gemv into q_buf (reuse) ────────────────
            layer.attn_o.gemv_into(&attn, &mut self.q8_ws, &mut scr.q_buf);

            // ── Residual add (parallel) ──────────────────────────────────
            scr.h.par_iter_mut()
                .zip(scr.residual.par_iter())
                .zip(scr.q_buf.par_iter())
                .for_each(|((h_j, &r), &q)| {
                    *h_j = r + q;
                });

            // ── FFN residual save ────────────────────────────────────────
            scr.residual.copy_from_slice(&scr.h);

            // ── FFN norm (in-place) ──────────────────────────────────────
            rms_norm_f32_inplace(&mut scr.h, &layer.ffn_norm_w, eps);

            // ── FFN gate + up gemv (quantize-once multi gemv) ────────────
            Q8Weights::gemv_multi_into(
                &[&layer.ffn_gate, &layer.ffn_up],
                &mut [&mut scr.gate_buf, &mut scr.up_buf],
                &scr.h,
                &mut self.q8_ws,
            );

            // ── SiLU(gate) * up → scr.ffn_mid (parallel) ─────────────────
            let ffn_dim = scr.gate_buf.len();
            let fm = &mut scr.ffn_mid[..ffn_dim];
            let silu_gate = silu_f32_par(&scr.gate_buf);
            fm.par_iter_mut()
                .zip(silu_gate.par_iter())
                .zip(scr.up_buf.par_iter())
                .for_each(|((dst, &g_act), &u)| {
                    *dst = g_act * u;
                });

            // ── Down projection gemv into q_buf (reuse) ──────────────────
            layer.ffn_down.gemv_into(fm, &mut self.q8_ws, &mut scr.q_buf);

            // ── Residual add (parallel) ──────────────────────────────────
            scr.h.par_iter_mut()
                .zip(scr.residual.par_iter())
                .zip(scr.q_buf.par_iter())
                .for_each(|((h_j, &r), &q)| {
                    *h_j = r + q;
                });
        }

        // ── Final output norm ────────────────────────────────────────────
        match &self.output_norm_w_f32 {
            Some(w) => rms_norm_f32(&scr.h, w, self.output_norm_eps),
            None => scr.h.clone(),
        }
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
    /// * `talker_hidden` — last talker hidden state `[talker_hidden]` (f32, no Tensor)
    /// * `c0_embed` — talker's embedding of codebook 0 token `[talker_hidden]` (f32)
    /// * `temperature` — sampling temperature (0.0 = argmax)
    /// * `top_k` — optional top-k filter
    /// * `top_p` — optional top-p (nucleus) filter
    /// * `rng` — mutable RNG handle
    ///
    /// Returns a `CodeFrame` of `num_acoustic` code token IDs (c1..cN).
    pub fn predict_one_frame_sampled(
        &mut self,
        talker_hidden: &[f32],
        c0_embed: &[f32],
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        rng: &mut impl rand::Rng,
    ) -> CodeFrame {
        // Reset KV cache (fresh per frame) — just clear the Vecs, no alloc.
        for kc in &mut self.k_cache_data { kc.clear(); }
        for vc in &mut self.v_cache_data { vc.clear(); }

        // ── Prefill position 0 ───────────────────────────────────────────
        // Input: talker hidden state → project → forward (no lm_head here)
        let proj_0 = self.project_f32(talker_hidden);
        self.forward_at_pos_fused(0, &proj_0);

        // ── Prefill position 1 ───────────────────────────────────────────
        // Input: c0_embed → project → forward → lm_head[0] → sample c1
        let proj_1 = self.project_f32(c0_embed);
        let h1_v = self.forward_at_pos_fused(1, &proj_1);
        let logits_0 = self.apply_lm_head_f32(0, &h1_v);
        let (c1, _prob) =
            sampling::sample_token(&logits_0, temperature, top_k, top_p, rng, None, 1.0);
        let mut codes: Vec<u32> = vec![c1];

        // ── Decode positions 2..(num_acoustic+1) ─────────────────────────
        // For g = 1..(num_acoustic-1):
        //   position = g + 1
        //   embed codes[g-1] (the just-predicted token) via codec_embd[g-1]
        //   project → forward → lm_head[g] → sample → push
        for g in 1..self.num_acoustic {
            let prev_token = codes[g - 1];
            // Embed + project in one step (all f32, no Tensor)
            let proj = self.embed_codec_f32(g - 1, prev_token);
            let pos = g + 1; // positions 2..15
            let h_v = self.forward_at_pos_fused(pos, &proj);
            let logits = self.apply_lm_head_f32(g, &h_v);
            let (code, _prob) =
                sampling::sample_token(&logits, temperature, top_k, top_p, rng, None, 1.0);
            codes.push(code);
        }

        debug_assert_eq!(codes.len(), self.num_acoustic);
        codes
    }

    /// Predict a single audio code frame with argmax (fully deterministic).
    ///
    /// Convenience wrapper: calls `predict_one_frame_sampled` with `temperature=0.0`.
    pub fn predict_one_frame_argmax(
        &mut self,
        talker_hidden: &[f32],
        c0_embed: &[f32],
    ) -> CodeFrame {
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

    /// Return the talker hidden dimension.
    #[must_use]
    pub fn talker_hidden_size(&self) -> usize {
        self.talker_hidden
    }

    /// Embed a single acoustic codebook token (codebook g, for g in 1..num_acoustic)
    /// into a flat `[talker_hidden]` f32 vec (dequantized from Q8).
    pub fn embed_acoustic_code(&self, codebook_idx: usize, token_id: u32) -> anyhow::Result<Vec<f32>> {
        let row_blocks = self.codec_embd_q8[codebook_idx].row_blocks(token_id as usize);
        let n_blocks = row_blocks.len();
        let padded_len = n_blocks * QK8_0;
        let mut row = vec![0.0f32; padded_len];
        <BlockQ8_0 as GgmlType>::to_float(row_blocks, &mut row);
        row.truncate(self.talker_hidden);
        Ok(row)
    }

    /// Sum-embed a full code frame (c0..cN including codebook 0) into a single
    /// `[talker_hidden]` f32 vec. Codebook 0 is looked up via `talker`, the
    /// remaining acoustic codebooks via the predictor's own embedding tables.
    ///
    /// `codes` must have length `1 + self.num_acoustic`.
    pub fn embed_frame(&self, talker: &Talker, codes: &[u32]) -> anyhow::Result<Vec<f32>> {
        let hidden = self.talker_hidden;
        let n_acoustic = self.num_acoustic;
        anyhow::ensure!(
            codes.len() >= 1 + n_acoustic,
            "embed_frame: expected {} codes, got {}",
            1 + n_acoustic,
            codes.len(),
        );

        let mut sum = vec![0.0f32; hidden];

        // c0: talker's codec embedding table
        let c0_vec = talker.lookup_codec_row(codes[0])?;
        for (i, &v) in c0_vec.iter().enumerate() {
            sum[i] += v;
        }

        // c1..c15: predictor's codec_embd[g-1]
        for g in 0..n_acoustic {
            let cg = codes[1 + g];
            let cg_vec = self.embed_acoustic_code(g, cg)?;
            for (i, &v) in cg_vec.iter().enumerate() {
                sum[i] += v;
            }
        }

        Ok(sum)
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
