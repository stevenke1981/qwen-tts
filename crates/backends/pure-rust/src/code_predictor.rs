//! Audio code token predictor for Qwen3-TTS.
//!
//! The code predictor is a full transformer (typically 5 layers) that takes
//! the talker's final hidden state and iteratively predicts acoustic codebook
//! tokens for levels 1..N (codebook 0 is predicted by the talker itself).
//!
//! For each audio frame, the predictor runs one step per codebook level:
//!   embed(prev_code_token[g]) → transformer_layers → lm_head[g] → logits
//!
//! Tensor naming (from the talker GGUF):
//!   code_pred.output_norm.weight
//!   code_pred.mtp_proj.weight / .bias         (optional → talker→pred hidden)
//!   code_pred.codec_embd.{g}.weight            (g = 0..num_acoustic-1)
//!   code_pred.lm_head.{g}.weight
//!   code_pred.blk.{i}.attn_norm.weight
//!   code_pred.blk.{i}.ffn_norm.weight
//!   code_pred.blk.{i}.attn_q.weight
//!   code_pred.blk.{i}.attn_k.weight
//!   code_pred.blk.{i}.attn_v.weight
//!   code_pred.blk.{i}.attn_output.weight
//!   code_pred.blk.{i}.attn_q_norm.weight       (QK-norm)
//!   code_pred.blk.{i}.attn_k_norm.weight       (QK-norm)
//!   code_pred.blk.{i}.ffn_gate.weight           (SwiGLU)
//!   code_pred.blk.{i}.ffn_up.weight
//!   code_pred.blk.{i}.ffn_down.weight

use std::fs::File;

use candle_core::quantized::gguf_file::Content;
use candle_core::{Device, IndexOp, Module, Tensor, D};
use candle_nn::RmsNorm;

use crate::config::ModelConfig;
use crate::sampling;

/// Type alias: a frame of acoustic code token IDs (one per codebook level).
pub type CodeFrame = Vec<u32>;

/// Code predictor: projects talker hidden states → per-codebook logits.
///
/// For the initial implementation we load only the lm_head per codebook and
/// a projection layer. Full multi-layer transformer will follow.
pub struct CodePredictor {
    /// Number of acoustic codebooks (= total_code_groups - 1).
    num_acoustic: usize,
    /// Optional projection: talker_hidden_size → predictor_hidden_size.
    mtp_proj_w: Option<Tensor>,
    mtp_proj_b: Option<Tensor>,
    /// Output norm after projection (before per-codebook heads).
    output_norm: Option<RmsNorm>,
    /// Linear heads: one per acoustic codebook, shape [vocab_size, hidden].
    lm_heads: Vec<Tensor>,
    /// Predictor hidden dimension.
    hidden_size: usize,
    #[allow(dead_code)]
    device: Device,
}

impl CodePredictor {
    /// Load code predictor weights from the **talker** GGUF content.
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
        // Read code predictor metadata (with fallback to talker config)
        let cfg = ModelConfig::from_gguf(&content.metadata);

        // Number of acoustic codebooks = total_code_groups - 1
        let num_code_groups = content
            .metadata
            .get("qwen3-tts.num_code_groups")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(16) as usize;
        let num_acoustic = num_code_groups.saturating_sub(1);
        if num_acoustic == 0 {
            anyhow::bail!("num_code_groups <= 1, no acoustic codebooks to predict");
        }

        // Code predictor hidden size (may differ from talker hidden size)
        let pred_hidden = content
            .metadata
            .get("qwen3-tts.code_pred.embedding_length")
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(cfg.d_model);

        // Helper: load tensor by name from GGUF, dequantized to F32
        let mut load = |name: &str| -> anyhow::Result<Tensor> {
            let qt = content
                .tensor(file, name, device)
                .map_err(|e| anyhow::anyhow!("missing code_pred tensor {name}: {e}"))?;
            qt.dequantize(device)
                .map_err(|e| anyhow::anyhow!("dequantize {name}: {e}"))
        };

        // Optional MTP projection (talker hidden → predictor hidden)
        let mtp_proj_w = if content.tensor_infos.contains_key("code_pred.mtp_proj.weight") {
            Some(load("code_pred.mtp_proj.weight")?)
        } else {
            None
        };
        let mtp_proj_b = if content.tensor_infos.contains_key("code_pred.mtp_proj.bias") {
            Some(load("code_pred.mtp_proj.bias")?)
        } else {
            None
        };

        // Output norm (optional — present in most checkpoints)
        let output_norm = if content.tensor_infos.contains_key("code_pred.output_norm.weight") {
            let w = load("code_pred.output_norm.weight")?;
            let eps = content
                .metadata
                .get("qwen3-tts.code_pred.attention.layer_norm_rms_epsilon")
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(1e-6) as f64;
            Some(RmsNorm::new(w, eps))
        } else {
            None
        };

        // Per-codebook linear heads
        let mut lm_heads = Vec::with_capacity(num_acoustic);
        for g in 0..num_acoustic {
            let name = format!("code_pred.lm_head.{g}.weight");
            if !content.tensor_infos.contains_key(&name) {
                anyhow::bail!("missing code_pred.lm_head.{g}.weight — expected {num_acoustic} heads");
            }
            let w = load(&name)?;
            lm_heads.push(w);
        }

        Ok(Self {
            num_acoustic,
            mtp_proj_w,
            mtp_proj_b,
            output_norm,
            lm_heads,
            hidden_size: pred_hidden,
            device: device.clone(),
        })
    }

    /// Predict a single audio code frame from the talker's final hidden state.
    ///
    /// `hidden`: talker output `[batch, d_model]`.
    ///
    /// Returns a per-batch frame of `num_acoustic` code tokens.
    ///
    /// Note: this is a simplified single-step predictor. A full
    /// implementation would iterate over codebooks with causal masking
    /// and include the code predictor transformer layers.
    pub fn predict_one_frame(&self, hidden: &Tensor) -> anyhow::Result<Vec<CodeFrame>> {
        let batch = hidden.dims()[0];

        // 1. Optional MTP projection: talker_hidden → predictor_hidden
        let mut h = hidden.clone();
        if let Some(ref w) = self.mtp_proj_w {
            h = w.matmul(&h.t()?)?.t()?;
            if let Some(ref b) = self.mtp_proj_b {
                h = h.broadcast_add(b)?;
            }
        }

        // 2. Output norm (optional)
        if let Some(ref norm) = self.output_norm {
            h = norm.forward(&h)?;
        }

        // 3. For each codebook level, apply lm_head
        let mut frames = vec![CodeFrame::with_capacity(self.num_acoustic); batch];
        for (_g, head_w) in self.lm_heads.iter().enumerate() {
            let logits = if head_w.dims()[0] == self.hidden_size {
                head_w.matmul(&h.t()?)?.t()?
            } else {
                h.matmul(&head_w.t()?)?
            };

            // Argmax per batch item
            for b in 0..batch {
                let row = logits.i(b)?;
                let idx = row.argmax(D::Minus1)?;
                let token: u32 = idx.flatten_all()?.to_vec0()?;
                frames[b].push(token);
            }
        }

        Ok(frames)
    }

    /// Predict a frame using temperature/top-k/top-p sampling instead of argmax.
    ///
    /// Returns one `CodeFrame` for `batch=1`.
    pub fn predict_one_frame_sampled(
        &self,
        hidden: &Tensor,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        rng: &mut impl rand::Rng,
    ) -> anyhow::Result<CodeFrame> {
        let batch = hidden.dims()[0];
        assert_eq!(batch, 1, "sampled prediction only supports batch=1");

        // 1. Optional MTP projection
        let mut h = hidden.clone();
        if let Some(ref w) = self.mtp_proj_w {
            h = w.matmul(&h.t()?)?.t()?;
            if let Some(ref b) = self.mtp_proj_b {
                h = h.broadcast_add(b)?;
            }
        }

        // 2. Output norm
        if let Some(ref norm) = self.output_norm {
            h = norm.forward(&h)?;
        }

        // 3. For each codebook level, apply lm_head + sample
        let mut frame = CodeFrame::with_capacity(self.num_acoustic);
        for (_g, head_w) in self.lm_heads.iter().enumerate() {
            // Get logits as Vec<f32>
            let logits_t = if head_w.dims()[0] == self.hidden_size {
                head_w.matmul(&h.t()?)?.t()?
            } else {
                h.matmul(&head_w.t()?)?
            };

            let logits_flat: Vec<f32> = logits_t.flatten_all()?.to_vec1()?;

            // Sample
            let (token, _prob) = sampling::sample_token(&logits_flat, temperature, top_k, top_p, rng);
            frame.push(token);
        }

        Ok(frame)
    }

    /// Return the number of acoustic codebooks this predictor handles.
    #[must_use]
    pub fn num_acoustic(&self) -> usize {
        self.num_acoustic
    }

    /// Return the predictor's hidden dimension.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_frame_type() {
        let frame: CodeFrame = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        assert_eq!(frame.len(), 15);
    }
}
