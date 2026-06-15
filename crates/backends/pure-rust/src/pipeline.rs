//! Pipeline orchestration: autoregressive talker → code predictor → codec decoder.

use std::path::Path;

use candle_core::{Device, IndexOp, Tensor};
use qwen_tts_codec::CodecDecoder;
use qwen_tts_runtime::SynthesisRequest;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::code_predictor::CodePredictor;
use crate::debug_dumper::DebugDumper;
use crate::prompt::{build_prompt, load_gguf_metadata, parse_prompt_metadata, PromptCache, PromptMetadata};
use crate::sampling;
use crate::talker::{precompute_cos_sin, KvCache, Talker};
use crate::timing::TimingRecorder;
use crate::tokenizer::HfTokenizer;

/// Loaded pipeline ready for synthesis.
pub struct Pipeline {
    talker: Talker,
    code_predictor: CodePredictor,
    codec_decoder: CodecDecoder,
    tokenizer: HfTokenizer,
    device: Device,
    /// Prompt-relevant metadata parsed from GGUF (specials, languages, speakers).
    prompt_metadata: PromptMetadata,
    /// Pre-computed special embeddings + prefix cache.
    prompt_cache: PromptCache,
    /// Optional dump directory for deterministic parity fixtures (C++ compatible
    /// binary format). Set before calling synthesize() to enable debug dumps.
    pub dump_dir: Option<String>,
}

impl Pipeline {
    /// Load all model weights, discover tokenizer alongside talker GGUF.
    ///
    /// `device` — the device (CPU or CUDA) to load weights onto and run on.
    pub fn new(talker_path: &Path, codec_path: &Path, device: &Device) -> anyhow::Result<Self> {
        // Load prompt metadata from GGUF (opens the file separately, header-only).
        let gguf_meta = load_gguf_metadata(talker_path)?;
        let prompt_metadata = parse_prompt_metadata(&gguf_meta)?;

        let (talker, code_predictor) = Talker::load_with_predictor(talker_path, device)?;

        // Pre-compute special embeddings (tts_bos, tts_eos, tts_pad, codec_pad, codec_bos).
        let prompt_cache = PromptCache::new(&talker, &prompt_metadata, device)?;

        let codec_decoder = CodecDecoder::load(codec_path)
            .map_err(|e| anyhow::anyhow!("failed to load codec decoder: {e}"))?;

        let tokenizer = discover_tokenizer(talker_path)
            .map_err(|e| anyhow::anyhow!("tokenizer required — place tokenizer.json next to the talker GGUF: {e}"))?;

        log::info!(
            "pipeline loaded: device={:?}, talker={}, code_predictor={}, tokenizer=found",
            device,
            talker_path.display(),
            codec_path.display(),
        );

        Ok(Self {
            talker,
            code_predictor,
            codec_decoder,
            tokenizer,
            device: device.clone(),
            prompt_metadata,
            prompt_cache,
            dump_dir: None,
        })
    }

    /// Synthesize speech: text → WAV samples via autoregressive code generation.
    ///
    /// `timing_recorder` is optional — when provided, per-stage durations are
    /// recorded for benchmarking. Only active when `pipeline-timing` feature is
    /// enabled; otherwise the parameter is accepted but ignored.
    pub fn synthesize(
        &mut self,
        request: &SynthesisRequest,
        mut timing_recorder: Option<&mut TimingRecorder>,
    ) -> anyhow::Result<Vec<i16>> {
        let raw_text = &request.text;
        if raw_text.trim().is_empty() {
            anyhow::bail!("text cannot be empty");
        }

        // --- 1. Build prompt embedding via PromptBuilder ---
        let t1 = std::time::Instant::now();

        let tokenizer = &self.tokenizer;

        let ref_spk_emb: Option<&[f32]> = None;  // Voice clone not yet supported
        let ref_codes: Option<&[i32]> = None;     // ICL not yet supported
        let ref_codes_t: usize = 0;

        let prompt = build_prompt(
            &self.talker,
            &self.device,
            &self.prompt_metadata,
            &mut self.prompt_cache,
            tokenizer,
            Some(&self.code_predictor),  // needed for ICL mode
            raw_text,
            &request.language,
            request.instruct.as_deref().unwrap_or(""),
            request.speaker.as_deref().unwrap_or(""),
            ref_spk_emb,
            request.ref_text.as_deref().unwrap_or(""),
            ref_codes,
            ref_codes_t,
        )?;

        if let Some(ref mut tr) = timing_recorder {
            tr.record("prompt_embed".into(), "load".into(), t1.elapsed().as_secs_f64(), 0);
        }

        // Debug dump: prompt embedding (step 0 prefill input).
        let dumper = DebugDumper::new(self.dump_dir.clone());
        dumper.dump_2d("prompt-embed", prompt.T_ctx, prompt.hidden, &prompt.input_embed);

        let hidden = prompt.hidden;
        let t_ctx = prompt.T_ctx;

        // --- 2. Sampling configuration ---
        let temperature = request.temperature.unwrap_or(1.0);
        let top_k = request.top_k.map(|v| v as usize);
        let top_p = request.top_p.map(|v| v as f32);
        let repetition_penalty = request.repetition_penalty.unwrap_or(1.0);
        // Subtalker (code predictor) defaults — separate from talker params.
        // When SynthesisRequest doesn't expose subtalker fields, fall back to
        // the talker values (which match the C++ convention when both are the
        // same). TODO: add explicit subtalker fields to SynthesisRequest.
        let seed = request.seed.map(|s| s as u64);
        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let num_frames = request.max_new_tokens.unwrap_or(1024) as usize;
        let do_sample = request.do_sample.unwrap_or(true);
        let codec_eos_id = self.prompt_metadata.codec_specials.eos_id;
        let talker_vocab = self.talker.config().vocab_size;

        // ── 3. KV cache + autoregressive frame generation ──────────────────
        let codec_codebooks: usize = 16;
        let mut all_codes: Vec<i32> = Vec::with_capacity(num_frames * codec_codebooks);
        // c0 token history for repetition penalty (HF style: each unique token
        // touched once per call). Mirrors C++ `talker_history` in pipeline-tts.cpp.
        let mut talker_history: Vec<u32> = Vec::with_capacity(num_frames);

        // 3a. Precompute RoPE for the full context length
        let max_seq = self.talker.config().max_seq_len;
        let (cos_full, sin_full) =
            precompute_cos_sin(self.talker.config(), max_seq, &self.device)?;

        // 3b. Convert prompt embed to tensor and prefill one-by-one
        let input_embed = Tensor::from_slice(&prompt.input_embed, (1, t_ctx, hidden), &self.device)?;

        let mut kv_cache = KvCache::new(self.talker.config().n_layers);
        let mut last_hidden: Option<Tensor> = None;
        for pos in 0..t_ctx {
            let _step_timer = timing_recorder.as_mut().map(|tr| {
                tr.start("talker_step".into(), "step".into(), pos)
            });
            let emb = input_embed.i((.., pos..pos + 1, ..))?;
            let hidden_state = self
                .talker
                .forward_step(&emb, &mut kv_cache, &cos_full, &sin_full)?;
            last_hidden = Some(hidden_state);
            drop(_step_timer);
        }
        // last_hidden: [1, 1, d_model] — output-normed hidden state at the last text position
        let mut last_hidden = last_hidden
            .ok_or_else(|| anyhow::anyhow!("empty sequence after prompt prefill"))?;

        // 3c. Decode loop: generate audio frames with full codec embedding sum + trailing overlay
        for frame_idx in 0..num_frames {
            // ── Step A: sample codebook 0 with suppress + rep_penalty ─────
            let _cb0_timer = timing_recorder.as_mut().map(|tr| {
                tr.start("cb0_sample".into(), "step".into(), frame_idx)
            });
            let cb0_logits_t = self.talker.predict_codebook0(&last_hidden)?;
            // Convert from [1, 1, V] candle tensor to flat f32 slice for
            // suppress + rep_penalty + sampling chain.
            let cb0_logits_flat: Vec<f32> = cb0_logits_t.flatten_all()?.to_vec1()?;
            let mut cb0_logits = cb0_logits_flat;
            // Suppress codec reserved range [vocab-1024, vocab) except codec_eos.
            let suppress_lo = talker_vocab.saturating_sub(1024);
            let suppress_hi = talker_vocab;
            sampling::apply_suppress(&mut cb0_logits, suppress_lo, suppress_hi, codec_eos_id);
            let cb0_token = if do_sample {
                let (tok, _prob) = sampling::sample_token(
                    &cb0_logits, temperature, top_k, top_p, &mut rng,
                    Some(&talker_history), repetition_penalty,
                );
                tok
            } else {
                // Argmax path: still apply suppress (but skip rep_penalty
                // because argmax doesn't need it — it picks the highest logit
                // regardless).
                sampling::argmax_idx(&cb0_logits)
            };
            drop(_cb0_timer);

            // EOS detection — stop generation when codec_eos is emitted.
            if cb0_token == codec_eos_id {
                log::info!("[Pipeline] EOS at step {}, stopping", frame_idx);
                break;
            }

            // Track c0 history for repetition penalty.
            talker_history.push(cb0_token);

            // Embed codebook 0 token for predictor prefill
            let cb0_emb = self.talker.embed_codebook0(cb0_token)?;

            // ── Step B: predict acoustic codebooks 1..N ──────────────────
            let _pred_timer = timing_recorder.as_mut().map(|tr| {
                tr.start("predictor_frame".into(), "frame".into(), frame_idx)
            });
            let frame = if do_sample {
                self
                    .code_predictor
                    .predict_one_frame_sampled(&last_hidden, &cb0_emb, temperature, top_k, top_p, &mut rng)?
            } else {
                self
                    .code_predictor
                    .predict_one_frame_argmax(&last_hidden, &cb0_emb)?
            };
            drop(_pred_timer);

            // ── Step C: build next-token embedding ────────────────────────
            let codes: Vec<u32> = std::iter::once(cb0_token).chain(frame.iter().copied()).collect();
            let codec_sum = self.code_predictor.embed_frame(&self.talker, &codes)?;

            // Trailing text overlay
            let overlay: &[f32] = if frame_idx < prompt.T_trailing {
                let off = frame_idx * hidden;
                &prompt.trailing_text_hidden[off..off + hidden]
            } else {
                &prompt.tts_pad_embed
            };

            let mut next_emb_vec = vec![0.0f32; hidden];
            for i in 0..hidden {
                next_emb_vec[i] = codec_sum[i] + overlay[i];
            }
            // Debug dump: next-token embedding at step 0 (matches C++
            // `next-emb-step0`).
            if frame_idx == 0 {
                dumper.dump_1d("next-emb-step0", &next_emb_vec);
            }
            let next_emb = Tensor::from_slice(&next_emb_vec, (1, 1, hidden), &self.device)?;

            // Store all codes (pad to 16 codebooks)
            all_codes.push(cb0_token as i32);
            for &token in &frame {
                all_codes.push(token as i32);
            }
            while all_codes.len() < (frame_idx + 1) * codec_codebooks {
                all_codes.push(0);
            }
            // Debug dump: first frame codes (matches C++ `codes-step0`).
            if frame_idx == 0 {
                let frame_start = 0;
                let frame_end = frame_start + codec_codebooks;
                let codes_i32: Vec<i32> = all_codes[frame_start..frame_end]
                    .iter()
                    .copied()
                    .collect();
                dumper.dump_i32_as_f32("codes-step0", &[codec_codebooks as i32], &codes_i32);
            }

            // ── Step D: talker forward step ───────────────────────────────
            last_hidden = self
                .talker
                .forward_step(&next_emb, &mut kv_cache, &cos_full, &sin_full)?;

            // Debug dump: hidden state after step 1 (matches C++
            // `talker-hidden-step1`). The first frame (idx 0) produces the
            // hidden state that becomes step 1's talker output.
            if frame_idx == 0 {
                let hidden_flat: Vec<f32> = last_hidden.flatten_all()?.to_vec1()?;
                dumper.dump_1d("talker-hidden-step1", &hidden_flat);
            }
        }

        // Trim to exact multiples
        let total_frames = all_codes.len() / codec_codebooks;
        all_codes.truncate(total_frames * codec_codebooks);

        // --- 4. Decode audio via codec ---
        let _decode_timer = timing_recorder.as_mut().map(|tr| {
            tr.start("codec_decode".into(), "decode".into(), 0)
        });
        let audio_f32 = self.codec_decoder.decode(&all_codes, total_frames);
        drop(_decode_timer);

        // --- 5. Convert f32 → i16 PCM ---
        let audio_i16: Vec<i16> = audio_f32
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();

        // Debug dump: full codes matrix (matches C++ `codes-full`).
        dumper.dump_i32_as_f32(
            "codes-full",
            &[total_frames as i32, codec_codebooks as i32],
            &all_codes,
        );

        log::info!(
            "pure-rust synth: {} ctx tokens → {} code frames ({} code tokens) → {} audio samples ({}s)",
            t_ctx,
            total_frames,
            all_codes.len(),
            audio_i16.len(),
            audio_i16.len() / 24000,
        );

        Ok(audio_i16)
    }

    /// Convenience wrapper without timing recorder.
    pub fn synthesize_simple(&mut self, request: &SynthesisRequest) -> anyhow::Result<Vec<i16>> {
        self.synthesize(request, None)
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Look for `tokenizer.json` in the same directory as the talker GGUF.
fn discover_tokenizer(talker_path: &Path) -> anyhow::Result<HfTokenizer> {
    let parent = talker_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("talker path has no parent"))?;

    for name in &["tokenizer.json", "tokenizer.json"] {
        let candidate = parent.join(name);
        if candidate.exists() {
            return HfTokenizer::from_file(&candidate);
        }
    }

    anyhow::bail!(
        "no tokenizer.json found next to {}",
        talker_path.display()
    );
}
