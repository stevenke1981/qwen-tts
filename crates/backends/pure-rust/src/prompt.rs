// Allow non-snake-case names that match the C++ reference (T_ctx, N_text, etc.).
#![allow(non_snake_case)]

//! Prompt builder: assemble the talker prefix input embedding from a tokenized
//! text plus optional language / instruct / speaker / reference streams.
//!
//! Mirrors `vendor/qwentts.cpp/src/prompt-builder.h` bit-exact.
//!
//! Two streams (text and codec) are aligned and summed element-wise:
//!   text  stream: text_projection(text_embedding(token ids))
//!   codec stream: codec_embedding(lookup ids)
//!
//! Layout (standard, no instruct, no speaker):
//!   role         text(ids[0:3])                                3 vecs
//!   prefill      tts_pad x n_pad_pre + tts_bos                 5 vecs
//!                + codec_emb([think, think_bos, lang, think_eos, pad])
//!   trailing     text(ids[3:-5]) + tts_eos                     N_text + 1 vecs
//!                + codec_emb([pad x (N_text + 1)])
//!   final        tts_pad + codec_emb([bos])                    1 vec

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::Device;

use crate::code_predictor::CodePredictor;
use crate::talker::Talker;
use crate::tokenizer::HfTokenizer;

/// Open a GGUF file, parse only its metadata, return the metadata map.
/// Useful for loading prompt metadata before the full talker load.
pub fn load_gguf_metadata(path: &Path) -> anyhow::Result<HashMap<String, gguf_file::Value>> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("failed to open GGUF {path:?}: {e}"))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("bad GGUF header {path:?}: {e}"))?;
    Ok(content.metadata)
}

// ═════════════════════════════════════════════════════════════════════════════
// Types
// ═════════════════════════════════════════════════════════════════════════════

/// Codec special token IDs (from `qwen3-tts.codec.*` GGUF keys).
#[derive(Debug, Clone)]
pub struct CodecSpecials {
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub think_id: u32,
    pub nothink_id: u32,
    pub think_bos_id: u32,
    pub think_eos_id: u32,
}

/// Text special token IDs (from `qwen3-tts.text.*` GGUF keys).
#[derive(Debug, Clone)]
pub struct TextSpecials {
    pub im_start_id: u32,
    pub im_end_id: u32,
    pub tts_pad_id: u32,
    pub tts_bos_id: u32,
    pub tts_eos_id: u32,
}

/// Language entry from the GGUF language table.
#[derive(Debug, Clone)]
pub struct LanguageEntry {
    pub name: String,
    pub id: u32,
}

/// Speaker entry from the GGUF speaker table.
#[derive(Debug, Clone)]
pub struct SpeakerEntry {
    pub name: String,
    pub id: u32,
    pub dialect: String,
}

/// All prompt-relevant metadata parsed from the talker GGUF.
#[derive(Debug, Clone)]
pub struct PromptMetadata {
    pub codec_specials: CodecSpecials,
    pub text_specials: TextSpecials,
    pub languages: Vec<LanguageEntry>,
    pub speakers: Vec<SpeakerEntry>,
    pub num_code_groups: usize,
}

/// Pre-computed special embeddings + prefix cache.
pub struct PromptCache {
    pub tts_bos_emb: Vec<f32>,
    pub tts_eos_emb: Vec<f32>,
    pub tts_pad_emb: Vec<f32>,
    pub codec_pad_emb: Vec<f32>,
    pub codec_bos_emb: Vec<f32>,
    /// Prefix cache: key -> flat prefix embed rows [rows * hidden].
    prefix_cache: HashMap<String, Vec<f32>>,
    #[allow(dead_code)]
    max_prefix_entries: usize,
}

/// Output of the prompt builder.
pub struct PromptBuilderOutput {
    /// Full talker input embedding [T_ctx, hidden] f32 row-major.
    pub input_embed: Vec<f32>,
    pub T_ctx: usize,
    pub hidden: usize,
    /// Trailing text overlay [T_trailing, hidden].
    pub trailing_text_hidden: Vec<f32>,
    pub T_trailing: usize,
    /// tts_pad_embed [hidden]; fallback once trailing runs out.
    pub tts_pad_embed: Vec<f32>,
    /// Token IDs from the tokenizer (for debug parity).
    pub prompt_ids: Vec<u32>,
    /// Inner utterance text token count (ids[3:-5]).
    pub N_text: usize,
}

// ═════════════════════════════════════════════════════════════════════════════
// Metadata parsing
// ═════════════════════════════════════════════════════════════════════════════

fn get_u32(meta: &HashMap<String, gguf_file::Value>, key: &str) -> anyhow::Result<u32> {
    meta.get(key)
        .and_then(|v| v.to_u32().ok())
        .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata key: {key}"))
}

fn get_u32_opt(meta: &HashMap<String, gguf_file::Value>, key: &str) -> Option<u32> {
    meta.get(key).and_then(|v| v.to_u32().ok())
}

/// Parse codec specials from GGUF metadata.
pub fn parse_codec_specials(meta: &HashMap<String, gguf_file::Value>) -> anyhow::Result<CodecSpecials> {
    Ok(CodecSpecials {
        pad_id:       get_u32(meta, "qwen3-tts.codec.pad_id")?,
        bos_id:       get_u32(meta, "qwen3-tts.codec.bos_id")?,
        eos_id:       get_u32(meta, "qwen3-tts.codec.eos_id")?,
        think_id:     get_u32(meta, "qwen3-tts.codec.think_id")?,
        nothink_id:   get_u32(meta, "qwen3-tts.codec.nothink_id")?,
        think_bos_id: get_u32(meta, "qwen3-tts.codec.think_bos_id")?,
        think_eos_id: get_u32(meta, "qwen3-tts.codec.think_eos_id")?,
    })
}

/// Parse text specials from GGUF metadata.
pub fn parse_text_specials(meta: &HashMap<String, gguf_file::Value>) -> anyhow::Result<TextSpecials> {
    Ok(TextSpecials {
        im_start_id: get_u32(meta, "qwen3-tts.text.im_start_id")?,
        im_end_id:   get_u32(meta, "qwen3-tts.text.im_end_id")?,
        tts_pad_id:  get_u32(meta, "qwen3-tts.text.tts_pad_id")?,
        tts_bos_id:  get_u32(meta, "qwen3-tts.text.tts_bos_id")?,
        tts_eos_id:  get_u32(meta, "qwen3-tts.text.tts_eos_id")?,
    })
}

/// Parse language entries from GGUF metadata arrays.
fn parse_languages(meta: &HashMap<String, gguf_file::Value>) -> Vec<LanguageEntry> {
    let names = meta
        .get("qwen3-tts.codec.language_names")
        .and_then(|v| v.to_vec().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.to_string().ok().cloned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let ids = meta
        .get("qwen3-tts.codec.language_ids")
        .and_then(|v| v.to_vec().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.to_u32().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    names
        .into_iter()
        .zip(ids.into_iter())
        .map(|(name, id)| LanguageEntry { name, id })
        .collect()
}

/// Parse speaker entries from GGUF metadata arrays.
fn parse_speakers(meta: &HashMap<String, gguf_file::Value>) -> Vec<SpeakerEntry> {
    let names = meta
        .get("qwen3-tts.codec.speaker_names")
        .and_then(|v| v.to_vec().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.to_string().ok().cloned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let ids = meta
        .get("qwen3-tts.codec.speaker_ids")
        .and_then(|v| v.to_vec().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.to_u32().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let dialects = meta
        .get("qwen3-tts.codec.speaker_dialects")
        .and_then(|v| v.to_vec().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.to_string().ok().cloned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    names
        .into_iter()
        .zip(ids.into_iter())
        .zip(dialects.into_iter())
        .map(|((name, id), dialect)| SpeakerEntry { name, id, dialect })
        .collect()
}

/// Parse all prompt-relevant metadata from the talker GGUF metadata map.
pub fn parse_prompt_metadata(meta: &HashMap<String, gguf_file::Value>) -> anyhow::Result<PromptMetadata> {
    let codec_specials = parse_codec_specials(meta)?;
    let text_specials = parse_text_specials(meta)?;
    let languages = parse_languages(meta);
    let speakers = parse_speakers(meta);
    let num_code_groups = get_u32_opt(meta, "qwen3-tts.num_code_groups").unwrap_or(16) as usize;
    Ok(PromptMetadata { codec_specials, text_specials, languages, speakers, num_code_groups })
}

// ═════════════════════════════════════════════════════════════════════════════
// PromptCache
// ═════════════════════════════════════════════════════════════════════════════

impl PromptCache {
    /// Pre-compute special embeddings at load time.
    pub fn new(talker: &Talker, metadata: &PromptMetadata, device: &Device) -> anyhow::Result<Self> {
        let hidden = talker.config().d_model;
        let cs = &metadata.codec_specials;
        let ts = &metadata.text_specials;

        // Project special text tokens in one batch: [tts_bos, tts_eos, tts_pad]
        let special_ids = [ts.tts_bos_id, ts.tts_eos_id, ts.tts_pad_id];
        let special_proj = project_text_ids(talker, device, &special_ids)?;
        let tts_bos_emb = special_proj[..hidden].to_vec();
        let tts_eos_emb = special_proj[hidden..2 * hidden].to_vec();
        let tts_pad_emb = special_proj[2 * hidden..3 * hidden].to_vec();

        // Codec special embeddings: direct lookup into codec_embd
        let codec_pad_emb = talker.lookup_codec_row(cs.pad_id)?;
        let codec_bos_emb = talker.lookup_codec_row(cs.bos_id)?;

        Ok(Self {
            tts_bos_emb,
            tts_eos_emb,
            tts_pad_emb,
            codec_pad_emb,
            codec_bos_emb,
            prefix_cache: HashMap::new(),
            max_prefix_entries: 16,
        })
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Element-wise add: a[i] += b[i] for each i.
fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
        *ai += bi;
    }
}

/// Project text token IDs through text_embedding + text_proj -> flat [N * hidden] f32.
fn project_text_ids(talker: &Talker, _device: &Device, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    talker.embed_text_to_vec(ids)
}

/// Embed a single codec row (including sentinel handling for ref_spk_emb).
fn embed_codec_row(
    talker: &Talker,
    row_id: i32,
    ref_spk_emb: Option<&[f32]>,
    hidden: usize,
) -> anyhow::Result<Vec<f32>> {
    if row_id == -2 {
        let emb = ref_spk_emb
            .ok_or_else(|| anyhow::anyhow!("ref_spk_emb sentinel but no speaker embedding"))?;
        anyhow::ensure!(emb.len() == hidden, "ref_spk_emb len mismatch");
        return Ok(emb.to_vec());
    }
    talker.lookup_codec_row(row_id as u32)
}

/// Sum-embed reference codes across all code groups for ICL mode.
fn sum_ref_codes(
    talker: &Talker,
    predictor: &CodePredictor,
    ref_codes: &[i32],
    ref_codes_T: usize,
    groups: usize,
    hidden: usize,
) -> anyhow::Result<Vec<f32>> {
    let mut out = vec![0.0f32; ref_codes_T * hidden];
    for t in 0..ref_codes_T {
        let mut frame_codes = Vec::with_capacity(groups);
        for g in 0..groups {
            frame_codes.push(ref_codes[g * ref_codes_T + t] as u32);
        }
        let emb = predictor.embed_frame(talker, &frame_codes)?;
        let offset = t * hidden;
        out[offset..offset + hidden].copy_from_slice(&emb);
    }
    Ok(out)
}

/// Build the prefix cache key from instruct_ids, role_ids, and codec_left.
fn prefix_cache_key(instruct_ids: &[u32], role_ids: &[u32], codec_left: &[i32]) -> String {
    let mut key = String::new();
    key.push_str("instruct=");
    for id in instruct_ids {
        key.push_str(&id.to_string());
        key.push(',');
    }
    key.push_str(";role=");
    for id in role_ids {
        key.push_str(&id.to_string());
        key.push(',');
    }
    key.push_str(";codec=");
    for id in codec_left {
        key.push_str(&id.to_string());
        key.push(',');
    }
    key.push(';');
    key
}

// ═════════════════════════════════════════════════════════════════════════════
// Public API: build_prompt
// ═════════════════════════════════════════════════════════════════════════════

/// Build the full prompt embedding for the talker prefill.
///
/// Mirrors `prompt_builder_build()` in vendor/qwentts.cpp/src/prompt-builder.h.
///
/// # Arguments
/// * `talker` -- loaded talker weights (for text_proj and codec_embd lookups)
/// * `device` -- compute device
/// * `metadata` -- parsed GGUF prompt metadata
/// * `cache` -- pre-computed special embeddings + mutable prefix cache
/// * `tokenizer` -- BPE tokenizer
/// * `code_predictor` -- needed for ICL mode only (reference code projection)
/// * `utterance_text` -- the text to synthesize
/// * `language` -- language tag ("auto" for auto-detect)
/// * `instruct_text` -- style description (empty = none)
/// * `speaker_name` -- speaker name (empty = none)
/// * `ref_spk_emb` -- raw speaker embedding for voice clone mode
/// * `ref_text` -- reference text for ICL mode (empty = no ICL)
/// * `ref_codes` -- reference codes for ICL mode (flat [groups, T] i32 row-major)
/// * `ref_codes_T` -- number of frames in ref_codes
#[allow(clippy::too_many_arguments)]
pub fn build_prompt(
    talker: &Talker,
    device: &Device,
    metadata: &PromptMetadata,
    cache: &mut PromptCache,
    tokenizer: &HfTokenizer,
    code_predictor: Option<&CodePredictor>,
    utterance_text: &str,
    language: &str,
    instruct_text: &str,
    speaker_name: &str,
    ref_spk_emb: Option<&[f32]>,
    ref_text: &str,
    ref_codes: Option<&[i32]>,
    ref_codes_T: usize,
) -> anyhow::Result<PromptBuilderOutput> {
    let hidden = talker.config().d_model;
    let cs = &metadata.codec_specials;
    let _ts = &metadata.text_specials;
    let pc: &mut PromptCache = cache;

    // ── 1. Tokenize utterance ──────────────────────────────────────────────
    let full_text = format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        utterance_text
    );
    let ids = tokenizer.encode(&full_text)?;
    let N = ids.len();
    anyhow::ensure!(N >= 8, "tokenized prompt too short: {} tokens", N);
    let N_text = N - 8;
    anyhow::ensure!(N_text > 0, "no utterance text in prompt (N={})", N);

    // ── 2. Resolve language ────────────────────────────────────────────────
    let language_id: i32 = if language.eq_ignore_ascii_case("auto") {
        -1
    } else {
        let lang_lc = language.to_lowercase();
        let found = metadata.languages.iter().find(|e| e.name == lang_lc);
        anyhow::ensure!(found.is_some(), "unknown language '{}'", language);
        found.unwrap().id as i32
    };

    // ── 3. Resolve speaker + dialect override ─────────────────────────────
    let mut language_id = language_id; // may be overridden by dialect
    if !speaker_name.is_empty() {
        let spk_lc = speaker_name.to_lowercase();
        let found = metadata.speakers.iter().find(|e| e.name == spk_lc);
        anyhow::ensure!(found.is_some(), "unknown speaker '{}'", speaker_name);
        let entry = found.unwrap();
        let _speaker_id = entry.id as i32;

        // Dialect override
        if !entry.dialect.is_empty() {
            let lang_lc = language.to_lowercase();
            if lang_lc == "chinese" || lang_lc == "auto" {
                if let Some(dialect) = metadata.languages.iter().find(|le| le.name == entry.dialect) {
                    language_id = dialect.id as i32;
                }
            }
        }
    }

    anyhow::ensure!(
        pc.tts_pad_emb.len() == hidden,
        "prompt cache not initialized"
    );

    // ── 4. Build codec prefill list ────────────────────────────────────────
    let mut codec_prefill: Vec<i32> = if language_id < 0 {
        vec![
            cs.nothink_id as i32,
            cs.think_bos_id as i32,
            cs.think_eos_id as i32,
        ]
    } else {
        vec![
            cs.think_id as i32,
            cs.think_bos_id as i32,
            language_id,
            cs.think_eos_id as i32,
        ]
    };

    if !speaker_name.is_empty() {
        let spk_lc = speaker_name.to_lowercase();
        if let Some(entry) = metadata.speakers.iter().find(|e| e.name == spk_lc) {
            codec_prefill.push(entry.id as i32);
        }
    } else if ref_spk_emb.is_some() {
        codec_prefill.push(-2); // sentinel: copy ref_spk_emb
    }

    let _n_prefill = codec_prefill.len();
    let n_pad_pre = _n_prefill;
    let mut codec_left = codec_prefill.clone();
    codec_left.push(cs.pad_id as i32);

    // ── 5. Tokenize instruct ───────────────────────────────────────────────
    let instruct_ids: Vec<u32> = if !instruct_text.is_empty() {
        let wrapped = format!("<|im_start|>user\n{}<|im_end|>\n", instruct_text);
        tokenizer.encode(&wrapped)?
    } else {
        vec![]
    };
    let N_instruct = instruct_ids.len();

    // ── 6. Tokenize reference (ICL) ────────────────────────────────────────
    let icl = !ref_text.is_empty() && ref_codes.is_some() && ref_codes_T > 0;
    if icl && ref_spk_emb.is_none() {
        anyhow::bail!("ICL mode requires ref_spk_emb");
    }

    let ref_ids: Vec<u32>;
    let N_ref_text: usize;
    if icl {
        let ref_full = format!(
            "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
            ref_text
        );
        ref_ids = tokenizer.encode(&ref_full)?;
        anyhow::ensure!(ref_ids.len() >= 8, "ref_text too short: {}", ref_ids.len());
        N_ref_text = ref_ids.len() - 8;
        anyhow::ensure!(N_ref_text > 0, "empty ref_text body");
    } else {
        ref_ids = vec![];
        N_ref_text = 0;
    }

    // ── 7. ICL geometry ────────────────────────────────────────────────────
    let groups = metadata.num_code_groups;
    let _text_lens_icl = if icl { N_ref_text + N_text + 1 } else { 0 };
    let codec_lens_icl = if icl { 1 + ref_codes_T } else { 0 };
    let icl_T = if icl { codec_lens_icl } else { 0 };
    let text_lens_icl = _text_lens_icl;

    // ── 8. Calculate T_ctx ─────────────────────────────────────────────────
    let T_ctx = if icl {
        N_instruct + 3 + n_pad_pre + 1 + icl_T
    } else {
        N_instruct + 3 + n_pad_pre + 1 + N_text + 1 + 1
    };

    let mut out = PromptBuilderOutput {
        input_embed: vec![0.0f32; T_ctx * hidden],
        T_ctx,
        hidden,
        trailing_text_hidden: vec![],
        T_trailing: 0,
        tts_pad_embed: pc.tts_pad_emb.clone(),
        prompt_ids: ids.clone(),
        N_text,
    };

    let row_offset = |r: usize| -> usize { r * hidden };

    // ── 9. Build prefix (instruct + role + prefill) ────────────────────────
    let prefix_rows = N_instruct + 3 + codec_left.len();
    let cacheable_prefix = !icl && ref_spk_emb.is_none();
    let mut row: usize = 0;

    if cacheable_prefix {
        let key = prefix_cache_key(&instruct_ids, &ids[..3], &codec_left);
        if let Some(cached) = pc.prefix_cache.get(&key) {
            out.input_embed[..prefix_rows * hidden].copy_from_slice(cached);
            row = prefix_rows;
        } else {
            // Cache miss: compute prefix
            if N_instruct + 3 > 0 {
                let mut head_ids = Vec::with_capacity(N_instruct + 3);
                head_ids.extend_from_slice(&instruct_ids);
                head_ids.extend_from_slice(&ids[..3]);
                let proj = project_text_ids(talker, device, &head_ids)?;
                out.input_embed[..(N_instruct + 3) * hidden].copy_from_slice(&proj);
                row = N_instruct + 3;
            }

            for (i, &ccl) in codec_left.iter().enumerate() {
                let off = row_offset(row + i);
                let text_vec = if i == codec_left.len() - 1 {
                    &pc.tts_bos_emb
                } else {
                    &pc.tts_pad_emb
                };
                out.input_embed[off..off + hidden].copy_from_slice(text_vec);
                let ce = embed_codec_row(talker, ccl, ref_spk_emb, hidden)?;
                vec_add_inplace(&mut out.input_embed[off..off + hidden], &ce);
            }
            row += codec_left.len();

            if row == prefix_rows {
                pc.prefix_cache
                    .insert(key, out.input_embed[..prefix_rows * hidden].to_vec());
            }
        }
    } else {
        // Not cacheable: compute without caching
        if N_instruct + 3 > 0 {
            let mut head_ids = Vec::with_capacity(N_instruct + 3);
            head_ids.extend_from_slice(&instruct_ids);
            head_ids.extend_from_slice(&ids[..3]);
            let proj = project_text_ids(talker, device, &head_ids)?;
            out.input_embed[..(N_instruct + 3) * hidden].copy_from_slice(&proj);
            row = N_instruct + 3;
        }

        for (i, &ccl) in codec_left.iter().enumerate() {
            let off = row_offset(row + i);
            let text_vec = if i == codec_left.len() - 1 {
                &pc.tts_bos_emb
            } else {
                &pc.tts_pad_emb
            };
            out.input_embed[off..off + hidden].copy_from_slice(text_vec);
            let ce = embed_codec_row(talker, ccl, ref_spk_emb, hidden)?;
            vec_add_inplace(&mut out.input_embed[off..off + hidden], &ce);
        }
        row += codec_left.len();
    }

    // ── 10. Trailing (non-ICL) or ICL block ────────────────────────────────
    if !icl {
        let utter_body = &ids[3..3 + N_text];
        let proj = project_text_ids(talker, device, utter_body)?;
        for i in 0..N_text {
            let off = row_offset(row + i);
            let src_start = i * hidden;
            out.input_embed[off..off + hidden]
                .copy_from_slice(&proj[src_start..src_start + hidden]);
            vec_add_inplace(&mut out.input_embed[off..off + hidden], &pc.codec_pad_emb);
        }
        row += N_text;

        // tts_eos + codec_pad
        {
            let off = row_offset(row);
            out.input_embed[off..off + hidden].copy_from_slice(&pc.tts_eos_emb);
            vec_add_inplace(&mut out.input_embed[off..off + hidden], &pc.codec_pad_emb);
            row += 1;
        }

        // final: tts_pad + codec_bos
        {
            let off = row_offset(row);
            out.input_embed[off..off + hidden].copy_from_slice(&pc.tts_pad_emb);
            vec_add_inplace(&mut out.input_embed[off..off + hidden], &pc.codec_bos_emb);
            row += 1;
        }
    } else {
        // ICL mode
        let pred = code_predictor
            .ok_or_else(|| anyhow::anyhow!("ICL mode requires code_predictor"))?;
        let T_icl = icl_T;

        // Codec stream [T_icl, hidden]
        let mut codec_stream = vec![0.0f32; T_icl * hidden];
        codec_stream[..hidden].copy_from_slice(&pc.codec_bos_emb);
        let ref_codes_data = ref_codes
            .ok_or_else(|| anyhow::anyhow!("ICL mode requires ref_codes"))?;
        let ref_sum =
            sum_ref_codes(talker, pred, ref_codes_data, ref_codes_T, groups, hidden)?;
        codec_stream[hidden..(1 + ref_codes_T) * hidden].copy_from_slice(&ref_sum);

        // Text stream [text_lens_icl, hidden]
        let mut text_ids = Vec::with_capacity(N_ref_text + N_text);
        if N_ref_text > 0 {
            text_ids.extend_from_slice(&ref_ids[3..3 + N_ref_text]);
        }
        text_ids.extend_from_slice(&ids[3..3 + N_text]);
        let mut text_stream = project_text_ids(talker, device, &text_ids)?;

        // Append tts_eos
        {
            let eos_start = (text_lens_icl - 1) * hidden;
            if text_stream.len() < eos_start + hidden {
                text_stream.resize(eos_start + hidden, 0.0);
            }
            text_stream[eos_start..eos_start + hidden].copy_from_slice(&pc.tts_eos_emb);
        }

        // Align text stream to T_icl
        let mut aligned_text = vec![0.0f32; T_icl * hidden];
        if text_lens_icl >= T_icl {
            aligned_text.copy_from_slice(&text_stream[..T_icl * hidden]);
            let trailing_n = text_lens_icl - T_icl;
            out.T_trailing = if trailing_n > 0 { trailing_n } else { 1 };
            out.trailing_text_hidden = vec![0.0f32; out.T_trailing * hidden];
            if trailing_n > 0 {
                out.trailing_text_hidden
                    .copy_from_slice(&text_stream[T_icl * hidden..]);
            } else {
                out.trailing_text_hidden[..hidden].copy_from_slice(&pc.tts_pad_emb);
            }
        } else {
            if text_lens_icl > 0 {
                aligned_text[..text_lens_icl * hidden]
                    .copy_from_slice(&text_stream[..text_lens_icl * hidden]);
            }
            for i in text_lens_icl..T_icl {
                let off = i * hidden;
                aligned_text[off..off + hidden].copy_from_slice(&pc.tts_pad_emb);
            }
            out.T_trailing = 1;
            out.trailing_text_hidden = vec![0.0f32; hidden];
            out.trailing_text_hidden[..hidden].copy_from_slice(&pc.tts_pad_emb);
        }

        // Sum aligned_text + codec_stream
        for i in 0..T_icl {
            let off = row_offset(row + i);
            let t_off = i * hidden;
            for j in 0..hidden {
                out.input_embed[off + j] =
                    aligned_text[t_off + j] + codec_stream[t_off + j];
            }
        }
        row += T_icl;
    }

    // ── 11. Validate layout ────────────────────────────────────────────────
    anyhow::ensure!(row == T_ctx, "layout error: row={} != T_ctx={}", row, T_ctx);

    // ── 12. Trailing text hidden for non-ICL mode ──────────────────────────
    if !icl {
        out.T_trailing = 1;
        out.trailing_text_hidden = vec![0.0f32; hidden];
        out.trailing_text_hidden.copy_from_slice(&pc.tts_pad_emb);
    }

    log::info!(
        "[Prompt] Built: {} ids, N_text={}, N_instruct={}, T_ctx={}, hidden={}, lang_id={}, speaker={}, icl={}",
        N, N_text, N_instruct, T_ctx, hidden, language_id,
        if speaker_name.is_empty() { "none".to_string() } else { speaker_name.to_string() },
        icl,
    );

    Ok(out)
}
