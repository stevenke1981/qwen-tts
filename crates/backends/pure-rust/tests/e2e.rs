//! End-to-end integration tests against real GGUF model files.
//!
//! These tests load the actual Qwen3-TTS GGUF models from the `models/`
//! directory and exercise the pure-Rust pipeline. The full synthesis test
//! is `#[ignore]` by default because it dequantizes ~2 GB of Q8_0 weights
//! to F32 (~8 GB RAM) and takes several minutes.

use std::fs::File;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::Device;

use qwen_tts_backend_pure_rust::config::ModelConfig;
use qwen_tts_backend_pure_rust::pipeline::Pipeline;
use qwen_tts_backend_pure_rust::talker::Talker;
use qwen_tts_backend_pure_rust::code_predictor::CodePredictor;
use qwen_tts_runtime::SynthesisRequest;

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Root of the project — works when tests are run from the workspace root.
fn project_root() -> &'static Path {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("D:\\qwen_tts"));
    root
}

fn talker_path() -> std::path::PathBuf {
    project_root().join("models").join("qwen-talker-1.7b-base-Q8_0.gguf")
}

fn codec_path() -> std::path::PathBuf {
    project_root().join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
}

// ---------------------------------------------------------------------------
// Structural tests (fast, no dequantization)
// ---------------------------------------------------------------------------

/// Verify that the talker GGUF parses metadata correctly.
#[test]
fn test_talker_config_values() {
    let path = talker_path();
    assert!(path.exists(), "talker GGUF not found: {}", path.display());

    let mut file = File::open(&path).expect("open talker GGUF");
    let content = gguf_file::Content::read(&mut file).expect("parse GGUF header");

    let cfg = ModelConfig::from_gguf(&content.metadata);
    assert!(cfg.d_model > 0, "d_model={} should be >0", cfg.d_model);
    assert!(cfg.d_model > 0, "d_model={} should be >0", cfg.d_model);
    assert!(cfg.n_layers > 0, "n_layers={} should be >0", cfg.n_layers);
    assert!(cfg.n_heads > 0, "n_heads={} should be >0", cfg.n_heads);
    assert!(cfg.vocab_size > 0, "vocab_size={} should be >0", cfg.vocab_size);
    assert!((cfg.norm_eps - 1e-6).abs() < 1e-5, "norm_eps={} unexpected", cfg.norm_eps);

    // Qwen3-TTS 1.7B expected values (from architecture qwen3-tts)
    assert_eq!(cfg.d_model, 2048, "expect 2048 hidden dim");
    assert!(cfg.n_layers >= 28, "expect 28+ layers (talker), got {}", cfg.n_layers);
    assert_eq!(cfg.n_heads, 16, "expect 16 heads");
    assert!(cfg.n_kv_heads > 0, "KV heads must be >0");
    assert!(cfg.vocab_size > 100000, "expect large vocab, got {}", cfg.vocab_size);
    assert!(cfg.max_seq_len >= 8192, "expect >=8192 context length, got {}", cfg.max_seq_len);

    // Check for code predictor metadata
    let num_code_groups = content
        .metadata
        .get("qwen3-tts.num_code_groups")
        .and_then(|v| v.to_u32().ok())
        .expect("num_code_groups should exist");
    assert_eq!(num_code_groups, 16, "expect 16 code groups");

    let pred_hidden = content
        .metadata
        .get("qwen3-tts.code_pred.embedding_length")
        .and_then(|v| v.to_u32().ok())
        .expect("code_pred.embedding_length should exist");
    assert_eq!(pred_hidden, 1024, "expect 1024 predictor hidden dim");
}

/// Verify that the talker GGUF contains all required tensor names.
#[test]
fn test_talker_tensor_names() {
    let path = talker_path();
    let mut file = File::open(&path).expect("open talker GGUF");
    let content = gguf_file::Content::read(&mut file).expect("parse GGUF header");

    // Check talker tensors exist — qwen3-tts architecture uses "talker." prefix
    let required = &[
        "talker.text_embd.weight",
        "talker.output_norm.weight",
        "talker.blk.0.attn_q.weight",
        "talker.blk.0.attn_k.weight",
        "talker.blk.0.attn_v.weight",
        "talker.blk.0.attn_output.weight",
        "talker.blk.0.attn_norm.weight",
        "talker.blk.0.ffn_gate.weight",
        "talker.blk.0.ffn_up.weight",
        "talker.blk.0.ffn_down.weight",
        "talker.blk.0.ffn_norm.weight",
        "talker.blk.27.attn_q.weight",
        "talker.codec_embd.weight",
        "talker.codec_head.weight",
    ];
    for name in required {
        assert!(
            content.tensor_infos.contains_key(*name),
            "missing talker tensor: {name}",
        );
    }

    // Check code predictor tensors — qwen3-tts uses "code_pred.blk." (not "code_pred.layers.")
    let cp_required = &[
        "code_pred.output_norm.weight",
        "code_pred.mtp_proj.weight",
        "code_pred.codec_embd.0.weight",
        "code_pred.lm_head.0.weight",
        "code_pred.blk.0.attn_norm.weight",
        "code_pred.blk.0.ffn_norm.weight",
        "code_pred.blk.0.attn_q.weight",
        "code_pred.blk.0.attn_k.weight",
        "code_pred.blk.0.attn_v.weight",
        "code_pred.blk.0.attn_output.weight",
        "code_pred.blk.0.ffn_gate.weight",
        "code_pred.blk.0.ffn_up.weight",
        "code_pred.blk.0.ffn_down.weight",
        "code_pred.blk.0.attn_q_norm.weight",
        "code_pred.blk.0.attn_k_norm.weight",
    ];
    for name in cp_required {
        assert!(
            content.tensor_infos.contains_key(*name),
            "missing code_pred tensor: {name}",
        );
    }

    // Count total
    assert!(
        content.tensor_infos.len() >= 400,
        "expected 480+ tensors, got {}",
        content.tensor_infos.len(),
    );
}

/// Verify that the codec GGUF can be parsed and contains expected tensors.
#[test]
fn test_codec_tensors() {
    let path = codec_path();
    assert!(path.exists(), "codec GGUF not found: {}", path.display());

    let mut file = File::open(&path).expect("open codec GGUF");
    let content = gguf_file::Content::read(&mut file).expect("parse GGUF header");

    // Note: codebooks use .codebook (no .weight suffix), rest use normal .weight
    let required = &[
        "tok_dec.vq_first.0.codebook",
        "tok_dec.vq_rest.0.codebook",
        "tok_dec.vq_rest.14.codebook",
        "tok_dec.pre_conv.weight",
        "tok_dec.pre_conv.bias",
        "tok_dec.pre_tfm.input_proj.weight",
        "tok_dec.pre_tfm.input_proj.bias",
        "tok_dec.pre_tfm.blk.0.attn_q.weight",
        "tok_dec.pre_tfm.blk.0.attn_k.weight",
        "tok_dec.pre_tfm.blk.0.attn_v.weight",
        "tok_dec.pre_tfm.blk.0.attn_output.weight",
        "tok_dec.pre_tfm.blk.0.ffn_gate.weight",
        "tok_dec.pre_tfm.blk.0.ffn_up.weight",
        "tok_dec.pre_tfm.blk.0.ffn_down.weight",
        "tok_dec.pre_tfm.norm.weight",
        "tok_dec.pre_tfm.output_proj.weight",
        "tok_dec.pre_tfm.output_proj.bias",
        "tok_dec.upsample.0.dwconv.weight",
        "tok_dec.upsample.1.conv.weight",
        "tok_dec.dec.0.conv.weight",
        "tok_dec.dec.0.conv.bias",
        "tok_dec.dec.1.conv_t.weight",
        "tok_dec.dec.5.snake.alpha",
        "tok_dec.dec.5.snake.beta",
        "tok_dec.dec.6.conv.weight",
        "tok_dec.dec.6.conv.bias",
    ];
    for name in required {
        assert!(
            content.tensor_infos.contains_key(*name),
            "missing codec tensor: {name}",
        );
    }

    assert!(
        content.tensor_infos.len() >= 390,
        "expected 390+ tensors, got {}",
        content.tensor_infos.len(),
    );
}

/// Quick test: check if CUDA is available at runtime.
#[test]
fn test_cuda_available() {
    let result = Device::new_cuda(0);
    if let Ok(dev) = result {
        eprintln!("CUDA device available: {dev:?}");
    } else {
        eprintln!("CUDA not available (expected on non-CUDA builds): {:?}", result.err());
    }
    // Not asserting — CUDA may be absent in CPU-only builds
}

/// Verify Talker loads from the real GGUF (dequantizes all weights to F32).
/// This is the heaviest test — ~8 GB RAM for the 1.7B Q8_0 model.
#[test]
#[ignore = "requires ~8 GB RAM and ~5 seconds"]
fn test_talker_loads() {
    let device = candle_core::Device::Cpu;
    let talker = Talker::from_gguf(&talker_path(), &device)
        .expect("talker should load from real GGUF");
    assert_eq!(talker.config().d_model, 2048);
    // Qwen3-TTS talker has 28 layers; config reads metadata dynamically
    assert!(
        talker.config().n_layers >= 28,
        "expect 28+ layers, got {}",
        talker.config().n_layers
    );
}

/// Verify CodePredictor loads from the same GGUF.
#[test]
#[ignore = "requires ~8 GB RAM and 1-3 minutes"]
fn test_code_predictor_loads() {
    use std::fs::File;
    let device = candle_core::Device::Cpu;

    let path = talker_path();
    let mut file = File::open(&path).expect("open talker GGUF");
    let content = gguf_file::Content::read(&mut file).expect("parse GGUF header");

    let predictor = CodePredictor::from_gguf(&content, &mut file, &device)
        .expect("code predictor should load from talker GGUF");
    assert_eq!(predictor.num_acoustic(), 15, "expect 15 acoustic codebooks");
    assert_eq!(predictor.hidden_size(), 1024, "expect 1024 code predictor hidden dim");
}

/// Full pipeline end-to-end: load all models, synthesize a short text,
/// verify we get valid 24 kHz PCM audio.
#[test]
#[ignore = "requires ~10 GB RAM and ~19 minutes (128 frames, KV cache)"]
fn test_pipeline_full_synthesize() {
    let mut pipeline = Pipeline::new(&talker_path(), &codec_path(), &Device::Cpu)
        .expect("pipeline should load");

    let request = SynthesisRequest {
        text: "Hello, world.".into(),
        language: "en".into(),
        speaker: None,
        instruct: None,
        ref_audio_path: None,
        ref_text: None,
        seed: Some(42),
        max_new_tokens: Some(128),
        temperature: Some(1.0),
        top_k: Some(40),
        top_p: Some(0.9),
        repetition_penalty: None,
        do_sample: None,
        out_path: Path::new("/tmp/pure-rust-e2e.wav").to_path_buf(),
        device: qwen_tts_runtime::DeviceKind::Cpu,
        models: qwen_tts_core::TtsModelSet::new(&talker_path(), &codec_path()),
    };

    let audio = pipeline.synthesize(&request)
        .expect("synthesize should succeed");

    // Verify output
    assert!(!audio.is_empty(), "audio must not be empty");

    let sample_rate = 24000_u32;
    let min_duration_sec = 0.5;
    let min_samples = (sample_rate as f32 * min_duration_sec) as usize;
    assert!(
        audio.len() >= min_samples,
        "audio too short: {} samples < {} samples ({min_duration_sec}s at 24 kHz)",
        audio.len(),
        min_samples,
    );

    // Sanity check: not all zeros
    let nonzero = audio.iter().filter(|&&s| s != 0).count();
    assert!(
        nonzero > audio.len() / 2,
        "more than half the samples are silent — something is wrong",
    );

    // Expected data size for PCM16
    let data_bytes = audio.len() * 2;
    let wav_header = 44;
    let expected_file_size = data_bytes + wav_header;

    log::info!(
        "E2E synthesis OK: {} audio samples ({} bytes PCM16), expected WAV size ~{} bytes",
        audio.len(),
        data_bytes,
        expected_file_size,
    );
}

/// Minimal pipeline verification: 4 frames only, confirms forward pass works.
#[test]
#[ignore = "requires ~8 GB RAM and ~40 seconds"]
fn test_pipeline_minimal_synthesize() {
    let mut pipeline = Pipeline::new(&talker_path(), &codec_path(), &Device::Cpu)
        .expect("pipeline should load");

    let request = SynthesisRequest {
        text: "Hi.".into(),
        language: "en".into(),
        speaker: None,
        instruct: None,
        ref_audio_path: None,
        ref_text: None,
        seed: Some(42),
        max_new_tokens: Some(4),
        temperature: Some(1.0),
        top_k: Some(40),
        top_p: Some(0.9),
        repetition_penalty: None,
        do_sample: None,
        out_path: Path::new("/tmp/pure-rust-e2e-mini.wav").to_path_buf(),
        device: qwen_tts_runtime::DeviceKind::Cpu,
        models: qwen_tts_core::TtsModelSet::new(&talker_path(), &codec_path()),
    };

    let audio = pipeline.synthesize(&request)
        .expect("minimal synthesize should succeed");
    assert!(!audio.is_empty(), "audio must not be empty");
    // With 4 frames at 12 Hz * 24000 / 12Hz frame rate we expect audio
    // 4 frames * 24000/12 samples = 8000 samples minimum roughly
    let min_samples = 1000; // very conservative
    assert!(
        audio.len() >= min_samples,
        "audio too short: {} samples < {}",
        audio.len(),
        min_samples,
    );

    // Not all zeros
    let nonzero = audio.iter().filter(|&&s| s != 0).count();
    assert!(
        nonzero > audio.len() / 4, // at least 25% non-zero for 4 frames
        "too many silent samples — {}/{} nonzero",
        nonzero,
        audio.len(),
    );

    log::info!(
        "Minimal E2E OK: {} samples from 4 frames, {} nonzero",
        audio.len(),
        nonzero,
    );
}
