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
        language: "english".into(),
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

    let audio = pipeline.synthesize(&request, None)
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
        language: "english".into(),
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

    let audio = pipeline.synthesize(&request, None)
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

/// Diagnostic: compare generated codes between pure-Rust and C++ FFI reference.
///
/// C++ FFI reference (argmax, seed=42, "Hello, this is a test.", max-tokens 2):
///   Frame 0: c0=404, c1=[0, 901, 81, 366, 647, 1301, 609, 546, 351, 205, 1659, 1607, 181, 754, 121]
///   Frame 1: c0=1014, c1=[969, 769, 326, 21, 1719, 2009, 153, 417, 649, 145, 932, 1832, 411, 83, 404]
#[test]
#[ignore = "diagnostic: ~10 GB RAM, ~5 min"]
fn test_compare_codes_with_cpp() {
    let mut pipeline = Pipeline::new(&talker_path(), &codec_path(), &Device::Cpu)
        .expect("pipeline should load");

    let request = SynthesisRequest {
        text: "Hello, this is a test.".into(),
        language: "english".into(),
        speaker: None,
        instruct: None,
        ref_audio_path: None,
        ref_text: None,
        seed: Some(42),
        max_new_tokens: Some(2),
        temperature: Some(1.0),
        top_k: Some(40),
        top_p: Some(0.9),
        repetition_penalty: None,
        do_sample: None, // argmax
        out_path: std::path::Path::new("/tmp/compare-codes.wav").to_path_buf(),
        device: qwen_tts_runtime::DeviceKind::Cpu,
        models: qwen_tts_core::TtsModelSet::new(&talker_path(), &codec_path()),
    };

    let (codes, n_frames, audio) = pipeline.synthesize_raw(&request, None)
        .expect("synthesize_raw should succeed");

    // Print generated codes
    println!("Pure-Rust generated codes ({n_frames} frames, {} total tokens):", codes.len());
    for f in 0..n_frames {
        let start = f * 16;
        let end = start + 16;
        if end <= codes.len() {
            let frame_codes: Vec<String> = codes[start..end].iter().map(|c| c.to_string()).collect();
            println!("  Frame {}: [{}]", f, frame_codes.join(", "));
        }
    }

    // C++ reference (argmax, seed=42, "Hello, this is a test.", 2 tokens)
    let cpp_frame0: Vec<i32> = vec![404, 0, 901, 81, 366, 647, 1301, 609, 546, 351, 205, 1659, 1607, 181, 754, 121];
    let cpp_frame1: Vec<i32> = vec![1014, 969, 769, 326, 21, 1719, 2009, 153, 417, 649, 145, 932, 1832, 411, 83, 404];

    println!();
    println!("C++ FFI reference codes:");
    println!("  Frame 0: [{}]", cpp_frame0.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", "));
    println!("  Frame 1: [{}]", cpp_frame1.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", "));

    // Compare frame 0
    if n_frames >= 1 {
        let rust_f0: Vec<i32> = codes[0..16].to_vec();
        let match_f0: Vec<bool> = rust_f0.iter().zip(cpp_frame0.iter()).map(|(a, b)| a == b).collect();
        let match_count_f0 = match_f0.iter().filter(|&&m| m).count();
        println!();
        println!("Frame 0 match: {}/16", match_count_f0);
        for i in 0..16 {
            if rust_f0[i] != cpp_frame0[i] {
                println!("  c{}: rust={} cpp={} MISMATCH", i, rust_f0[i], cpp_frame0[i]);
            }
        }
    }

    if n_frames >= 2 {
        let rust_f1: Vec<i32> = codes[16..32].to_vec();
        let match_f1: Vec<bool> = rust_f1.iter().zip(cpp_frame1.iter()).map(|(a, b)| a == b).collect();
        let match_count_f1 = match_f1.iter().filter(|&&m| m).count();
        println!("Frame 1 match: {}/16", match_count_f1);
        for i in 0..16 {
            if rust_f1[i] != cpp_frame1[i] {
                println!("  c{}: rust={} cpp={} MISMATCH", i, rust_f1[i], cpp_frame1[i]);
            }
        }
    }

    // Print audio stats
    let peak = audio.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    let rms: f64 = (audio.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / audio.len() as f64).sqrt();
    println!();
    println!("Audio: {} samples, peak={peak}, rms={rms:.1}", audio.len());

    // Dump summary
    if n_frames >= 1 {
        let rust_f0: Vec<i32> = codes[0..16].to_vec();
        let match_f0 = rust_f0.iter().zip(cpp_frame0.iter()).filter(|(a, b)| a == b).count();
        if match_f0 == 16 {
            println!("RESULT: Frame 0 EXACT MATCH ✓");
        } else {
            println!("RESULT: Frame 0 DIFFERS ({} match, {} mismatch)", match_f0, 16 - match_f0);
        }
    }
    if n_frames >= 2 {
        let rust_f1: Vec<i32> = codes[16..32].to_vec();
        let match_f1 = rust_f1.iter().zip(cpp_frame1.iter()).filter(|(a, b)| a == b).count();
        if match_f1 == 16 {
            println!("RESULT: Frame 1 EXACT MATCH ✓");
        } else {
            println!("RESULT: Frame 1 DIFFERS ({} match, {} mismatch)", match_f1, 16 - match_f1);
        }
    }
}

/// Dump all metadata keys from the talker GGUF.
#[test]
#[ignore = "diagnostic"]
fn test_dump_metadata() {
    use gguf_file::Value;

    let mut file = File::open(&talker_path()).expect("open talker GGUF");
    let content = gguf_file::Content::read(&mut file).expect("parse GGUF header");
    let meta = &content.metadata;

    println!("GGUF metadata keys:");
    let mut keys: Vec<&String> = meta.keys().collect();
    keys.sort();
    for k in &keys {
        let v = &meta[*k];
        let preview = match v {
            Value::String(s) => format!("string(len={})", s.len()),
            Value::U32(u) => format!("u32({u})"),
            Value::F32(f) => format!("f32({f})"),
            Value::Array(arr) => format!("array(len={})", arr.len()),
            Value::Bool(b) => format!("bool({b})"),
            _ => format!("{v:?}"),
        };
        println!("  {k}: {preview}");
    }

    // Dump language names
    println!();
    if let Some(names_arr) = meta.get("qwen3-tts.codec.language_names").and_then(|v| v.to_vec().ok()) {
        let ids_vec = meta.get("qwen3-tts.codec.language_ids").and_then(|v| v.to_vec().ok()).cloned().unwrap_or_default();
        for (i, n) in names_arr.iter().enumerate() {
            let name = n.to_string().ok().cloned().unwrap_or_default();
            let id = ids_vec.get(i).and_then(|v| v.to_u32().ok()).unwrap_or(0);
            println!("  language[{i}]: id={id} name='{name}'");
        }
    } else {
        println!("  (no qwen3-tts.codec.language_names key)");
    }

    // Count codec_embd and lm_head tensors in the GGUF
    let codec_embd_count = content.tensor_infos.keys().filter(|k| k.contains("code_pred.codec_embd.")).count();
    let lm_head_count = content.tensor_infos.keys().filter(|k| k.contains("code_pred.lm_head.")).count();
    println!("  code_pred.codec_embd.* count: {codec_embd_count}");
    println!("  code_pred.lm_head.* count: {lm_head_count}");

    // Dump speaker names
    if let Some(spk_arr) = meta.get("qwen3-tts.codec.speaker_names").and_then(|v| v.to_vec().ok()) {
        let ids_vec = meta.get("qwen3-tts.codec.speaker_ids").and_then(|v| v.to_vec().ok()).cloned().unwrap_or_default();
        let dial_vec = meta.get("qwen3-tts.codec.speaker_dialects").and_then(|v| v.to_vec().ok()).cloned().unwrap_or_default();
        for (i, n) in spk_arr.iter().enumerate() {
            let name = n.to_string().ok().cloned().unwrap_or_default();
            let id = ids_vec.get(i).and_then(|v| v.to_u32().ok()).unwrap_or(0);
            let dial = dial_vec.get(i).and_then(|v| v.to_string().ok().cloned()).unwrap_or_default();
            println!("  speaker[{i}]: id={id} name='{name}' dialect='{dial}'");
        }
    } else {
        println!("  (no qwen3-tts.codec.speaker_names key)");
    }
}

/// DEFINITIVE DIAGNOSTIC: Compare Q8Weights::from_gguf gemv against candle's
/// dequantized matmul for a REAL GGUF tensor (talker layer 0 attn_k).
///
/// This isolates whether the Q8 block layout interpretation is correct.
/// If the two results differ significantly, the block layout is wrong.
#[test]
#[ignore = "diagnostic: ~8 GB RAM"]
fn test_q8_vs_candle_dequant() {
    use qwen_tts_backend_pure_rust::qgemv::{Q8Weights, Q8Workspace};

    let path = talker_path();
    let device = Device::Cpu;
    let mut file = File::open(&path).expect("open talker GGUF");
    let content = gguf_file::Content::read(&mut file).expect("parse GGUF header");

    // Load attn_k of layer 0 via both paths
    let tensor_name = "talker.blk.0.attn_k.weight";

    // 1. Candle dequantized (reference)
    let qt = content.tensor(&mut file, tensor_name, &device)
        .expect("load tensor via candle");
    let w_f32 = qt.dequantize(&device).expect("dequantize");
    let w_f32_vec: Vec<f32> = w_f32.flatten_all().unwrap().to_vec1().unwrap();

    let dims_after_reversal = w_f32.dims().to_vec();
    println!("Tensor '{}':", tensor_name);
    println!("  Candle dims (after reversal): {:?}", dims_after_reversal);

    // 2. Our Q8 loading
    let q8 = Q8Weights::from_gguf(&content, &mut file, tensor_name)
        .expect("load Q8 weight");
    let (q8_n, q8_k) = q8.shape();
    println!("  Q8Weights: n={}, k={}, blocks_per_row={}, padded_k={}", q8_n, q8_k, q8.blocks_per_row(), q8.padded_k());

    // 3. Create a deterministic test input
    let k_dim = q8_k; // in_features from our Q8 interpretation
    let mut x: Vec<f32> = Vec::with_capacity(k_dim);
    for i in 0..k_dim {
        x.push(((i * 7 + 3) % 100) as f32 / 50.0 - 1.0);
    }

    // 4. Our Q8 gemv result
    let mut ws = Q8Workspace::new();
    let y_q8 = q8.gemv(&x, &mut ws);

    // 5. Candle reference: need to match the same operation.
    //    Our gemv computes: y[i] = sum_j(W_q8[i][j] * x[j])  (i=output, j=input)
    //    Candle has W as a 2D tensor with dims = dims_after_reversal.
    //    If dims = [out, in], then: y = W @ x_tensor
    //    If dims = [in, out], then: y = x_tensor @ W (wrong!) or y = W^T @ x_tensor
    let x_tensor = candle_core::Tensor::from_vec(x.clone(), (k_dim,), &device).unwrap();

    // Try the correct matmul direction based on dims
    let out_dim_from_dims = dims_after_reversal[0];
    let in_dim_from_dims = dims_after_reversal[1];

    println!("  Candle dims[0]={}, dims[1]={}", out_dim_from_dims, in_dim_from_dims);
    println!("  Q8 n={} (our output dim), k={} (our input dim)", q8_n, q8_k);

    // Candle matmul: W[1024, 2048] @ x[2048, 1] → [1024, 1]
    // x needs to be column vector for candle matmul convention
    let x_col = x_tensor.reshape((k_dim, 1)).unwrap();
    let y_candle_2d = w_f32.matmul(&x_col).unwrap();
    let y_candle: Vec<f32> = y_candle_2d.reshape((out_dim_from_dims,)).unwrap().to_vec1().unwrap();
    println!("  Matmul: W[{},{}] @ x[{}] → y[{}]", out_dim_from_dims, in_dim_from_dims, k_dim, y_candle.len());

    if !y_candle.is_empty() {
        assert_eq!(y_q8.len(), y_candle.len(),
            "output length mismatch: Q8={} candle={}", y_q8.len(), y_candle.len());

        // Compare
        let mut max_rel_err = 0.0f32;
        let mut max_abs_err = 0.0f32;
        let mut first_mismatch = None;
        for i in 0..y_q8.len().min(32) {
            let abs_err = (y_q8[i] - y_candle[i]).abs();
            let rel_err = abs_err / y_candle[i].abs().max(1e-6);
            if rel_err > max_rel_err { max_rel_err = rel_err; }
            if abs_err > max_abs_err { max_abs_err = abs_err; }
            if first_mismatch.is_none() && abs_err > 0.1 {
                first_mismatch = Some(i);
            }
            if i < 8 {
                println!("  [{i}] Q8={:.6}  candle={:.6}  abs_err={:.6}  rel_err={:.4}%",
                    y_q8[i], y_candle[i], abs_err, rel_err * 100.0);
            }
        }
        println!("  Max abs err: {:.6}", max_abs_err);
        println!("  Max rel err: {:.2}%", max_rel_err * 100.0);
        if let Some(idx) = first_mismatch {
            println!("  First mismatch at index {idx}");
        } else {
            println!("  All first 32 elements match within 0.1 abs tolerance ✓");
        }
    }

}
