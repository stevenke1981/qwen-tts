//! End-to-end integration tests for `CpuBackend` (native-cpu-ffi).
//!
//! These tests load real Qwen3-TTS GGUF model files pointed to by the
//! `QWEN_TTS_TALKER` and `QWEN_TTS_CODEC` environment variables.  If either
//! variable is unset the tests skip gracefully with a printed message.
//!
//! On Windows the `qwen.dll` must also be locatable; the easiest way is to
//! prepend its directory to `PATH`:
//! ```text
//! $env:PATH = "D:\qwen_tts\vendor\qwentts.cpp\build\Release;$env:PATH"
//! $env:QWEN_TTS_TALKER = "path\to\talker.gguf"
//! $env:QWEN_TTS_CODEC  = "path\to\codec.gguf"
//! cargo test -p qwen-tts-backend-cpu --test e2e -- --nocapture
//! ```

use std::{
    env,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use qwen_tts_core::{validate_wav_file, AudioSpec, TtsModelSet};
use qwen_tts_backend_cpu::CpuBackend;
use qwen_tts_runtime::{DeviceKind, RuntimeBackend, SynthesisRequest};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read model paths from environment, or return `None` to skip.
fn get_models() -> Option<TtsModelSet> {
    let talker = env::var_os("QWEN_TTS_TALKER")?;
    let codec = env::var_os("QWEN_TTS_CODEC")?;
    Some(TtsModelSet::new(talker, codec))
}

/// Generate a unique output path in the temp directory.
fn out_path(label: &str) -> PathBuf {
    let dir = env::temp_dir().join("qwen-tts-e2e-cpu");
    let _ = std::fs::create_dir_all(&dir);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.join(format!("{label}_{nonce}.wav"))
}

/// Build a `SynthesisRequest` with the given text and common defaults.
fn make_request(text: &str, path: PathBuf, models: &TtsModelSet) -> SynthesisRequest {
    SynthesisRequest {
        text: text.to_owned(),
        language: "english".to_owned(),
        speaker: None,
        instruct: None,
        ref_audio_path: None,
        ref_text: None,
        seed: None,
        max_new_tokens: None,
        temperature: None,
        top_k: None,
        top_p: None,
        repetition_penalty: None,
        do_sample: None,
        out_path: path,
        device: DeviceKind::Cpu,
        models: models.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn e2e_basic_synthesis() {
    let Some(models) = get_models() else {
        eprintln!("SKIP: set QWEN_TTS_TALKER and QWEN_TTS_CODEC");
        return;
    };

    let backend = CpuBackend::new();

    let path = out_path("basic");
    let req = make_request(
        "Hello, this is a basic test of the text to speech system.",
        path,
        &models,
    );
    let response = backend.synthesize(&req).expect("basic synthesis");

    // WAV metadata
    assert_eq!(response.sample_rate_hz, 24_000);
    assert_eq!(response.channels, 1);
    assert_eq!(response.bits_per_sample, 16);
    assert!(
        response.data_size_bytes > 200,
        "audio data should be non-trivial (got {} bytes)",
        response.data_size_bytes
    );

    // File on disk validates
    let meta = validate_wav_file(&response.wav_path, AudioSpec::default())
        .expect("WAV file should be valid");
    assert!(meta.data_size_bytes > 200);
    assert_eq!(meta.sample_rate_hz, 24_000);

    eprintln!(
        "PASS: basic synthesis -> {} bytes @ {} Hz, {} channels, {} bps",
        meta.data_size_bytes, meta.sample_rate_hz, meta.channels, meta.bits_per_sample
    );
}

#[test]
fn e2e_synthesis_with_sampling_params() {
    let Some(models) = get_models() else {
        eprintln!("SKIP: set QWEN_TTS_TALKER and QWEN_TTS_CODEC");
        return;
    };

    let backend = CpuBackend::new();

    let path = out_path("sampling");
    let mut req = make_request(
        "Testing custom sampling parameters for synthesis.",
        path,
        &models,
    );
    req.seed = Some(42);
    req.temperature = Some(0.8);
    req.top_k = Some(40);
    req.top_p = Some(0.9);
    req.repetition_penalty = Some(1.1);

    let response = backend.synthesize(&req).expect("sampling synthesis");
    assert!(
        response.data_size_bytes > 200,
        "sampling output should be non-trivial"
    );

    let meta = validate_wav_file(&response.wav_path, AudioSpec::default())
        .expect("WAV should be valid");
    assert_eq!(meta.sample_rate_hz, 24_000);

    eprintln!(
        "PASS: sampling synthesis -> {} bytes",
        meta.data_size_bytes
    );
}

#[test]
fn e2e_deterministic_seed() {
    let Some(models) = get_models() else {
        eprintln!("SKIP: set QWEN_TTS_TALKER and QWEN_TTS_CODEC");
        return;
    };

    let backend = CpuBackend::new();
    let text = "Deterministic seed test with a fixed random seed.";

    let path_a = out_path("seed_a");
    let mut req_a = make_request(text, path_a.clone(), &models);
    req_a.seed = Some(12345);
    req_a.do_sample = Some(true);
    let resp_a = backend.synthesize(&req_a).expect("first deterministic run");

    let path_b = out_path("seed_b");
    let mut req_b = make_request(text, path_b.clone(), &models);
    req_b.seed = Some(12345);
    req_b.do_sample = Some(true);
    let resp_b = backend.synthesize(&req_b).expect("second deterministic run");

    let wav_a = std::fs::read(&resp_a.wav_path).expect("read WAV A");
    let wav_b = std::fs::read(&resp_b.wav_path).expect("read WAV B");

    assert_eq!(
        wav_a, wav_b,
        "same seed + text should produce identical WAV output"
    );

    eprintln!(
        "PASS: deterministic seed -> {} bytes (identical)",
        wav_a.len()
    );
}

#[test]
fn e2e_rejects_empty_text() {
    let Some(models) = get_models() else {
        eprintln!("SKIP: set QWEN_TTS_TALKER and QWEN_TTS_CODEC");
        return;
    };

    let backend = CpuBackend::new();

    let path = out_path("empty");
    let req = make_request("", path, &models);
    let result = backend.synthesize(&req);

    assert!(
        result.is_err(),
        "empty text should be rejected"
    );
    eprintln!("PASS: empty text correctly rejected");
}
