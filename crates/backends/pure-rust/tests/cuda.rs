//! CUDA-aware tests for the pure-rust backend.
//!
//! These tests require `--features cuda` and a CUDA-capable GPU.
//! Run with: cargo test --release --features cuda --test cuda -- --nocapture

use std::path::Path;

use candle_core::Device;

use qwen_tts_backend_pure_rust::pipeline::Pipeline;
use qwen_tts_backend_pure_rust::talker::Talker;
use qwen_tts_runtime::SynthesisRequest;

fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("D:\\qwen_tts"))
}

fn talker_path() -> std::path::PathBuf {
    project_root().join("models").join("qwen-talker-1.7b-base-Q8_0.gguf")
}

fn codec_path() -> std::path::PathBuf {
    project_root().join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
}

/// Verify that CUDA is available at runtime.
#[test]
fn cuda_device_available() {
    let dev = Device::new_cuda(0).expect("CUDA device 0 should be available");
    eprintln!("CUDA device: {dev:?}");
}

/// Manual check: load talker on GPU (uses ~8GB VRAM, conflicts with pipeline tests).
/// Run in isolation: cargo test --release --features cuda --test cuda talker_loads_on_cuda -- --nocapture
#[test]
#[ignore]
fn talker_loads_on_cuda() {
    let device = Device::new_cuda(0).expect("CUDA should be available");
    let talker = Talker::from_gguf(&talker_path(), &device)
        .expect("talker should load on CUDA");
    eprintln!("Talker loaded on CUDA: {} layers", talker.config().n_layers);
}

/// Quick check: can we initialize the pipeline with CUDA and synthesize audio?
fn cuda_synthesize_inner(num_frames: u32) {
    let device = Device::new_cuda(0).expect("CUDA should be available");
    let mut pipeline = Pipeline::new(&talker_path(), &codec_path(), &device)
        .expect("pipeline should load on CUDA");

    let request = SynthesisRequest {
        text: "Hello, world.".into(),
        language: "en".into(),
        speaker: None,
        instruct: None,
        ref_audio_path: None,
        ref_text: None,
        seed: Some(42),
        max_new_tokens: Some(num_frames as i32),
        temperature: Some(1.0),
        top_k: Some(40),
        top_p: Some(0.9),
        repetition_penalty: None,
        do_sample: None,
        out_path: Path::new("tests/cuda-e2e.wav").to_path_buf(),
        device: qwen_tts_runtime::DeviceKind::Cuda,
        models: qwen_tts_core::TtsModelSet::new(&talker_path(), &codec_path()),
    };

    let start = std::time::Instant::now();
    let audio = pipeline.synthesize(&request, None)
        .expect("CUDA synthesis should succeed");
    let elapsed = start.elapsed();

    assert!(!audio.is_empty(), "audio must not be empty");
    assert!(audio.len() >= (num_frames as usize) * 100, "audio too short: {} samples", audio.len());

    let nonzero = audio.iter().filter(|&&s| s != 0).count();
    assert!(
        nonzero > audio.len() / 2,
        "more than half silent — CUDA output is wrong"
    );

    eprintln!(
        "CUDA pipeline OK: {} frames, {} audio samples, {:.2}s",
        num_frames, audio.len(), elapsed.as_secs_f64(),
    );
}

#[test]
fn cuda_pipeline_synthesize_8_frames() {
    cuda_synthesize_inner(8);
}

#[test]
fn cuda_pipeline_synthesize_32_frames() {
    cuda_synthesize_inner(32);
}
