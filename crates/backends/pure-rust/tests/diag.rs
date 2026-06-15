//! Diagnostic: compare prompt layout and first-frame codes between pure Rust and C++ FFI.

use std::path::Path;

use candle_core::Device;
use qwen_tts_backend_pure_rust::pipeline::Pipeline;
use qwen_tts_runtime::SynthesisRequest;

fn project_root() -> &'static Path {
    Path::new("D:\\qwen_tts")
}

#[test]
#[ignore = "diagnostic: ~10 GB RAM, ~5 min"]
fn test_diag_prompt_and_first_frame() {
    let talker_path = project_root().join("models").join("qwen-talker-1.7b-base-Q8_0.gguf");
    let codec_path = project_root().join("models").join("qwen-tokenizer-12hz-Q8_0.gguf");

    let mut pipeline = Pipeline::new(&talker_path, &codec_path, &Device::Cpu)
        .expect("pipeline should load");

    let request = SynthesisRequest {
        text: "Hello, this is a test.".into(),
        language: "english".into(),
        speaker: None,
        instruct: None,
        ref_audio_path: None,
        ref_text: None,
        seed: Some(42),
        max_new_tokens: Some(1),
        temperature: Some(1.0),
        top_k: Some(40),
        top_p: Some(0.9),
        repetition_penalty: None,
        do_sample: None, // argmax
        out_path: Path::new("/tmp/diag-test.wav").to_path_buf(),
        device: qwen_tts_runtime::DeviceKind::Cpu,
        models: qwen_tts_core::TtsModelSet::new(&talker_path, &codec_path),
    };

    let (codes, n_frames, audio) = pipeline.synthesize_raw(&request, None)
        .expect("synthesize_raw should succeed");

    println!("Generated codes ({} frames):", n_frames);
    for f in 0..n_frames {
        let start = f * 16;
        let end = start + 16;
        if end <= codes.len() {
            let frame_codes: Vec<String> = codes[start..end].iter().map(|c| c.to_string()).collect();
            println!("  Frame {}: [{}]", f, frame_codes.join(", "));
        }
    }

    println!("Audio: {} samples, peak={}", audio.len(),
        audio.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0));
}
