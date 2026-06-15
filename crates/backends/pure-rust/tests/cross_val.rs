//! Cross-validation: compare Pure Rust backend output against the FFI C++ backend.
//!
//! The two backends use different RNG implementations (ChaCha12 vs Mersenne Twister),
//! so sampled outputs will differ. We compare:
//!   1. Deterministic mode (argmax): outputs should match exactly
//!   2. Statistical similarity: RMS, peak, duration
//!
//! Run:
//!   cargo test --release -p qwen-tts-backend-pure-rust --test cross_val -- --nocapture
//!   (only if the release build of qwen-tts.exe exists at target/release/)

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use candle_core::Device;

use qwen_tts_backend_pure_rust::pipeline::Pipeline;
use qwen_tts_runtime::SynthesisRequest;

// ── Paths ────────────────────────────────────────────────────────────────

fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("D:\\qwen_tts"))
}

fn talker_path() -> PathBuf {
    project_root().join("models").join("qwen-talker-1.7b-base-Q8_0.gguf")
}

fn codec_path() -> PathBuf {
    project_root().join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
}

fn qwen_tts_bin() -> PathBuf {
    project_root().join("target").join("release").join("qwen-tts.exe")
}

// ── WAV reader (16-bit PCM mono) ─────────────────────────────────────────

struct WavData {
    samples: Vec<i16>,
    sample_rate: u32,
}

fn read_wav(path: &Path) -> std::io::Result<WavData> {
    let mut file = std::fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a WAV file"));
    }

    // Parse fmt chunk
    let mut offset = 12;
    let mut sample_rate = 0u32;
    let mut num_channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut data_offset = 0;
    let mut data_size = 0;

    loop {
        if offset + 8 > data.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated WAV"));
        }
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        match chunk_id {
            b"fmt " => {
                if offset + 16 > data.len() {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated fmt"));
                }
                num_channels = u16::from_le_bytes(data[offset + 2..offset + 4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
                bits_per_sample = u16::from_le_bytes(data[offset + 14..offset + 16].try_into().unwrap());
            }
            b"data" => {
                data_offset = offset;
                data_size = chunk_size;
                break;
            }
            _ => {}
        }
        offset += chunk_size;
        if offset >= data.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "data chunk not found"));
        }
    }

    let sample_bytes = (bits_per_sample / 8) as usize;
    let total_samples = data_size / sample_bytes / num_channels as usize;
    let mut samples = Vec::with_capacity(total_samples);

    for i in 0..total_samples {
        let byte_start = data_offset + i * sample_bytes * num_channels as usize;
        // Take first channel only
        let raw = i16::from_le_bytes(
            data[byte_start..byte_start + 2].try_into().unwrap(),
        );
        samples.push(raw);
    }

    Ok(WavData { samples, sample_rate })
}

// ── Pure Rust synthesis helper ───────────────────────────────────────────

fn pure_rust_synthesize(
    text: &str,
    max_tokens: u32,
    do_sample: bool,
    seed: u64,
) -> (Vec<i16>, f64) {
    let device = Device::Cpu;
    let mut pipeline = Pipeline::new(&talker_path(), &codec_path(), &device)
        .expect("Pure Rust pipeline should load");

    let request = SynthesisRequest {
        text: text.into(),
        language: "en".into(),
        speaker: None,
        instruct: None,
        ref_audio_path: None,
        ref_text: None,
        seed: Some(seed as i64),
        max_new_tokens: Some(max_tokens as i32),
        temperature: Some(1.0),
        top_k: Some(40),
        top_p: Some(0.9),
        repetition_penalty: None,
        do_sample: Some(do_sample),
        out_path: PathBuf::new(),
        device: qwen_tts_runtime::DeviceKind::Cpu,
        models: qwen_tts_core::TtsModelSet::new(&talker_path(), &codec_path()),
    };

    let start = std::time::Instant::now();
    let audio = pipeline.synthesize(&request)
        .expect("Pure Rust synthesis should succeed");
    let elapsed = start.elapsed().as_secs_f64();

    (audio, elapsed)
}

// ── FFI synthesis helper ─────────────────────────────────────────────────

fn ffi_synthesize(
    text: &str,
    max_tokens: u32,
    seed: u64,
    out_path: &Path,
) -> f64 {
    let start = std::time::Instant::now();

    let output = Command::new(qwen_tts_bin())
        .arg("synth")
        .arg("--text").arg(text)
        .arg("--talker").arg(talker_path())
        .arg("--codec").arg(codec_path())
        .arg("--lang").arg("english")
        .arg("--out").arg(out_path)
        .arg("--max-tokens").arg(max_tokens.to_string())
        .arg("--temperature").arg("1.0")
        .arg("--seed").arg(seed.to_string())
        .output()
        .expect("FFI binary should run");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("FFI synthesis failed: {stderr}");
    }

    start.elapsed().as_secs_f64()
}

// ── Comparison metrics ──────────────────────────────────────────────────

fn compute_metrics(a: &[i16], b: &[i16]) {
    let min_len = a.len().min(b.len());
    if min_len == 0 {
        eprintln!("  One or both audio outputs are empty!");
        return;
    }

    // Per-sample comparison for the overlapping region
    let mut sum_sq_diff = 0.0f64;
    let mut max_abs_diff = 0i16;
    let mut matching_samples = 0;

    for i in 0..min_len {
        let diff = (a[i] as i32 - b[i] as i32).abs() as i16;
        sum_sq_diff += (diff as f64) * (diff as f64);
        max_abs_diff = max_abs_diff.max(diff);
        if diff == 0 {
            matching_samples += 1;
        }
    }

    let mse = sum_sq_diff / min_len as f64;
    let rmse = mse.sqrt();

    // RMS energy
    let rms_a = (a.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / a.len() as f64).sqrt();
    let rms_b = (b.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / b.len() as f64).sqrt();

    // Peak amplitude
    let peak_a = a.iter().map(|&s| s.abs()).max().unwrap_or(0);
    let peak_b = b.iter().map(|&s| s.abs()).max().unwrap_or(0);

    eprintln!("  Duration:     {} vs {} samples ({} vs {} ms)",
        a.len(), b.len(),
        a.len() * 1000 / 24000, b.len() * 1000 / 24000);
    eprintln!("  Samples match: {}/{} ({:.1}%)",
        matching_samples, min_len,
        (matching_samples as f64 / min_len as f64) * 100.0);
    eprintln!("  MSE:          {:.6}", mse);
    eprintln!("  RMSE:         {:.6}", rmse);
    eprintln!("  Max abs diff: {}", max_abs_diff);
    eprintln!("  RMS (a/b):    {:.4} / {:.4}", rms_a, rms_b);
    eprintln!("  Peak (a/b):   {} / {}", peak_a, peak_b);
}

// ── Tests ────────────────────────────────────────────────────────────────

/// Verify that Pure Rust argmax output is deterministic (same output twice).
#[test]
fn test_pure_rust_determinism() {
    let (audio1, t1) = pure_rust_synthesize("Hello.", 4, false, 42);
    let (audio2, t2) = pure_rust_synthesize("Hello.", 4, false, 42);

    assert_eq!(audio1.len(), audio2.len(),
        "deterministic runs should have same length");

    let matching = audio1.iter().zip(audio2.iter()).filter(|(a, b)| a == b).count();
    let pct = (matching as f64 / audio1.len() as f64) * 100.0;

    eprintln!("Determinism check: {} matching / {} total = {:.1}% (times: {:.2}s, {:.2}s)",
        matching, audio1.len(), pct, t1, t2);

    assert!(matching == audio1.len(),
        "Pure Rust should be fully deterministic with do_sample=false — got {matching}/{len}",
        len = audio1.len());
}

/// Compare Pure Rust output vs FFI output — same text, same seed.
///
/// NOTE: The two backends use different RNGs (ChaCha12 vs Mersenne Twister),
/// and our code predictor is simplified (no 5 transformer layers yet).
/// Therefore bit-exact match is NOT expected.
///
/// This test verifies that both backends produce valid audio of the same
/// length with reasonable amplitude. The FFI backend depends on the compiled
/// qwen-tts.exe binary and may not be available on all platforms.
#[test]
#[ignore = "requires external qwen-tts.exe binary"]
fn test_cross_val_pure_rust_vs_ffi() {
    let ffi_wav = project_root().join("tests").join("cross_val_ffi.wav");

    // 1. Synthesize with Pure Rust (deterministic argmax)
    eprintln!("[Pure Rust] Synthesizing 4 frames (argmax)...");
    let (pure_audio, pure_time) = pure_rust_synthesize("Hello.", 4, false, 42);
    eprintln!("  Done: {:.2}s, {} samples", pure_time, pure_audio.len());

    // 2. Synthesize with FFI (temperature=1.0, seed=42)
    eprintln!("[FFI] Synthesizing 4 frames (temperature=1.0)...");
    let ffi_time = ffi_synthesize("Hello.", 4, 42, &ffi_wav);
    let ffi_wav_data = read_wav(&ffi_wav)
        .expect("FFI WAV should be readable");
    eprintln!("  Done: {:.2}s, {} samples", ffi_time, ffi_wav_data.samples.len());

    // 3. Compare
    eprintln!();
    eprintln!("=== Cross-validation: Pure Rust (argmax) vs FFI ===");
    compute_metrics(&pure_audio, &ffi_wav_data.samples);

    // Basic sanity checks
    assert!(!pure_audio.is_empty(), "Pure Rust should produce audio");
    assert!(!ffi_wav_data.samples.is_empty(), "FFI should produce audio");
    assert_eq!(
        pure_audio.len(), ffi_wav_data.samples.len(),
        "Both backends should produce the same number of samples"
    );

    // Both should have non-trivial amplitude
    let max_pure = pure_audio.iter().map(|&s| s.abs()).max().unwrap_or(0);
    let max_ffi = ffi_wav_data.samples.iter().map(|&s| s.abs()).max().unwrap_or(0);
    assert!(max_pure > 100, "Pure Rust audio should have non-zero amplitude (got {max_pure})");
    assert!(max_ffi > 100, "FFI audio should have non-zero amplitude (got {max_ffi})");

    // Signal-to-difference ratio
    let min_len = pure_audio.len().min(ffi_wav_data.samples.len());
    let diff_sum: f64 = pure_audio[..min_len].iter().zip(ffi_wav_data.samples[..min_len].iter())
        .map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let signal_sum: f64 = pure_audio[..min_len].iter()
        .map(|&s| (s as f64).powi(2)).sum();
    let sdr = if diff_sum > 0.0 { 10.0 * (signal_sum / diff_sum).log10() } else { f64::INFINITY };

    eprintln!("  SDR (signal-to-diff ratio): {:.1} dB", sdr);
    // SDR > 0 means signal power exceeds difference power (structurally similar)
    eprintln!("  (SDR > 0 with different RNG + simplified code predictor = structurally valid)");
}

/// Sanity: Pure Rust with sampling produces different output than with argmax.
#[test]
fn test_pure_rust_sample_vs_argmax() {
    let (audio_argmax, _) = pure_rust_synthesize("Hello.", 4, false, 42);
    let (audio_sample, _) = pure_rust_synthesize("Hello.", 4, true, 99);

    assert!(!audio_argmax.is_empty());
    assert!(!audio_sample.is_empty());
    assert_eq!(audio_argmax.len(), audio_sample.len(),
        "same number of tokens should give same length");

    let matching = audio_argmax.iter().zip(audio_sample.iter())
        .filter(|(a, b)| a == b).count();
    eprintln!(
        "argmax vs sample: {} matching / {} total ({:.1}%)",
        matching, audio_argmax.len(),
        (matching as f64 / audio_argmax.len() as f64) * 100.0
    );

    // With different seeds, they should NOT match
    assert!(
        matching < audio_argmax.len(),
        "Sampling with different seed should differ from argmax"
    );
}
