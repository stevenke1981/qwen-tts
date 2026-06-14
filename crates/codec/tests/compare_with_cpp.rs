//! Cross-verification test: compare our Rust codec decoder against
//! the C++ reference (qwen-codec.exe from the qwentts.cpp project).
//!
//! Strategy:
//! 1. Generate synthetic codes in T-major layout [T, 16]
//! 2. Run through our Rust CodecDecoder → audio_rust
//! 3. Transpose codes to K-major layout [16, T] (C++ layout)
//! 4. Pack as .rvq file (11 bits/code, LSB-first)
//! 5. Run qwen-codec.exe to decode → audio_cpp
//! 6. Compare audio_rust vs audio_cpp sample-by-sample

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

/// Path to the qwen-codec.exe reference binary.
fn qwen_codec_exe() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir.parent().and_then(|p| p.parent()).unwrap();
    ws_root
        .join("vendor")
        .join("qwentts.cpp")
        .join("build")
        .join("Release")
        .join("qwen-codec.exe")
}

/// Path to the codec GGUF model.
fn codec_gguf() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir.parent().and_then(|p| p.parent()).unwrap();
    ws_root
        .join("models")
        .join("qwen-tokenizer-12hz-Q8_0.gguf")
}

// ---------------------------------------------------------------------------
// .rvq file format: packed code stream, 11 bits per code, LSB-first.
// Layout is [K, T] K-major (K outer, T inner).
// ---------------------------------------------------------------------------

/// Pack codes (K-major flat [K*T]) into .rvq format bytes (11 bits/code).
fn pack_rvq(codes: &[i32], code_bits: u32) -> Vec<u8> {
    let mask = (1u32 << code_bits) - 1;
    let total_bits = codes.len() as u64 * code_bits as u64;
    let n_bytes = ((total_bits + 7) / 8) as usize;
    let mut out = vec![0u8; n_bytes];

    let mut acc: u64 = 0;
    let mut bits_in_acc = 0;
    let mut out_pos = 0;

    for &code in codes {
        acc |= ((code as u32 & mask) as u64) << bits_in_acc;
        bits_in_acc += code_bits as i32;
        while bits_in_acc >= 8 {
            out[out_pos] = (acc & 0xFF) as u8;
            out_pos += 1;
            acc >>= 8;
            bits_in_acc -= 8;
        }
    }
    if bits_in_acc > 0 {
        out[out_pos] = (acc & 0xFF) as u8;
    }
    out
}

/// Unpack .rvq format bytes back to flat i32 codes.
fn unpack_rvq(data: &[u8], n_codes: usize, code_bits: u32) -> Vec<i32> {
    let mask = (1u32 << code_bits) - 1;
    let mut out = vec![0i32; n_codes];

    let mut acc: u64 = 0;
    let mut bits_in_acc = 0;
    let mut in_pos = 0;

    for i in 0..n_codes {
        while bits_in_acc < code_bits as i32 && in_pos < data.len() {
            acc |= (data[in_pos] as u64) << bits_in_acc;
            in_pos += 1;
            bits_in_acc += 8;
        }
        out[i] = (acc & mask as u64) as i32;
        acc >>= code_bits;
        bits_in_acc -= code_bits as i32;
    }
    out
}

/// Transpose codes from T-major [T, K] to K-major [K, T].
fn transpose_codes(t_major: &[i32], t: usize, k: usize) -> Vec<i32> {
    assert_eq!(t_major.len(), t * k);
    let mut k_major = vec![0i32; t * k];
    for ti in 0..t {
        for ki in 0..k {
            // T-major: index = ti * k + ki
            // K-major: index = ki * t + ti
            k_major[ki * t + ti] = t_major[ti * k + ki];
        }
    }
    k_major
}

/// Read a WAV file that was written as f32 mono and return the samples.
/// Simple reader for our generated files: expects 24kHz, mono, f32.
fn read_wav_f32_samples(path: &Path) -> Vec<f32> {
    let data = fs::read(path).expect("cannot read WAV file");

    // Parse WAV header
    // RIFF header
    assert_eq!(&data[0..4], b"RIFF", "not a WAV file");
    let _file_size = u32::from_le_bytes(data[4..8].try_into().unwrap());
    assert_eq!(&data[8..12], b"WAVE", "not WAVE format");

    // Find fmt chunk
    let mut offset = 12;
    loop {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        if chunk_id == b"fmt " {
            let audio_format = u16::from_le_bytes(data[offset + 8..offset + 10].try_into().unwrap());
            let num_channels = u16::from_le_bytes(data[offset + 10..offset + 12].try_into().unwrap());
            let sample_rate = u32::from_le_bytes(data[offset + 12..offset + 16].try_into().unwrap());
            let _byte_rate = u32::from_le_bytes(data[offset + 16..offset + 20].try_into().unwrap());
            let _block_align = u16::from_le_bytes(data[offset + 20..offset + 22].try_into().unwrap());
            let bits_per_sample = u16::from_le_bytes(data[offset + 22..offset + 24].try_into().unwrap());

            assert_eq!(sample_rate, 24000, "expected 24kHz sample rate");
            assert_eq!(num_channels, 1, "expected mono");

            // Find data chunk
            let mut data_offset = offset + 8 + chunk_size;
            // Align to even boundary
            if data_offset % 2 != 0 {
                data_offset += 1;
            }
            loop {
                if data_offset + 8 > data.len() {
                    panic!("data chunk not found");
                }
                let data_chunk_id = &data[data_offset..data_offset + 4];
                let data_chunk_size =
                    u32::from_le_bytes(data[data_offset + 4..data_offset + 8].try_into().unwrap()) as usize;
                if data_chunk_id == b"data" {
                    let raw = &data[data_offset + 8..data_offset + 8 + data_chunk_size];
                    return match audio_format {
                        1 => {
                            // PCM integer
                            let bytes_per_sample = bits_per_sample as usize / 8;
                            let n_samples = raw.len() / bytes_per_sample;
                            let mut samples = Vec::with_capacity(n_samples);
                            for i in 0..n_samples {
                                let sample_start = i * bytes_per_sample;
                                let sample = match bits_per_sample {
                                    16 => {
                                        let val =
                                            i16::from_le_bytes(raw[sample_start..sample_start + 2].try_into().unwrap());
                                        val as f32 / 32768.0
                                    }
                                    32 => {
                                        let val =
                                            i32::from_le_bytes(raw[sample_start..sample_start + 4].try_into().unwrap());
                                        val as f32 / 2147483648.0
                                    }
                                    _ => panic!("unsupported bits_per_sample {bits_per_sample}"),
                                };
                                samples.push(sample);
                            }
                            samples
                        }
                        3 => {
                            // IEEE float
                            let n_samples = raw.len() / 4;
                            let mut samples = Vec::with_capacity(n_samples);
                            for i in 0..n_samples {
                                let val = f32::from_le_bytes(raw[i * 4..(i + 1) * 4].try_into().unwrap());
                                samples.push(val);
                            }
                            samples
                        }
                        _ => panic!("unsupported audio format {audio_format}"),
                    };
                }
                data_offset += 8 + data_chunk_size;
                if data_offset % 2 != 0 {
                    data_offset += 1;
                }
            }
        }
        offset += 8 + chunk_size;
        if offset % 2 != 0 {
            offset += 1;
        }
        if offset >= data.len() {
            panic!("fmt chunk not found");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const K: usize = 16; // codebooks
const CODE_BITS: u32 = 11;

/// Run both Rust and C++ decoders with given T-major codes and compare.
/// Returns (audio_rust, audio_cpp).
fn run_both_decoders(codes_tmajor: &[i32], t: usize) -> (Vec<f32>, Vec<f32>) {
    let exe = qwen_codec_exe();
    let gguf = codec_gguf();

    // Run through Rust decoder
    let decoder = qwen_tts_codec::CodecDecoder::load(&gguf).expect("load CodecDecoder");
    let audio_rust = decoder.decode(codes_tmajor, t);

    // Transpose to K-major for C++ decoder
    let codes_kmajor = transpose_codes(codes_tmajor, t, K);

    // Pack as .rvq file
    let tmp_dir = std::env::temp_dir().join("qwen-tts-codec-cross-check");
    let _ = fs::create_dir_all(&tmp_dir);
    let rvq_path = tmp_dir.join("test_codes.rvq");
    let wav_path = tmp_dir.join("test_codes.wav");

    let packed = pack_rvq(&codes_kmajor, CODE_BITS);
    fs::write(&rvq_path, &packed).expect("write .rvq file");

    // Run qwen-codec.exe to decode
    let output = Command::new(&exe)
        .arg("--model")
        .arg(&gguf)
        .arg("-i")
        .arg(&rvq_path)
        .arg("--format")
        .arg("wav32")
        .output()
        .expect("failed to run qwen-codec.exe");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("qwen-codec.exe failed:\n{}", stderr);
    }

    // Read C++ decoder output from WAV
    assert!(wav_path.exists(), "WAV not created at {}", wav_path.display());
    let audio_cpp = read_wav_f32_samples(&wav_path);

    // Cleanup
    let _ = fs::remove_file(&rvq_path);
    let _ = fs::remove_file(&wav_path);
    let _ = fs::remove_dir(&tmp_dir);

    (audio_rust, audio_cpp)
}

#[test]
fn compare_zero_codes() {
    // All zero codes: both decoders should produce identical audio
    let t = 2;
    let n_codes = t * K;
    let codes_tmajor = vec![0i32; n_codes];

    let (audio_rust, audio_cpp) = run_both_decoders(&codes_tmajor, t);

    assert_eq!(audio_rust.len(), audio_cpp.len(),
        "zero: audio length mismatch {} vs {}", audio_rust.len(), audio_cpp.len());

    let n = audio_rust.len();
    let max_diff = audio_rust.iter().zip(audio_cpp.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let avg_diff = audio_rust.iter().zip(audio_cpp.iter())
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>() / n as f64;

    println!("Zero codes: {} samples, max_diff={:.10}, avg_diff={:.10}",
        n, max_diff, avg_diff);

    // Zero codes test must match very closely (floating-point tolerance)
    assert!(max_diff < 1e-2,
        "Zero codes max_diff={max_diff} too large. Fundamental bug.");
}

#[test]
fn compare_with_cpp_small() {
    let gguf = codec_gguf();
    assert!(gguf.exists(), "GGUF not found at {}", gguf.display());

    // Generate synthetic codes: 3 frames, T-major [T, K]
    let t = 3;
    let n_codes = t * K;
    let codes_tmajor: Vec<i32> = (0..n_codes).map(|i| ((i * 7 + 13) % 2048) as i32).collect();

    let (audio_rust, audio_cpp) = run_both_decoders(&codes_tmajor, t);

    // Compare lengths - should be identical
    assert_eq!(
        audio_rust.len(),
        audio_cpp.len(),
        "Audio length mismatch"
    );

    // Compare sample by sample
    let n = audio_rust.len();
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut sum_diff = 0.0f64;
    let mut n_mismatch = 0usize;

    // Allow tolerance for floating point differences between
    // ggml computation (Q8_0 integer dot products, different accumulation
    // order) and pure Rust (f32 accumulators, dequantized weights).
    let tolerance = 1e-2;
    let mut first_failure = None;

    for i in 0..n {
        let diff = (audio_rust[i] - audio_cpp[i]).abs();
        sum_diff += diff as f64;

        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }

        if diff > tolerance {
            n_mismatch += 1;
            if first_failure.is_none() {
                first_failure = Some((i, audio_rust[i], audio_cpp[i], diff));
            }
        }
    }

    let avg_diff = sum_diff / n as f64;
    println!(
        "Comparison: max_diff={:.8} at idx={}, avg_diff={:.10}, mismatches={}/{} ({:.2}%)",
        max_diff,
        max_diff_idx,
        avg_diff,
        n_mismatch,
        n,
        n_mismatch as f64 / n as f64 * 100.0
    );

    if let Some((idx, r, c, d)) = first_failure {
        // Print context around first failure
        let start = idx.saturating_sub(5);
        let end = (idx + 5).min(n);
        println!("\nFirst mismatch at sample {idx}:");
        println!("  Rust[{idx}] = {r:.10}");
        println!("  C++[{idx}]  = {c:.10}");
        println!("  diff       = {d:.10}");
        println!("\nContext ({start}..{end}):");
        for j in start..end {
            println!(
                "  [{j:>4}] Rust={:.10}, C++={:.10}, diff={:.10}",
                audio_rust[j],
                audio_cpp[j],
                (audio_rust[j] - audio_cpp[j]).abs()
            );
        }
    }

    // Allow up to 10% mismatches at tolerance level 1e-2
    let max_allowed_mismatches = (n as f64 * 0.10) as usize;
    assert!(
        n_mismatch <= max_allowed_mismatches,
        "Too many mismatches: {n_mismatch} > {max_allowed_mismatches} (tolerance={tolerance})"
    );
    assert!(
        max_diff < 0.15,
        "Max diff {max_diff} exceeds 0.15 at sample {max_diff_idx}"
    );

}

#[test]
fn rvq_roundtrip() {
    // Verify our pack/unpack matches
    let codes: Vec<i32> = (0..48).map(|i| ((i * 131) % 2048) as i32).collect();
    let packed = pack_rvq(&codes, CODE_BITS);
    let unpacked = unpack_rvq(&packed, codes.len(), CODE_BITS);

    assert_eq!(codes, unpacked, "RVQ roundtrip failed");
    println!("RVQ roundtrip: {} codes -> {} bytes -> OK", codes.len(), packed.len());
}

#[test]
fn qwen_codec_load_selftest() {
    // Run qwen-codec.exe without -i to test loading
    let exe = qwen_codec_exe();
    let gguf = codec_gguf();

    let output = Command::new(&exe)
        .arg("--model")
        .arg(&gguf)
        .output()
        .expect("failed to run qwen-codec.exe");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "qwen-codec self-test failed:\n{stderr}");
    assert!(stderr.contains("Load self-test passed"), "unexpected output:\n{stderr}");
    println!("qwen-codec.exe self-test: OK");
}
