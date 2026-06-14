//! Standalone codec decoder CLI: decode .rvq -> WAV, model info, benchmarks.
//!
//! Compatible with qwen-codec.exe's decode interface:
//!   qwen-codec --model <gguf> -i <input.rvq>

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "qwen-codec", about = "Qwen3-TTS codec decoder (Rust native)")]
enum Cli {
    /// Decode RVQ codes to WAV audio
    Decode {
        /// Path to codec GGUF model
        #[arg(long, short)]
        model: PathBuf,

        /// Input file (.rvq for decode)
        #[arg(long, short)]
        input: PathBuf,

        /// Output WAV path (default: <input>.wav)
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
    /// Show codec model information
    Info {
        /// Path to codec GGUF model
        #[arg(long, short)]
        model: PathBuf,
    },
    /// Benchmark decode performance
    Bench {
        /// Path to codec GGUF model
        #[arg(long, short)]
        model: PathBuf,

        /// Number of frames to decode (default: 100)
        #[arg(long, short, default_value = "100")]
        frames: usize,

        /// Number of benchmark iterations (default: 5)
        #[arg(long, short = 'n', default_value = "5")]
        iterations: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli {
        Cli::Decode { model, input, output } => cmd_decode(&model, &input, output.as_ref()),
        Cli::Info { model } => cmd_info(&model),
        Cli::Bench {
            model,
            frames,
            iterations,
        } => cmd_bench(&model, frames, iterations),
    }
}

fn cmd_decode(model: &PathBuf, input: &PathBuf, output: Option<&PathBuf>) {
    // Read .rvq file
    let rvq_data = std::fs::read(input).unwrap_or_else(|e| {
        eprintln!("Error: cannot read {}: {}", input.display(), e);
        std::process::exit(1);
    });

    // Determine number of codes: .rvq stores [K, T] packed at 11 bits/code
    let total_bits = rvq_data.len() as u64 * 8;
    let n_codes = (total_bits / 11) as usize;
    let t = n_codes / 16;
    if n_codes % 16 != 0 {
        eprintln!(
            "Warning: {} codes not divisible by 16 (K), rounding to {} frames",
            n_codes, t
        );
    }

    // Unpack codes (K-major layout from file)
    let codes_k_major = qwen_tts_codec::unpack_rvq(&rvq_data, t * 16);

    // Transpose from K-major [K, T] to T-major [T, K] for the decoder
    let mut codes_t_major = vec![0i32; t * 16];
    for ti in 0..t {
        for ki in 0..16 {
            codes_t_major[ti * 16 + ki] = codes_k_major[ki * t + ti];
        }
    }

    // Load model and decode
    let decoder = qwen_tts_codec::pipeline::CodecDecoder::load(model).unwrap_or_else(|e| {
        eprintln!("Error loading model: {e}");
        std::process::exit(1);
    });
    let audio = decoder.decode(&codes_t_major, t);

    // Determine output path
    let out_path = output.map_or_else(|| input.with_extension("wav"), |p| p.to_path_buf());

    // Write WAV (24 kHz, mono, f32)
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&out_path, spec).unwrap_or_else(|e| {
        eprintln!("Error creating WAV: {e}");
        std::process::exit(1);
    });
    for &sample in &audio {
        writer.write_sample(sample.clamp(-1.0, 1.0)).unwrap_or_else(|e| {
            eprintln!("Error writing sample: {e}");
            std::process::exit(1);
        });
    }
    writer.finalize().unwrap_or_else(|e| {
        eprintln!("Error finalizing WAV: {e}");
        std::process::exit(1);
    });

    println!(
        "Decoded {} frames -> {} samples @ 24kHz -> {}",
        t,
        audio.len(),
        out_path.display()
    );
}

fn cmd_info(model: &PathBuf) {
    use qwen_tts_core::{GgufMetadata, GgufProbe};

    let probe = GgufProbe::open(model).unwrap_or_else(|e| {
        eprintln!("Error opening GGUF: {e}");
        std::process::exit(1);
    });

    println!("=== Codec Model Info ===");
    println!("Path: {}", model.display());
    println!("Format version: {}", probe.version);
    println!("Tensor count: {}", probe.tensor_count);
    println!("Metadata keys: {}", probe.metadata_kv_count);
    println!();

    // Read selected metadata keys
    let meta = GgufMetadata::read_selected(model, &[
        "general.architecture",
        "general.description",
        "tokenizer.hop_length",
        "tokenizer.sample_rate",
        "tokenizer.num_codebooks",
    ])
    .unwrap_or_else(|e| {
        eprintln!("Warning: could not read metadata: {e}");
        GgufMetadata::default()
    });

    if let Some(arch) = meta.get_str("general.architecture") {
        println!("Architecture: {arch}");
    }
    if let Some(desc) = meta.get_str("general.description") {
        println!("Description: {desc}");
    }
    if let Some(hops) = meta.get_u32("tokenizer.hop_length") {
        println!("Hop length: {hops}");
    }
    if let Some(sr) = meta.get_u32("tokenizer.sample_rate") {
        println!("Sample rate: {sr}");
    }
    if let Some(cbs) = meta.get_u32("tokenizer.num_codebooks") {
        println!("Codebooks: {cbs}");
    }
}

fn cmd_bench(model: &PathBuf, frames: usize, iterations: usize) {
    use std::time::Instant;

    // Load decoder once
    let decoder = qwen_tts_codec::pipeline::CodecDecoder::load(model).unwrap_or_else(|e| {
        eprintln!("Error loading model: {e}");
        std::process::exit(1);
    });

    // Generate zero codes for benchmarking (deterministic)
    let codes = vec![0i32; frames * 16];

    println!(
        "Benchmarking decode: {} frames, {} iterations",
        frames, iterations
    );
    println!();

    let mut total_duration = std::time::Duration::ZERO;

    for i in 0..iterations {
        let start = Instant::now();
        let audio = decoder.decode(&codes, frames);
        let duration = start.elapsed();

        total_duration += duration;
        let audio_len = audio.len();
        let audio_secs = audio_len as f64 / 24000.0;
        let realtime_factor = audio_secs / duration.as_secs_f64();

        println!(
            "  Iteration {}: {:?} ({} samples, {:.2}s audio, {:.2}x realtime)",
            i + 1,
            duration,
            audio_len,
            audio_secs,
            realtime_factor
        );
    }

    let avg = total_duration / iterations as u32;
    let avg_audio_secs = (frames as f64 * 1920.0) / 24000.0;
    let avg_rtf = avg_audio_secs / avg.as_secs_f64();

    println!();
    println!(
        "Average: {:?} ({:.2}x realtime, {:.2}s audio in {:?})",
        avg, avg_rtf, avg_audio_secs, avg
    );
}
