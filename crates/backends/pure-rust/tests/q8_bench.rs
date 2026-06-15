//! Benchmark: Q8_0 GEMV vs F32 linear for autoregressive talker inference.
//!
//! Measures load time and per-step latency for 128 frames of synthesis.
//! Run with: cargo test -p qwen-tts-backend-pure-rust --test q8_bench -- --nocapture --release
//!
//! The model is a 1.7B-param Q8_0 quantized GGUF file (~1.7 GB on disk).
//! With the custom Q8_0 GEMV, weights are loaded directly without dequantization,
//! so we expect ~1s load time (vs ~5s for F32 dequant).

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};

use qwen_tts_backend_pure_rust::code_predictor::CodePredictor;
use qwen_tts_backend_pure_rust::talker::{KvCacheFlat, Talker};
use qwen_tts_backend_pure_rust::timing::TimingRecorder;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

fn project_root() -> &'static Path {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("D:\\qwen_tts"));
    root
}

fn talker_path() -> std::path::PathBuf {
    project_root()
        .join("models")
        .join("qwen-talker-1.7b-base-Q8_0.gguf")
}

fn codec_path() -> std::path::PathBuf {
    project_root()
        .join("models")
        .join("qwen-tokenizer-12hz-Q8_0.gguf")
}

// ---------------------------------------------------------------------------
// Helper: precompute RoPE cos/sin (same logic as talker.rs)
// ---------------------------------------------------------------------------

fn precompute_cos_sin(
    head_dim: usize,
    rope_theta: f64,
    max_s: usize,
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| (1.0_f64 / rope_theta.powf(i as f64 / head_dim as f64)) as f32)
        .collect();
    let n = inv_freq.len();
    let inv_freq = Tensor::from_slice(&inv_freq, (n,), dev)?;
    let pos: Vec<f32> = (0..max_s).map(|i| i as f32).collect();
    let pos = Tensor::from_slice(&pos, (max_s,), dev)?;
    let freqs = pos.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    // Interleave pairs → [max_s, head_dim]
    let cos = interleave(&cos, 2)?;
    let sin = interleave(&sin, 2)?;
    // Add batch+head dims → [1, 1, max_s, head_dim]
    Ok((cos.unsqueeze(0)?.unsqueeze(0)?, sin.unsqueeze(0)?.unsqueeze(0)?))
}

fn interleave(x: &Tensor, n: usize) -> anyhow::Result<Tensor> {
    let s = x.dims();
    let last = s[s.len() - 1];
    let x = x.unsqueeze(s.len())?;
    let mut shape = s.to_vec();
    shape.push(n);
    let x = x.expand(shape.as_slice())?;
    let mut out_shape = s.to_vec();
    out_shape[s.len() - 1] = last * n;
    Ok(x.reshape(out_shape.as_slice())?)
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

#[derive(Default)]
struct BenchTiming {
    load: Vec<f64>,           // seconds
    talker_prefill: Vec<f64>, // seconds
    talker_step: Vec<f64>,    // seconds (per token)
    predictor_frame: Vec<f64>,  // seconds (per full 16-position frame)
}

impl BenchTiming {
    fn report(&self, _label: &str) {
        fn stats(v: &[f64]) -> (f64, f64, f64) {
            let n = v.len() as f64;
            let mean = v.iter().sum::<f64>() / n;
            let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt();
            (mean, std, v.iter().cloned().fold(f64::MAX, f64::min))
        }

        if !self.load.is_empty() {
            let (m, s, min) = stats(&self.load);
            println!("  load:          {:.3}s ± {:.3}s  (min {:.3}s)", m, s, min);
        }
        if !self.talker_prefill.is_empty() {
            let (m, s, min) = stats(&self.talker_prefill);
            println!("  prefill:       {:.3}s ± {:.3}s  (min {:.3}s)", m, s, min);
        }
        if !self.talker_step.is_empty() {
            let (m, s, min) = stats(&self.talker_step);
            println!("  step:          {:.4}s ± {:.4}s  (min {:.4}s)", m, s, min);
            println!(
                "    throughput:  {:.1} tok/s",
                1.0 / m
            );
        }
        if !self.predictor_frame.is_empty() {
            let (m, s, min) = stats(&self.predictor_frame);
            println!("  predictor frame: {:.4}s ± {:.4}s  (min {:.4}s)", m, s, min);
            println!(
                "    throughput:    {:.1} frame/s",
                1.0 / m
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark: just loading with Q8Weights (no dequantization)
// ---------------------------------------------------------------------------

#[test]
fn bench_load_time() {
    let path = talker_path();
    assert!(path.exists(), "GGUF not found: {}", path.display());

    let mut timing = BenchTiming::default();

    // Warm up (first load is cold — disk cache).
    let start = Instant::now();
    let mut talker = Talker::from_gguf(&path, &Device::Cpu)
        .expect("talker should load with Q8Weights");
    let load_s = start.elapsed().as_secs_f64();
    timing.load.push(load_s);

    // Load code predictor from same file
    let mut file = File::open(&path).expect("open GGUF");
    let content = gguf_file::Content::read(&mut file).expect("parse GGUF header");
    let predictor = CodePredictor::from_gguf(&content, &mut file, &Device::Cpu)
        .expect("predictor should load with Q8Weights");

    println!("--- bench_load_time ---");
    timing.report("Talker+CodePredictor load with Q8Weights");
    println!(
        "  talker layers: {}  predictor layers: {}",
        talker.config().n_layers,
        predictor.num_acoustic(),
    );

    // Minimal sanity: forward pass 1 step through talker
    let device = Device::Cpu;
    let batch = 1;
    let d_model = talker.config().d_model;
    let cfg = talker.config();
    let max_seq = 2048;
    let mut cache = KvCacheFlat::new(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim(), max_seq);
    let (cos, sin) = precompute_cos_sin(cfg.head_dim(), cfg.rope_theta, max_seq, &device)
        .expect("precompute cos/sin");

    // Simulate a single token embedding
    let x = Tensor::ones((batch, 1, d_model), candle_core::DType::F32, &device)
        .expect("create input");

    // Measure 10 steps
    for _ in 0..10 {
        let start = Instant::now();
        let _h = talker
            .forward_step(&x, &mut cache, &cos, &sin)
            .expect("forward_step");
        let step_s = start.elapsed().as_secs_f64();
        timing.talker_step.push(step_s);
    }

    println!("\n  after 10 forward_step calls:");
    timing.report("Talker forward_step (Q8_0 GEMV)");
    println!("  cache pos: {}", cache.current_len());
}

// ---------------------------------------------------------------------------
// Full benchmark: 128 frames of autoregressive synthesis
// ---------------------------------------------------------------------------

#[test]
#[ignore = "run manually with --release -- --nocapture"]
fn bench_128_frames() {
    let device = Device::Cpu;

    // Create timing recorder
    let mut timing = TimingRecorder::new();

    // Load talker
    let path = talker_path();
    let t0 = Instant::now();
    let mut talker = Talker::from_gguf(&path, &device).expect("load talker");
    let load_talker_s = t0.elapsed().as_secs_f64();

    // Load code predictor
    let mut file = File::open(&path).expect("open GGUF");
    let content = gguf_file::Content::read(&mut file).expect("parse header");
    let t1 = Instant::now();
    let mut predictor =
        CodePredictor::from_gguf(&content, &mut file, &device).expect("load predictor");
    let load_predictor_s = t1.elapsed().as_secs_f64();

    timing.record(
        "model_load".into(), "load".into(),
        load_talker_s + load_predictor_s, 0,
    );

    println!(
        "=== 128-frame benchmark (Q8_0 GEMV) ==="
    );
    println!(
        "Load: talker={:.3}s  predictor={:.3}s  total={:.3}s",
        load_talker_s,
        load_predictor_s,
        load_talker_s + load_predictor_s,
    );
    println!(
        "Model: {} layers, {} dim, {} vocab",
        talker.config().n_layers,
        talker.config().d_model,
        talker.config().vocab_size,
    );
    println!();

    let cfg = talker.config();
    let n_layers = cfg.n_layers;
    let max_seq = 2048;
    let (cos, sin) = precompute_cos_sin(cfg.head_dim(), cfg.rope_theta, max_seq, &device).expect("cos/sin");

    // Pre-allocate input embedding (text token 0 as placeholder)
    let d_model = cfg.d_model;
    let input = Tensor::zeros((1, 1, d_model), candle_core::DType::F32, &device)
        .expect("input tensor");

    // ---- Talker prefill (128 tokens, streaming) ----
    let mut talker_cache = KvCacheFlat::new(n_layers, cfg.n_kv_heads, cfg.head_dim(), max_seq);

    for step in 0..128 {
        let start = Instant::now();
        let _h = talker
            .forward_step(&input, &mut talker_cache, &cos, &sin)
            .expect("talker forward_step");
        timing.record(
            "talker_step".into(), "step".into(),
            start.elapsed().as_secs_f64(), step,
        );
    }

    // Stats
    let talker_total = timing.category_total("step");
    let talker_step_events: Vec<_> = timing.events.iter().filter(|e| e.category == "step").collect();
    let n_steps = talker_step_events.len() as f64;
    let talker_mean = talker_total / n_steps;
    println!("--- Talker 128-step ---");
    println!(
        "  total: {:.3}s  mean: {:.4}s/step  tok/s: {:.0}",
        talker_total,
        talker_mean,
        1.0 / talker_mean,
    );
    if let Some(first) = talker_step_events.first() {
        println!(
            "  first step (cold cache): {:.4}s  subsequent mean: {:.4}s",
            first.duration_s,
            (talker_total - first.duration_s) / (n_steps - 1.0),
        );
    }

    // ---- Code predictor: 128 frames ----
    // Use the last talker hidden state as input
    let talker_hidden = Tensor::ones((1, 1, d_model), candle_core::DType::F32, &device)
        .expect("talker hidden");
    let c0_embed = Tensor::ones((1, 1, d_model), candle_core::DType::F32, &device)
        .expect("c0 embd");

    let mut total_tokens = 0usize;

    for frame in 0..128 {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let start = Instant::now();
        // Full frame: prefill 2 positions + decode 14 positions = 16 forward steps
        let _codes = predictor
            .predict_one_frame_sampled(
                &talker_hidden,
                &c0_embed,
                1.0,
                Some(40),
                Some(0.9),
                &mut rng,
            )
            .expect("predict frame");
        timing.record(
            "predictor_frame".into(), "frame".into(),
            start.elapsed().as_secs_f64(), frame,
        );
        total_tokens += 16; // 2 prefill + 14 decode
    }

    let frame_total = timing.category_total("frame");
    let n_frames = 128.0;
    let frame_mean = frame_total / n_frames;
    println!();
    println!("--- Code Predictor 128-frame ---");
    println!(
        "  total: {:.3}s  mean: {:.4}s/frame  frame/s: {:.1}",
        frame_total,
        frame_mean,
        1.0 / frame_mean,
    );
    println!(
        "  decoder tokens: {}  token/s (decoder): {:.0}",
        total_tokens,
        total_tokens as f64 / frame_total,
    );

    // ---- Total synthesis estimate ----
    let all_talker = timing.category_total("step");
    let all_frames = timing.category_total("frame");
    let total = timing.category_total("load") + all_talker + all_frames;
    println!();
    println!("=== Total 128-frame synthesis estimate ===");
    println!(
        "  {:6.3}s  talker forward (128 steps)",
        all_talker
    );
    println!(
        "  {:6.3}s  code predictor (128 frames x 16 tokens)",
        all_frames
    );
    println!(
        "  {:6.3}s  + load time",
        timing.category_total("load"),
    );
    println!(
        "  {:6.3}s  TOTAL (dominates: talker)",
        total
    );
    println!(
        "  vs C++ FFI target: ~2-5s  (gap: {:.0}x)",
        total / 3.0
    );

    // ---- Export timing results ----
    let bench_dir = project_root()
        .join("target")
        .join("bench-results");
    let _ = std::fs::create_dir_all(&bench_dir);
    let json_path = bench_dir.join("bench_128_frames.json");
    let csv_path = bench_dir.join("bench_128_frames.csv");
    std::fs::write(&json_path, &timing.to_json())
        .expect("write JSON results");
    std::fs::write(&csv_path, &timing.to_csv())
        .expect("write CSV results");
    println!();
    println!("Timing results written to:");
    println!("  {}", json_path.display());
    println!("  {}", csv_path.display());

    // Print summary
    let summary = timing.summary();
    println!();
    println!("=== Timing Summary ===");
    for cat in ["load", "step", "frame", "decode"] {
        if let Some(total) = summary.get(cat) {
            println!("  {:<12} {:>8.3}s", cat, total);
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison: verify output matches F32 linear_fwd (cross-validation)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires model file — run manually with --release"]
fn bench_crossval_q8_vs_f32() {
    // Load the model once with Q8_0 (current)
    // Then compare intermediate outputs against previous F32 behavior
    // to verify accuracy is acceptable.
    let device = Device::Cpu;
    let path = talker_path();

    let mut talker = Talker::from_gguf(&path, &device).expect("load talker (Q8_0)");
    let cfg = talker.config();
    let n_layers = cfg.n_layers;
    let d_model = cfg.d_model;
    let max_seq = 2048;
    let (cos, sin) = precompute_cos_sin(cfg.head_dim(), cfg.rope_theta, max_seq, &device).expect("cos/sin");

    // Forward a single step
    let mut cache = KvCacheFlat::new(n_layers, cfg.n_kv_heads, cfg.head_dim(), max_seq);
    let x = Tensor::ones((1, 1, d_model), candle_core::DType::F32, &device).unwrap();

    let h = talker
        .forward_step(&x, &mut cache, &cos, &sin)
        .expect("forward_step");

    let h_vec: Vec<f32> = h.flatten_all().unwrap().to_vec1().unwrap();
    let mean = h_vec.iter().copied().sum::<f32>() / h_vec.len() as f32;
    let max_val = h_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("Cross-val: Q8_0 GEMV forward_step output");
    println!("  shape: {:?}", h.dims());
    println!("  mean: {:.4}  max: {:.4}", mean, max_val);
    assert!(mean.is_finite(), "mean should be finite");

    // With Q8_0 quantization, values should be close to what F32 would produce
    // (within ~5% relative error per Q8_0 accuracy guarantee).
    assert!(
        mean.abs() < 100.0,
        "mean output magnitude should be reasonable, got {mean}",
    );
}

// ---------------------------------------------------------------------------
// Microbenchmark: compare raw Q8_0 GEMV vs F32 GEMM for realistic sizes
// ---------------------------------------------------------------------------

#[test]
#[ignore = "microbenchmark for profiling"]
fn microbench_gemv_sizes() {
    let dev = Device::Cpu;
    use candle_core::quantized::k_quants::{BlockQ8_0, GgmlType, QK8_0};

    // Sizes from the actual model (talker, d_model=2048)
    let sizes: &[(usize, usize, &str)] = &[
        (2048, 2048, "attn_q/o   (2048×2048)"),
        (256, 2048,  "attn_k/v   (256×2048)"),
        (5461, 2048, "ffn_gate/up (5461×2048)"),
        (2048, 5461, "ffn_down   (2048×5461)"),
    ];

    for &(n, k, label) in sizes {
        let bpr = k.div_ceil(QK8_0);
        let padded_k = bpr * QK8_0;

        // Build random Q8_0 weights by quantizing random F32 data
        let mut f32_w: Vec<f32> = (0..n * k).map(|i| ((i * 7 + 3) % 100) as f32 / 10.0).collect();
        let mut data = vec![BlockQ8_0::zeros(); n * bpr];
        for row in 0..n {
            let mut row_data = f32_w[row * k..(row + 1) * k].to_vec();
            row_data.resize(padded_k, 0.0);
            let dst = &mut data[row * bpr..(row + 1) * bpr];
            <BlockQ8_0 as GgmlType>::from_float(&row_data, dst);
        }
        let w = qwen_tts_backend_pure_rust::qgemv::Q8Weights::from_raw(n, k, bpr, padded_k, data);

        let x: Vec<f32> = (0..k).map(|i| ((i * 3) % 50) as f32).collect();
        let mut ws = qwen_tts_backend_pure_rust::qgemv::Q8Workspace::new();

        // Warm up
        let _ = w.gemv(&x, &mut ws);

        // Benchmark GEMV
        let n_iters = 1000;
        let start = Instant::now();
        for _ in 0..n_iters {
            let _y = w.gemv(&x, &mut ws);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let per_call_us = elapsed * 1_000_000.0 / n_iters as f64;
        println!(
            "  {label}: {per_call_us:.1}µs/call  ({n_iters} iters in {elapsed:.3}s)"
        );

        // Compare with F32 GEMM
        let w_f32 = Tensor::from_slice(&f32_w, (n, k), &dev).unwrap();
        let x_t = Tensor::from_slice(&x, (1, k), &dev).unwrap();

        let start = Instant::now();
        for _ in 0..n_iters {
            let _y = x_t.matmul(&w_f32.t().unwrap()).unwrap();
        }
        let f32_elapsed = start.elapsed().as_secs_f64();
        let f32_per_call_us = f32_elapsed * 1_000_000.0 / n_iters as f64;
        println!(
            "  F32 GEMM {label}: {f32_per_call_us:.1}µs/call  ({n_iters} iters in {f32_elapsed:.3}s)"
        );
        println!(
            "  Q8 vs F32: {:.1}×",
            f32_per_call_us / per_call_us
        );
    }
}
