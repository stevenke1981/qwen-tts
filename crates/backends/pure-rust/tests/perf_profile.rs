//! Per-layer profiling: measure how long each section of forward_step takes.
//!
//! Run: cargo test -p qwen-tts-backend-pure-rust --test perf_profile -- --release --nocapture

use std::path::Path;
use std::time::Instant;

use candle_core::{Device, Tensor};

use qwen_tts_backend_pure_rust::talker::{KvCacheFlat, Talker};

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
    let cos = interleave(&cos, 2)?;
    let sin = interleave(&sin, 2)?;
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

#[test]
fn profile_cache_growth() {
    let path = talker_path();
    assert!(path.exists(), "GGUF not found: {}", path.display());

    let mut talker = Talker::from_gguf(&path, &Device::Cpu).expect("load talker");
    let device = Device::Cpu;
    let cfg = talker.config();
    let d_model = cfg.d_model;
    let n_layers = cfg.n_layers;
    let max_seq = 2048;
    let (cos, sin) = precompute_cos_sin(cfg.head_dim(), cfg.rope_theta, max_seq, &device).expect("cos/sin");

    let x = Tensor::ones((1, 1, d_model), candle_core::DType::F32, &device).expect("input");

    // 128 steps measuring cache growth
    let mut cache = KvCacheFlat::new(n_layers, cfg.n_kv_heads, cfg.head_dim(), max_seq);
    let mut times = Vec::with_capacity(128);
    for step in 0..128 {
        let start = Instant::now();
        let _h = talker.forward_step(&x, &mut cache, &cos, &sin).expect("step");
        times.push(start.elapsed().as_secs_f64());
    }

    // Print first 10 steps
    for (i, &t) in times.iter().enumerate().take(10) {
        println!("  step {:3}: {:.4}s  (cache_len={})", i, t, i + 1);
    }

    // Bucket analysis: average time per 16-step bucket
    println!("\n  --- cache-growth buckets ---");
    for bucket in 0..8 {
        let start = bucket * 16;
        let end = start + 16;
        let bucket_times: Vec<f64> = times[start..end.min(128)].to_vec();
        let avg = bucket_times.iter().sum::<f64>() / bucket_times.len() as f64;
        println!("  steps {:3}-{:3} (cache {}-{}): avg {:.4}s",
            start, end - 1, start + 1, end, avg);
    }

    // Estimate: base cost + per-cache-position overhead
    // Use first 8 steps (small cache) for base estimation
    let base = times[..8].iter().sum::<f64>() / 8.0;
    // Use last 8 steps for full cache estimation
    let full = times[120..].iter().sum::<f64>() / 8.0;

    println!("\n  --- summary ---");
    println!("  base cost (cache≈4):   {:.4}s/step", base);
    println!("  full cost (cache≈124): {:.4}s/step", full);
    println!("  growth per 120 cache positions: {:.4}s", full - base);
    println!("  avg per cache position: {:.6}s", (full - base) / 120.0);

    // Projected total for 128-step synthesis
    let total_128: f64 = times.iter().sum();
    println!("  actual total 128 steps: {:.3}s", total_128);
    println!("  avg across all steps:   {:.4}s/step", total_128 / 128.0);
}
