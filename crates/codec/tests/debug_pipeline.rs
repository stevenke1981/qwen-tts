//! Debug the pipeline by printing intermediate values at each stage.
//! Run with: cargo test -p qwen-tts-codec --test debug_pipeline -- --nocapture

use std::path::PathBuf;

fn codec_gguf_path() -> PathBuf {
    let md = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ws = md.parent().and_then(|p| p.parent()).unwrap();
    ws.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
}

#[test]
fn debug_pipeline_stages() {
    use qwen_tts_codec::pipeline::*;
    use qwen_tts_codec::CodecDecoder;

    let path = codec_gguf_path();
    let decoder = CodecDecoder::load(&path).expect("load");

    // Zero codes
    let t = 2;
    let codes = vec![0i32; t * NUM_CODEBOOKS];

    // Stage 1: quantizer
    let h1 = decoder.quantizer.decode(&codes, t);
    println!("=== Quantizer output [512, 2] (first 20 values) ===");
    for i in 0..20.min(h1.len()) {
        println!("  h1[{i}] = {:.10}", h1[i]);
    }
    // Show C-first layout: channel 0, time 0 and channel 0, time 1
    println!("  h1[0*2+0] (ch0,t0) = {:.10}", h1[0]);
    println!("  h1[0*2+1] (ch0,t1) = {:.10}", h1[1]);
    println!("  h1[1*2+0] (ch1,t0) = {:.10}", h1[2]);
    println!("  h1[1*2+1] (ch1,t1) = {:.10}", h1[3]);

    // Stage 2: pre_conv
    let h2 = decoder.pre_conv.forward(&h1, t);
    println!("\n=== Pre-conv output [1024, 2] (first 20 values) ===");
    for i in 0..20.min(h2.len()) {
        println!("  h2[{i}] = {:.10}", h2[i]);
    }
    println!("  Mean of h2: {:.10}", h2.iter().sum::<f32>() / h2.len() as f32);

    // Stage 3: transformer
    let h3 = decoder.transformer.forward(&h2, t);
    println!("\n=== Transformer output [1024, 2] (first 20 values) ===");
    for i in 0..20.min(h3.len()) {
        println!("  h3[{i}] = {:.10}", h3[i]);
    }
    println!("  Mean of h3: {:.10}", h3.iter().sum::<f32>() / h3.len() as f32);

    // Stage 4: upsample
    let h4 = decoder.upsample.forward(&h3, t);
    println!("\n=== Upsample output [1024, 8] (first 20 values) ===");
    for i in 0..20.min(h4.len()) {
        println!("  h4[{i}] = {:.10}", h4[i]);
    }
    println!("  Mean of h4: {:.10}", h4.iter().sum::<f32>() / h4.len() as f32);

    // Stage 5: DAC
    let t_upsampled = t * 4;
    let h5 = decoder.dac.forward(&h4, t_upsampled);
    println!("\n=== DAC output [1, 3840] (first 20 values) ===");
    for i in 0..20.min(h5.len()) {
        println!("  h5[{i}] = {:.10}", h5[i]);
    }
    println!("  Mean of h5: {:.10}", h5.iter().sum::<f32>() / h5.len() as f32);
    println!(
        "  Range: [{:.10}, {:.10}]",
        h5.iter().cloned().fold(f32::NAN, f32::min),
        h5.iter().cloned().fold(f32::NAN, f32::max)
    );

    // Full decode
    let audio = decoder.decode(&codes, t);
    println!("\n=== Final audio (first 20 of {} samples) ===", audio.len());
    for i in 0..20.min(audio.len()) {
        println!("  audio[{i}] = {:.10}", audio[i]);
    }
    println!(
        "  Range: [{:.10}, {:.10}]",
        audio.iter().cloned().fold(f32::NAN, f32::min),
        audio.iter().cloned().fold(f32::NAN, f32::max)
    );
}
