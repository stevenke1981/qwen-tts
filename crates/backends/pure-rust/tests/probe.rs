//! Quick GGUF tensor name probe — run with:
//! cargo test -p qwen-tts-backend-pure-rust --test probe -- --nocapture

use std::fs::File;
use candle_core::quantized::gguf_file;

#[test]
fn probe_talker_tensor_names() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .parent().unwrap()
        .join("models")
        .join("qwen-talker-1.7b-base-Q8_0.gguf");
    println!("Talker GGUF: {}", path.display());

    let mut file = File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut file).unwrap();

    // Show all names that DON'T start with "code_pred"
    let mut non_cp: Vec<&String> = content.tensor_infos.keys()
        .filter(|k| !k.starts_with("code_pred"))
        .collect();
    non_cp.sort();
    println!("\nNon-code_pred tensors ({} total):", non_cp.len());
    for (i, name) in non_cp.iter().enumerate() {
        println!("  [{i:3}] {name}");
    }
}

#[test]
fn probe_codec_tensor_names() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .parent().unwrap()
        .join("models")
        .join("qwen-tokenizer-12hz-Q8_0.gguf");
    println!("\nCodec GGUF: {}", path.display());

    let mut file = File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut file).unwrap();

    // Quantizer tensors
    let mut quant: Vec<&String> = content.tensor_infos.keys()
        .filter(|k| k.contains("vq_levels") || k.contains("codebook") || k.contains("quantizer"))
        .collect();
    quant.sort();
    println!("\nQuantizer-related tensors ({} total):", quant.len());
    for (i, name) in quant.iter().enumerate() {
        println!("  [{i:3}] {name}");
    }

    if quant.is_empty() {
        // Show first 20 tensors instead
        println!("\nFirst 20 tensors:");
        let mut names: Vec<&String> = content.tensor_infos.keys().collect();
        names.sort();
        for (i, name) in names.iter().take(20).enumerate() {
            println!("  [{i:3}] {name}");
        }
    }
}
