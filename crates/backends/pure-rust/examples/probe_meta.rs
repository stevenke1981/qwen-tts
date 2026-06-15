use std::fs::File;
use std::path::Path;
use candle_core::quantized::gguf_file;

fn main() {
    let cargo_manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = cargo_manifest.parent().unwrap().parent().unwrap().parent().unwrap();
    let path = root.join("models").join("qwen-talker-1.7b-base-Q8_0.gguf");
    println!("Talker GGUF: {}", path.display());

    let mut file = File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut file).unwrap();

    println!("\n=== METADATA ===");
    let mut keys: Vec<&String> = content.metadata.keys().collect();
    keys.sort();
    for (i, key) in keys.iter().enumerate() {
        let val = &content.metadata[*key];
        let display = format!("{:?}", val);
        println!("  [{i:3}] {key} = {display}");
    }

    println!("\n=== TENSOR COUNT ===");
    println!("Total tensors: {}", content.tensor_infos.len());
}
