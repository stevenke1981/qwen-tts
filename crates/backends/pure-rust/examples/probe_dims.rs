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

    // Target tensor names
    let targets = [
        "talker.text_embd.weight",
        "talker.text_proj.fc1.weight",
        "talker.text_proj.fc1.bias",
        "talker.text_proj.fc2.weight",
        "talker.text_proj.fc2.bias",
        "talker.codec_embd.weight",
        "talker.codec_head.weight",
        "talker.output_norm.weight",
        "talker.blk.0.attn_q.weight",
        "talker.blk.0.attn_q_norm.weight",
        "talker.blk.0.attn_k_norm.weight",
        "code_pred.lm_head.0.weight",
        "code_pred.mtp_proj.weight",
        "code_pred.output_norm.weight",
        "code_pred.blk.0.attn_q.weight",
    ];

    for name in &targets {
        if let Some(info) = content.tensor_infos.get(*name) {
            let dims: Vec<usize> = info.shape.dims().to_vec();
            println!("  {name:45} {:?}", dims);
        } else {
            println!("  {name:45} NOT FOUND");
        }
    }
}
