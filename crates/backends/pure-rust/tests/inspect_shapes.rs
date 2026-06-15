//! Quick inspection of GGUF tensor shapes for hot-path weight optimization.
//! Run: cargo test -p qwen-tts-backend-pure-rust --test inspect_shapes -- --nocapture --ignored

#[test]
#[ignore]
fn inspect_predictor_gguf_shapes() {
    use candle_core::quantized::gguf_file;
    use std::fs::File;

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().parent().unwrap()
        .join("models")
        .join("qwen-talker-1.7b-base-Q8_0.gguf");

    let mut file = File::open(&path).expect("open model");
    let content = gguf_file::Content::read(&mut file).expect("read GGUF");

    let names = [
        "code_pred.mtp_proj.weight",
        "code_pred.lm_head.0.weight",
        "code_pred.codec_embd.0.weight",
    ];
    for name in &names {
        match content.tensor_infos.get(*name) {
            Some(info) => {
                let shape = info.shape.dims();
                println!("{:45} shape={shape:?}  dtype={:?}", name, info.ggml_dtype);
            }
            None => println!("{:45} NOT FOUND", name),
        }
    }
}
