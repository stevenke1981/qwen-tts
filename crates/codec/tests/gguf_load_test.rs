//! Integration test for the GGUF Q8_0 tensor loader.
//!
//! Opens the real codec GGUF file, verifies tensor count, key tensor names,
//! and dequantizes weight tensors to confirm correctness.
//!
//! The file uses mixed precision:
//!   type=0 (F32)  — bias, norm weights, scales, codebooks
//!   type=1 (F16)  — most convolution weights
//!   type=8 (Q8_0) — transformer attention + FFN weights

use qwen_tts_codec::gguf::GgufFile;
use std::path::PathBuf;

/// Helper: find the codec GGUF file relative to the workspace root.
fn codec_gguf_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/codec should be two levels below workspace");
    workspace.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
}

#[test]
fn load_codec_gguf_metadata() {
    let path = codec_gguf_path();
    assert!(path.exists(), "codec GGUF not found at '{}'", path.display());

    let gguf = GgufFile::open(&path).expect("should open codec GGUF");

    // The codec GGUF has 398 tensors
    assert_eq!(gguf.tensors.len(), 398, "expected 398 tensors");

    // Verify key tensor names and shapes
    // NOTE: dec.0 (DAC block 0) has NO snake activation — only conv1d layers.
    //       dec.5 has only snake (no conv_t / res blocks — it's the last bottleneck).
    //       dec.6 is a simple post-conv1d (no snake).
    //       No snake_post tensors exist in this model.
    let expected: &[(&str, u32, &[u64])] = &[
        // Pre-convolution
        ("tok_dec.pre_conv.bias",   0, &[1024]),
        ("tok_dec.pre_conv.weight", 1, &[3, 512, 1024]),
        // Upsample stages
        ("tok_dec.upsample.0.conv.bias",   0, &[1024]),
        ("tok_dec.upsample.0.conv.weight", 1, &[2, 1024, 1024]),
        ("tok_dec.upsample.0.dwconv.weight", 1, &[7, 1, 1024]),
        ("tok_dec.upsample.1.conv.bias",     0, &[1024]),
        ("tok_dec.upsample.1.conv.weight",   1, &[2, 1024, 1024]),
        // DAC decoder blocks — block 0 (conv only, no snake)
        ("tok_dec.dec.0.conv.bias",   0, &[1536]),
        ("tok_dec.dec.0.conv.weight", 1, &[7, 1024, 1536]),
        // Block 1 (stride 8 upsample: conv_t → snake → 3× ResUnit)
        ("tok_dec.dec.1.snake.alpha", 0, &[1536]),
        ("tok_dec.dec.1.snake.beta",  0, &[1536]),
        ("tok_dec.dec.1.conv_t.bias",   0, &[768]),
        ("tok_dec.dec.1.conv_t.weight", 1, &[16, 768, 1536]),
        // Blocks 2-4 (stride 5/4/3 upsample)
        ("tok_dec.dec.2.snake.alpha", 0, &[768]),
        ("tok_dec.dec.2.snake.beta",  0, &[768]),
        ("tok_dec.dec.2.conv_t.bias",   0, &[384]),
        ("tok_dec.dec.2.conv_t.weight", 1, &[10, 384, 768]),
        ("tok_dec.dec.3.snake.alpha", 0, &[384]),
        ("tok_dec.dec.3.snake.beta",  0, &[384]),
        ("tok_dec.dec.3.conv_t.bias",   0, &[192]),
        ("tok_dec.dec.3.conv_t.weight", 1, &[8, 192, 384]),
        ("tok_dec.dec.4.snake.alpha", 0, &[192]),
        ("tok_dec.dec.4.snake.beta",  0, &[192]),
        ("tok_dec.dec.4.conv_t.bias",   0, &[96]),
        ("tok_dec.dec.4.conv_t.weight", 1, &[6, 96, 192]),
        // Block 5 (bottleneck: snake only, no conv_t — stride=1)
        ("tok_dec.dec.5.snake.alpha", 0, &[96]),
        ("tok_dec.dec.5.snake.beta",  0, &[96]),
        // Post-conv (block 6)
        ("tok_dec.dec.6.conv.bias",   0, &[1]),
        ("tok_dec.dec.6.conv.weight", 1, &[7, 96]),
        // Transformer (Q8_0 quantized)
        ("tok_dec.pre_tfm.input_proj.weight", 8, &[1024, 512]),
        ("tok_dec.pre_tfm.output_proj.weight", 8, &[512, 1024]),
        ("tok_dec.pre_tfm.blk.0.attn_q.weight",   8, &[512, 1024]),
        ("tok_dec.pre_tfm.blk.0.attn_k.weight",   8, &[512, 1024]),
        ("tok_dec.pre_tfm.blk.0.attn_v.weight",   8, &[512, 1024]),
        ("tok_dec.pre_tfm.blk.0.attn_output.weight", 8, &[1024, 512]),
        // RVQ codebooks (F32)
        ("tok_dec.vq_first.0.codebook",  0, &[256, 2048]),
        ("tok_dec.vq_rest.0.codebook",   0, &[256, 2048]),
        ("tok_dec.vq_rest.14.codebook",  0, &[256, 2048]),
        // Encoder
        ("tok_enc.conv.0.bias",   0, &[64]),
        ("tok_enc.conv.0.weight", 1, &[7, 1, 64]),
        ("tok_enc.conv.12.bias",    0, &[1024]),
        ("tok_enc.conv.12.weight",  1, &[16, 512, 1024]),
    ];

    for (name, expected_type, expected_shape) in expected {
        let info = gguf.tensor(name).unwrap_or_else(|| panic!("missing tensor '{name}'"));
        assert_eq!(
            info.ggml_type, *expected_type,
            "tensor '{name}' type mismatch: expected {expected_type}, got {}",
            info.ggml_type
        );
        assert_eq!(
            info.shape, *expected_shape,
            "tensor '{name}' shape mismatch: expected {expected_shape:?}, got {:?}",
            info.shape
        );
    }
}

#[test]
fn load_f16_conv_weight() {
    let path = codec_gguf_path();
    let mut gguf = GgufFile::open(&path).expect("should open codec GGUF");

    // Load pre_conv weight — shape [3, 512, 1024] = 1,572,864 elements
    let weight = gguf
        .read_tensor_f32("tok_dec.pre_conv.weight")
        .expect("should load pre_conv.weight");

    assert_eq!(weight.len(), 3 * 512 * 1024, "element count mismatch");

    // Check that values look valid (not all zero, finite)
    let has_nonzero = weight.iter().any(|&v| v.abs() > 0.001);
    assert!(has_nonzero, "all weights are zero — dequant likely wrong");

    let has_finite = weight.iter().all(|v| v.is_finite());
    assert!(has_finite, "some weights are non-finite");

    let mean = weight.iter().sum::<f32>() / weight.len() as f32;
    println!("tok_dec.pre_conv.weight: {} elems, mean={mean:.6}", weight.len());
}

#[test]
fn load_q8_0_transformer_weight() {
    let path = codec_gguf_path();
    let mut gguf = GgufFile::open(&path).expect("should open codec GGUF");

    // Load a Q8_0 quantized transformer weight
    let weight = gguf
        .read_tensor_f32("tok_dec.pre_tfm.input_proj.weight")
        .expect("should load input_proj.weight");

    assert_eq!(weight.len(), 1024 * 512, "element count mismatch");

    let has_nonzero = weight.iter().any(|&v| v.abs() > 0.001);
    assert!(has_nonzero, "all Q8_0 weights are zero — dequant likely wrong");

    let mean = weight.iter().sum::<f32>() / weight.len() as f32;
    println!("tok_dec.pre_tfm.input_proj.weight: {} elems, mean={mean:.6}", weight.len());
}

#[test]
fn load_f32_codebook() {
    let path = codec_gguf_path();
    let mut gguf = GgufFile::open(&path).expect("should open codec GGUF");

    let cb = gguf
        .read_tensor_f32("tok_dec.vq_first.0.codebook")
        .expect("should load codebook");

    assert_eq!(cb.len(), 256 * 2048, "codebook size mismatch");

    let has_nonzero = cb.iter().any(|&v| v.abs() > 0.001);
    assert!(has_nonzero, "codebook all zero");

    let mean = cb.iter().sum::<f32>() / cb.len() as f32;
    println!("tok_dec.vq_first.0.codebook: {} elems, mean={mean:.6}", cb.len());
}


