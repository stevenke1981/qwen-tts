//! Dump all tensor names from the codec GGUF file for debugging.
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

fn open_and_read_tensors(path: &std::path::Path) -> Vec<(String, u32, Vec<i64>, u32, u64)> {
    let mut file = File::open(path).expect("cannot open GGUF");

    // Header
    let _magic = read_u32(&mut file);
    let _version = read_u32(&mut file);
    let tensor_count = read_u64(&mut file);
    let metadata_kv_count = read_u64(&mut file);

    // Skip metadata
    for _ in 0..metadata_kv_count {
        let _key = read_gguf_string(&mut file);
        let val_type = read_u32(&mut file);
        skip_value(&mut file, val_type).expect("skip_value");
    }

    // Tensor info
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut file);
        let n_dims = read_u32(&mut file);
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            dims.push(read_i64(&mut file));
        }
        let ggml_type = read_u32(&mut file);
        let offset = read_u64(&mut file);
        tensors.push((name, n_dims, dims, ggml_type, offset));
    }
    tensors
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.parent().and_then(|p| p.parent()).unwrap();
    let path = workspace.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf");
    println!("Opening: {}", path.display());

    let tensors = open_and_read_tensors(&path);
    println!("Total tensors: {}", tensors.len());

    // First 10 tensors
    println!("\nFirst 10 tensors:");
    for (name, _n_dims, dims, ggml_type, _offset) in tensors.iter().take(10) {
        println!("  name='{name}' n_dims={_n_dims} shape={dims:?} type={ggml_type} offset={_offset}");
    }

    // Tensors containing snake, snake_post, or pre_tfm
    println!("\nTensors containing 'snake':");
    for (name, _n_dims, dims, ggml_type, _offset) in tensors.iter() {
        if name.contains("snake") {
            println!("  name='{name}' shape={dims:?} type={ggml_type}");
        }
    }

    println!("\nTensors containing 'pre_tfm':");
    for (name, _n_dims, dims, ggml_type, _offset) in tensors.iter() {
        if name.contains("pre_tfm") {
            println!("  name='{name}' shape={dims:?} type={ggml_type}");
        }
    }

    // Check specific tensors referenced by test
    println!("\nTest-referenced tensors:");
    let test_names = [
        "tok_dec.dec.0.snake.alpha",
        "tok_dec.dec.0.snake.beta", 
        "tok_dec.dec.1.snake.beta",
        "tok_dec.snake_post.alpha",
        "tok_dec.snake_post.beta",
        "tok_dec.dec.6.conv.bias",
        "tok_dec.dec.6.conv.weight",
        "tok_dec.pre_conv.bias",
        "tok_dec.pre_conv.weight",
        "tok_dec.upsample.0.conv.bias",
        "tok_dec.upsample.0.conv.weight",
        "tok_dec.upsample.0.dwconv.weight",
        "tok_dec.upsample.1.conv.bias",
        "tok_dec.upsample.1.conv.weight",
        "tok_dec.pre_tfm.output_proj.weight",
        "tok_dec.vq_first.0.codebook",
        "tok_dec.vq_rest.0.codebook",
        "tok_dec.vq_rest.14.codebook",
        "tok_enc.conv.0.bias",
        "tok_enc.conv.0.weight",
        "tok_enc.conv.12.bias",
        "tok_enc.conv.12.weight",
    ];
    for target in test_names {
        let found = tensors.iter().find(|(n,_,_,_,_)| n == target);
        match found {
            Some((name, _nd, dims, typ, off)) => println!("  name='{name}' shape={dims:?} type={typ} offset={off}"),
            None => println!("  MISSING: '{target}'"),
        }
    }
    
    // Also check all dec.0-6 tensors
    for i in 0..=6usize {
        println!("\nAll 'dec.{i}' tensors:");
        for (name, _nd, dims, typ, _off) in tensors.iter() {
            if name.contains(&format!("dec.{i}.")) {
                println!("  name='{name}' shape={dims:?} type={typ}");
            }
        }
    }
}

fn read_u32(file: &mut File) -> u32 {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap();
    u32::from_le_bytes(buf)
}

fn read_u64(file: &mut File) -> u64 {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).unwrap();
    u64::from_le_bytes(buf)
}

fn read_i64(file: &mut File) -> i64 {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).unwrap();
    i64::from_le_bytes(buf)
}

fn read_gguf_string(file: &mut File) -> String {
    let len = read_u64(file) as usize;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}

fn skip_value(reader: &mut File, val_type: u32) -> Result<(), String> {
    match val_type {
        0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 | 11 | 12 => {
            let sizes: [i64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];
            let size = sizes[val_type as usize];
            reader.seek(SeekFrom::Current(size)).map_err(|e| e.to_string())?;
        }
        8 => {
            let len = read_u64(reader) as i64;
            reader.seek(SeekFrom::Current(len)).map_err(|e| e.to_string())?;
        }
        9 => {
            let arr_type = read_u32(reader);
            let arr_len = read_u64(reader);
            for _ in 0..arr_len {
                skip_value(reader, arr_type)?;
            }
        }
        _ => return Err(format!("unknown type {val_type}")),
    }
    Ok(())
}
