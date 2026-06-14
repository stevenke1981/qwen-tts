//! Pure Rust codec decoder for the qwen3-tts TTS model.
//!
//! This crate implements the full codec decoder pipeline in pure Rust,
//! reading Q8_0 quantized weights from the codec GGUF file:
//!
//! ```text
//! codes [T, 16] → quantizer → pre_conv → transformer(8L) → upsample(2×) → DAC(4 blocks) → audio
//! ```
//!
//! All modules are implemented. The pipeline produces 24 kHz mono audio.

pub mod conv;
pub mod dac;
pub mod gguf;
pub mod pipeline;
pub mod pre_conv;
pub mod pre_transformer;
pub mod q8_0;
pub mod quantizer;
pub mod resunit;
pub mod snake;
pub mod tconv;
pub mod upsample;

// Re-export main entry point
pub use pipeline::CodecDecoder;

// ---------------------------------------------------------------------------
// RVQ packing constants and utilities
// ---------------------------------------------------------------------------

/// Bits per RVQ code (11 bits → indices 0..2047).
pub const CODE_BITS: u32 = 11;

/// Number of codebooks per frame (1 semantic + 15 acoustic).
pub const NUM_CODEBOOKS: usize = 16;

/// Unpack .rvq format bytes back to flat i32 codes (K-major layout).
///
/// The .rvq file stores codes packed at 11 bits/code, LSB-first,
/// in K-major layout: [K, T] where K=16 varies fastest.
///
/// # Panics
///
/// Panics if the data is too small for `n_codes`.
pub fn unpack_rvq(data: &[u8], n_codes: usize) -> Vec<i32> {
    let mask = (1u32 << CODE_BITS) - 1;
    let mut out = vec![0i32; n_codes];
    let mut acc: u64 = 0;
    let mut bits_in_acc = 0;
    let mut in_pos = 0;
    for i in 0..n_codes {
        while bits_in_acc < CODE_BITS as i32 && in_pos < data.len() {
            acc |= (data[in_pos] as u64) << bits_in_acc;
            in_pos += 1;
            bits_in_acc += 8;
        }
        out[i] = (acc & mask as u64) as i32;
        acc >>= CODE_BITS;
        bits_in_acc -= CODE_BITS as i32;
    }
    out
}

/// Transpose codes from K-major [K, T] to T-major [T, K] layout.
pub fn transpose_codes(k_major: &[i32], t: usize, k: usize) -> Vec<i32> {
    assert_eq!(k_major.len(), t * k);
    let mut t_major = vec![0i32; t * k];
    for ti in 0..t {
        for ki in 0..k {
            t_major[ti * k + ki] = k_major[ki * t + ti];
        }
    }
    t_major
}

/// Pack codes (K-major flat [K*T]) into .rvq format bytes (11 bits/code).
pub fn pack_rvq(codes: &[i32]) -> Vec<u8> {
    let mask = (1u32 << CODE_BITS) - 1;
    let total_bits = codes.len() as u64 * CODE_BITS as u64;
    let n_bytes = ((total_bits + 7) / 8) as usize;
    let mut out = vec![0u8; n_bytes];
    let mut acc: u64 = 0;
    let mut bits_in_acc = 0;
    let mut out_pos = 0;
    for &code in codes {
        acc |= ((code as u32 & mask) as u64) << bits_in_acc;
        bits_in_acc += CODE_BITS as i32;
        while bits_in_acc >= 8 {
            out[out_pos] = (acc & 0xFF) as u8;
            out_pos += 1;
            acc >>= 8;
            bits_in_acc -= 8;
        }
    }
    if bits_in_acc > 0 {
        out[out_pos] = (acc & 0xFF) as u8;
    }
    out
}
