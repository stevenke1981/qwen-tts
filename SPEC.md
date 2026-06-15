# SPEC.md - Qwen TTS

## Product Goal

Build a local Qwen3-TTS application whose production inference path is written
in Rust and consumes Qwen3-TTS GGUF models directly.

The product must provide:

- Native Rust CLI and egui GUI.
- GGUF model inspection, download, and configuration.
- Pure Rust prompt construction, Talker, Code Predictor, codec decode, sampling,
  WAV output, and optional streaming.
- Language, speaker, instruct, and reference-audio modes.
- CPU performance comparable to the existing qwentts.cpp FFI reference.
- Optional Rust-managed accelerator backends after CPU parity is established.

## Reference And Release Policy

- qwentts.cpp/FFI is a temporary development oracle for behavior, stage dumps,
  audio quality, and benchmark comparison.
- The strict release path must not load `qwen.dll`, GGML DLLs, or
  `qwen-tts-sys`.
- After parity and performance acceptance, C++ vendor/FFI code is removed from
  the product workspace or retained only in an excluded comparison tool.

## Performance Targets

Reference workload: 128 generated codec frames on the designated Windows CPU.

- Warm synthesis target: <=5.0 s.
- Warm synthesis stretch target: <=3.0 s.
- Cold start target: <=7.0 s, including model load.
- Time reports must include prompt, Talker, Code Predictor, codec, and WAV output.

## Crate Responsibilities

### `crates/core`

- Model paths, GGUF metadata, audio/WAV types, and backend-neutral contracts.

### `crates/runtime`

- `RuntimeBackend`, scheduler, request/response types, configuration, logging,
  model management, and output naming.
- No inference implementation details.

### `crates/backends/pure-rust`

- Prompt builder and tokenizer integration.
- Q8_0 Talker and Code Predictor inference.
- KV caches, sampling, special-token rules, and request-mode handling.
- Rust CPU execution and optional Rust-managed accelerator features.

### `crates/codec`

- GGUF codec weights, RVQ decode, transformer, upsample, DAC, and chunked audio
  decode.

### `crates/cli` And `crates/app`

- User interfaces only. The strict release defaults to `pure-rust` after its
  acceptance gates pass.

### Reference-only crates

- `crates/qwentts-sys`, `crates/backends/cpu`, and subprocess/FFI runtime paths
  are temporary parity tools, not the final product architecture.

## Pure Rust Data Flow

```text
request
  -> tokenizer + prompt builder
  -> Talker prefill + KV cache
  -> Talker codebook-0 sampling
  -> Code Predictor codebooks 1-15
  -> sum 16 codebook embeddings + text/pad overlay
  -> repeat until EOS / frame limit
  -> Rust codec chunk decode
  -> 24 kHz mono PCM/WAV or streaming chunks
```

## Correctness Requirements

- No request field is silently ignored.
- Argmax mode is used for deterministic stage parity.
- Sampled mode is compared with structural/audio metrics when RNG algorithms
  differ.
- Every optimization must preserve finite outputs, token constraints, duration,
  and documented numeric tolerances.
- Unsupported model modes return explicit errors.

## Non-goals

- Training or fine-tuning.
- Implementing every accelerator before CPU parity.
- Bit-exact sampled audio when the reference and Rust RNG algorithms differ.
