# TASKS.md - Pure Rust Production Roadmap

Updated: 2026-06-15

Source documents: `SPEC.md`, `PLAN.md`, and
`docs/superpowers/plans/2026-06-15-pure-rust-ffi-parity-optimization.md`.

## Goal

Ship Qwen3-TTS through a production `pure-rust` backend with no qwentts.cpp,
`qwen.dll`, or `qwen-tts-sys` dependency in the release path. Match the FFI
backend's speech behavior first, then reach comparable CPU latency.

## Current Status

Milestone-weighted production readiness: approximately **45%**.

| Area | Status | Evidence / remaining gap |
|------|--------|--------------------------|
| Rust workspace, CLI, GUI, model management | Complete | Existing release binaries and workspace crates |
| Pure Rust codec decoder | Implemented | Decoder pipeline and GGUF tests exist |
| Pure Rust Talker | Implemented, not parity-complete | 28-layer autoregressive path and KV cache exist |
| Pure Rust Code Predictor | Implemented, not parity-complete | 5-layer predictor and per-frame cache exist |
| Prompt and generation parity | Incomplete | No equivalent of the C++ `PromptBuilder`; next embedding is incomplete |
| Voice clone / speaker / instruct parity | Incomplete | Pure Rust pipeline does not consume these request fields |
| CPU performance | Incomplete | 128-frame estimate is 25.733 s versus 2-5 s target |
| Strict Rust-only dependency graph | Incomplete | `tokenizers` currently brings `onig_sys`/`cc`; FFI crates remain defaults |
| Production verification | Incomplete | No mandatory stage-by-stage parity or performance gate |

### Measured CPU baseline

Reference command:

```powershell
cargo test -p qwen-tts-backend-pure-rust --release `
  --test q8_bench bench_128_frames -- --ignored --nocapture
```

Measured on 2026-06-15:

| Stage | Time |
|-------|------|
| Model load | 2.514 s |
| Talker, 128 steps | 10.392 s |
| Code Predictor, 128 frames | 12.826 s |
| Estimated subtotal | 25.733 s |

This benchmark does not include full prompt construction or a measured
end-to-end codec stage. The real synthesis baseline must be recorded before a
performance claim is accepted.

## P0 - Correctness And Audio Parity

- [ ] Port the C++ prompt builder into Rust.
  - Parse model special tokens, language metadata, speaker metadata, and
    generation defaults from GGUF.
  - Produce the same prompt IDs, input embeddings, trailing text hidden states,
    and TTS pad embedding as the FFI reference.
- [ ] Build the next Talker embedding from all 16 generated codebooks.
  - Add codebook 0 from `talker.codec_embd.weight`.
  - Add codebooks 1-15 from `code_pred.codec_embd.*.weight`.
  - Add trailing text hidden state or TTS pad overlay.
- [ ] Match generation controls.
  - Apply codec-token suppression before Talker sampling.
  - Stop on codec EOS instead of always generating `max_new_tokens` frames.
  - Implement repetition penalty and separate Talker/subtalker sampling params.
  - Define deterministic argmax parity separately from sampled RNG parity.
- [ ] Implement all `SynthesisRequest` modes in the pure Rust pipeline.
  - `language`, `speaker`, and `instruct`.
  - `ref_audio_path` and `ref_text` for voice cloning.
  - Clear unsupported-mode errors until each mode is implemented.
- [ ] Add stage dump comparison against FFI.
  - Prompt IDs and prompt embeddings.
  - First Talker hidden/logits.
  - First-frame 16 codec tokens.
  - Step-0 next embedding and step-1 hidden.
  - Codec intermediate/output checks.

### P0 acceptance

- [ ] Argmax mode produces matching first-frame tokens and matching stage dumps
  within documented numeric tolerances.
- [ ] Sampled mode passes duration, RMS, peak, finite-value, and audible speech
  checks with the same request settings.
- [ ] Language, speaker, instruct, and reference-audio modes either work or fail
  explicitly; no request field is silently ignored.

## P1 - Reach The 2-5 Second CPU Target

- [ ] Replace the synthetic estimate with a repeatable end-to-end benchmark.
  - Report cold start, warm synthesis, TTFA, Talker, Predictor, Codec, and WAV I/O.
  - Test 32 and 128 frames with identical FFI and pure Rust requests.
  - Persist results as JSON/CSV so regressions are visible.
- [ ] Make workspaces persistent.
  - Store `Q8Workspace` on `Talker`/`CodePredictor` execution state instead of
    creating it inside each forward call.
  - Reuse sampling, logits, embedding, and output buffers.
- [ ] Replace Talker `Tensor::cat` KV growth with preallocated flat K/V storage.
  - Allocate the maximum active context once.
  - Append by writing into a position, not copying the full cache every step.
- [ ] Remove Tensor-to-Vec-to-Tensor churn from the hot path.
  - Keep RMSNorm, RoPE, attention, residual, SwiGLU, and Q8 GEMV on borrowed
    slices/buffers through an entire layer.
  - Convert to `Tensor` only at stable API boundaries.
- [ ] Fuse CPU kernels around the autoregressive M=1 path.
  - RMSNorm + input quantization.
  - Q/K/V projection dispatch.
  - Gate/up projection + SiLU multiply.
  - Residual add + output writeback.
- [ ] Tune parallelism for each matrix size.
  - Reuse one Rayon pool.
  - Add row-count thresholds to avoid parallel overhead on small projections.
  - Benchmark physical-core count and thread affinity on the reference machine.
- [ ] Profile and optimize the Rust codec separately.
  - Measure quantizer, pre-conv, transformer, upsample, and DAC.
  - Add SIMD/Rayon only to measured hot loops.
  - Add rolling chunk decode with left context for lower TTFA.
- [ ] Reduce cold-start cost.
  - Memory-map GGUF weights where safe.
  - Keep a loaded pipeline alive across requests.
  - Avoid parsing/loading the Talker file twice for Talker and Predictor.

### P1 acceptance

- [ ] Warm 128-frame synthesis is **<= 5.0 s** on the reference CPU.
- [ ] Stretch goal: warm 128-frame synthesis is **<= 3.0 s**.
- [ ] Initial cold-start target is **<= 7.0 s**; stretch target is **<= 5.0 s**.
- [ ] Performance results include codec and WAV output, not only model steps.
- [ ] Audio parity gates still pass after every kernel optimization.

## P2 - Strict Rust-Only Release

- [ ] Add a release feature/profile where CLI and GUI default to `pure-rust` and
  do not enable `ffi`.
- [ ] Remove `qwen-tts-backend-cpu` from the pure Rust CLI/GUI dependency path;
  it still wraps C++ through FFI.
- [ ] Replace or reconfigure the tokenizer stack so the strict build does not
  compile `onig_sys`, `cc`, or other C/C++ runtime components.
- [ ] Add a dependency audit command that fails if `qwen-tts-sys`, `onig_sys`,
  `cmake`, `bindgen`, or qwentts.cpp enters the strict release graph.
- [ ] Keep FFI only as a development parity oracle until P0/P1 pass.
- [ ] After parity sign-off, archive or remove `vendor/qwentts.cpp`,
  `crates/qwentts-sys`, and the FFI-only CPU backend from the product workspace.

### P2 acceptance

- [ ] `qwen-tts.exe` and `qwen-tts-gui.exe` start without `qwen.dll` or GGML DLLs.
- [ ] The strict release dependency audit contains no C/C++ build/runtime crates.
- [ ] A clean machine can build and synthesize using the Rust toolchain plus
  model files only.

## P3 - Quality Gates And Cleanup

- [ ] Fix `cargo fmt --all -- --check` across the workspace.
- [ ] Fix pure Rust warnings and make
  `cargo clippy -p qwen-tts-backend-pure-rust --all-targets -- -D warnings` pass.
- [ ] Split fast unit tests from model-heavy parity/performance tests.
- [ ] Mark model-heavy tests consistently and provide one documented command for
  each test tier.
- [ ] Update stale docs and delete superseded plans only after their useful
  decisions have been merged into this file.
- [ ] Regenerate the Codebase Memory MCP index after structural changes.

## Immediate Work Order

1. Implement prompt/next-embedding parity and stage dumps.
2. Establish true FFI-versus-Rust end-to-end benchmark output.
3. Eliminate hot-path allocation and Tensor/Vec conversions.
4. Optimize the Code Predictor first, then Talker, then codec based on measured
   time share.
5. Switch the product default to pure Rust only after parity and <=5 s gates pass.
