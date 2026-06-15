# Pure Rust FFI Parity And 5x Optimization Plan

**Goal:** Make `pure-rust` behaviorally equivalent to the FFI reference, then
reduce warm 128-frame CPU synthesis from 25.733 s to <=5.0 s.

**Complexity:** L4

## Baseline

- Talker: 10.392 s / 128 steps.
- Code Predictor: 12.826 s / 128 frames.
- Model load: 2.514 s.
- Current estimate excludes full prompt and measured codec time.

## Work Packages

1. [ ] Add stage timers and reproducible FFI/Rust benchmark output.
   - Files: `pipeline.rs`, `tests/cross_val.rs`, `tests/q8_bench.rs`.
   - Output: cold/warm JSON or CSV with prompt, Talker, Predictor, Codec, WAV,
     TTFA, and total time.
2. [ ] Port prompt construction and generation metadata.
   - New modules: `prompt.rs`, `metadata.rs`.
   - Output: matching prompt IDs, input embeddings, trailing hidden states, and
     TTS pad embedding.
3. [ ] Correct autoregressive feedback and sampling.
   - Files: `pipeline.rs`, `talker.rs`, `code_predictor.rs`, `sampling.rs`.
   - Output: sum all 16 codebook embeddings, apply overlay, suppress invalid
     tokens, stop on EOS, and apply repetition penalty.
4. [ ] Add deterministic parity fixtures.
   - Output: first prompt, hidden, logits, frame tokens, next embedding, and
     codec output compared with the FFI dump path.
5. [ ] Remove Talker hot-path allocation.
   - Replace `Tensor::cat` KV growth with preallocated flat K/V buffers.
   - Persist `Q8Workspace` and reusable layer buffers.
6. [ ] Fuse the M=1 CPU layer path.
   - Eliminate per-op Tensor/Vec conversion.
   - Fuse norm/quantize, QKV dispatch, SwiGLU, residual, and output writes.
7. [ ] Optimize Code Predictor and codec by measured time share.
   - Target Predictor first because it currently costs 12.826 s.
   - Add chunked codec timing and optimize only proven hot loops.
8. [ ] Create a strict Rust-only release profile.
   - Disable FFI defaults.
   - Remove `onig_sys`/native tokenizer dependencies.
   - Add dependency-tree and DLL absence checks.

## Risks

| Risk | Mitigation |
|------|------------|
| Faster kernels preserve current wrong semantics | Complete parity stages before accepting performance work |
| RNG differences hide deterministic defects | Compare argmax tokens/stages; use audio metrics for sampled runs |
| Rayon overhead erases Q8 gains | Benchmark representative M=1 shapes and use per-size thresholds |
| Codec becomes the new bottleneck | Include codec in every end-to-end benchmark |
| Strict Rust build still compiles native code | Audit `cargo tree` and built DLL/import dependencies in CI |

## Definition Of Done

- [ ] P0 parity gates in `TASKS.md` pass.
- [ ] Warm 128-frame end-to-end synthesis is <=5.0 s.
- [ ] Strict release runs without qwen.dll/GGML DLLs/native C/C++ dependencies.
- [ ] `cargo fmt`, tests, and clippy acceptance gates pass.
- [ ] Benchmark and parity results are committed with the implementation.

## Assumptions

- The 2-5 s FFI target is measured on the same reference machine and workload.
- FFI remains available during development only as a comparison oracle.
- CPU parity is completed before optional GPU work becomes the main path.
