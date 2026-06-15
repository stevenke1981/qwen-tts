# PLAN.md - Implementation Plan

## Phase 0 - Workspace And Product Shell

- [x] Convert the project into a Cargo workspace.
- [x] Add core, runtime, CLI, GUI, backend, codec, and FFI crates.
- [x] Add GGUF inspection, model download, TOML config, logging, batch synthesis,
  and WAV validation.

## Phase 1 - FFI Reference Backend

- [x] Add subprocess and in-process qwentts.cpp backends.
- [x] Support speaker, instruct, sampling parameters, and reference audio.
- [x] Keep FFI operational as the behavioral and performance reference.

## Phase 2 - Desktop App

- [x] Add the egui GUI and background workers.
- [x] Add model setup, backend selection, synthesis controls, and WAV playback.

## Phase 3 - Pure Rust Structural Pipeline

- [x] Add `crates/backends/pure-rust` to the workspace and CLI/GUI feature gates.
- [x] Load Talker and Code Predictor Q8_0 weights from GGUF.
- [x] Implement Talker autoregressive decode with KV cache.
- [x] Implement the full Code Predictor transformer and per-frame cache.
- [x] Implement the codec decoder in Rust.
- [x] Add custom Q8_0 GEMV and initial fused CPU operations.

## Phase 4 - Pure Rust Behavioral Parity (Active)

- [ ] Port prompt construction and model metadata handling.
- [ ] Sum all 16 codebook embeddings and apply text/pad overlay per frame.
- [ ] Match token suppression, EOS, repetition penalty, and sampling semantics.
- [ ] Implement language, speaker, instruct, and reference-audio modes.
- [ ] Add stage-by-stage FFI comparison fixtures.

## Phase 5 - Pure Rust Performance (Active After Parity Baseline)

- [x] Establish the first release benchmark: 25.733 s estimated for 128 frames.
- [ ] Add complete end-to-end cold/warm benchmark instrumentation.
- [ ] Make workspaces and KV caches allocation-stable.
- [ ] Remove Tensor/Vec conversion boundaries from each decoder layer.
- [ ] Fuse M=1 Q8_0 CPU kernels and tune parallel thresholds.
- [ ] Optimize codec decode and streaming TTFA.
- [ ] Reach <=5 s warm 128-frame synthesis; stretch goal <=3 s.

## Phase 6 - Strict Rust-Only Product

- [ ] Make CLI/GUI pure Rust builds independent of FFI crates and qwen.dll.
- [ ] Remove native tokenizer dependencies such as `onig_sys` from the strict
  release graph.
- [ ] Audit the release dependency tree for C/C++ build/runtime components.
- [ ] Retire qwentts.cpp from the product workspace after parity sign-off.

## Phase 7 - Optional Accelerators

- [ ] Add fused CUDA kernels only after CPU correctness and profiling are stable.
- [ ] Evaluate Metal and WGPU with the same parity and benchmark gates.
- [ ] Keep ROCm and SYCL lower priority until required by a target platform.

The authoritative checklist and acceptance criteria are in `TASKS.md`.
