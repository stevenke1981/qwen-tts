# PLAN.md — Implementation Plan

## Phase 0 — Workspace skeleton

- [x] Convert single-crate project into Cargo workspace.
- [x] Add `core`, `runtime`, backend crates, and `cli`.
- [x] Keep backend crates lightweight and compilable without GPU SDKs.

## Phase 1 — MVP speech generation

- [x] Add GGUF header inspection.
- [x] Add qwentts.cpp CLI adapter.
- [x] Add `synth` command.
- [x] Add setup-script generator.
- [ ] Test against local qwentts.cpp binary and Qwen3-TTS GGUF files.

## Phase 2 — Robust runtime

- [x] Replace ad-hoc CLI args with a real parser such as `clap`.
- [x] Add TOML config loader.
- [x] Add structured logs.
- [x] Add wav metadata validation after generation.
- [x] Add batch synthesis.
- [x] Add default GGUF model status and download workflow.

## Phase 3 — qwentts.cpp FFI ✅

- [x] Add `crates/qwentts-sys`.
- [x] Generate bindings from qwentts.cpp public C ABI.
- [x] Add safe wrapper crate.
- [x] Replace process execution with direct in-process inference.
- [x] `--backend ffi` is now the default CLI backend.
- [x] Voice reference (ref_audio / ref_text) support.
- [x] Instruct / speaker / sampling params (seed, temp, top-k/p, rep-penalty).

## Phase 4 — GUI ✅

- [x] Add GUI crate: `crates/app`.
- [x] Choose Tauri, egui, or Slint.
- [x] Add model path settings.
- [x] Add text box, language selector, voice selector, output path selector.
- [x] FfiBackend backend selection.
- [x] Instruct / ref_audio / ref_text form fields.
- [x] Collapsible advanced params (seed, temp, top-k/p, flash attention, etc.)
- [x] Audio playback via rodio (auto-play + play/pause/stop + progress bar).

## Phase 5 — Native backend experiments

Recommended order:

1. CPU backend — `crates/backends/cpu` (FFI-to-C++ via qwentts-sys, functional).
2. CUDA backend — skeleton crate exists.
3. Metal backend — skeleton crate exists.
4. WGPU backend — skeleton crate exists.
5. ROCm backend — skeleton crate exists.
6. SYCL backend — skeleton crate exists.

Pure Rust reimplementation of the TTS pipeline (ggml-free) is a future
milestone once the FFI path is fully stabilized.
