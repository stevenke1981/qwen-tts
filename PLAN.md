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

## Phase 3 — qwentts.cpp FFI

- [ ] Add `crates/qwentts-sys`.
- [ ] Generate bindings from qwentts.cpp public C ABI.
- [ ] Add safe wrapper crate.
- [ ] Replace process execution with direct in-process inference.

## Phase 4 — GUI

- [ ] Add GUI crate: `crates/app`.
- [ ] Choose Tauri, egui, or Slint.
- [ ] Add model path settings.
- [ ] Add text box, language selector, voice selector, output path selector.
- [ ] Add playback.

## Phase 5 — Native backend experiments

Recommended order:

1. CPU backend.
2. CUDA backend.
3. Metal backend.
4. WGPU backend.
5. ROCm backend.
6. SYCL backend.

Do not implement every backend at once. Stabilize the runtime trait and FFI first.
