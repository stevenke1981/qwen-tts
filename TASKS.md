# TASKS.md - Parallel Implementation Tasks

Source documents: `PLAN.md`, `SPEC.md`.

## Active Phase

Phase 2 - Robust runtime.

## Task A - CLI Parser

Owner: CLI worker.

Scope:

- `crates/cli/Cargo.toml`
- `crates/cli/src/main.rs`
- Optional CLI usage docs only when needed.

Goals:

- Replace manual argument scanning with a real parser.
- Preserve the existing commands: `inspect`, `graph`, `setup-script`, `synth`.
- Keep existing defaults and environment-variable fallbacks.
- Make help and validation predictable.

Acceptance:

- `cargo check -p qwen-tts-cli`
- Command parsing remains compatible with existing README examples.

## Task B - Runtime Config And Logging

Owner: runtime config worker.

Scope:

- `crates/runtime/Cargo.toml`
- `crates/runtime/src/lib.rs`
- `crates/runtime/src/config.rs`
- `crates/runtime/src/logging.rs`
- `qwen-tts.toml.example`

Goals:

- Add TOML config loading for executable path, model paths, default language, default device, and output directory.
- Add structured logging initialization that can be called by CLI or future GUI.
- Keep config types independent from CLI parsing.

Acceptance:

- Runtime unit tests cover TOML parsing.
- Logging initialization is idempotent enough for tests and callers.

## Task C - WAV Validation And Batch Runtime

Owner: runtime synthesis worker.

Scope:

- `crates/core/src/audio.rs`
- `crates/core/src/lib.rs`
- `crates/runtime/src/backend.rs`
- `crates/runtime/src/scheduler.rs`
- Optional new modules under `crates/core/src/` or `crates/runtime/src/`.

Goals:

- Validate generated WAV headers after synthesis.
- Report actual sample rate and channel count instead of assuming 24 kHz mono.
- Add batch synthesis while keeping the single-request API.

Acceptance:

- Unit tests cover minimal WAV header validation.
- Batch API tests cover success and per-item failure reporting.
- `cargo check --workspace`

## Integration Notes

- Baseline `cargo check --workspace` currently fails before Phase 2 changes:
  - `qwen_tts_core::TtsGraph` is not re-exported from `crates/core/src/lib.rs`.
  - `crates/cli/src/main.rs` moves a `PathBuf` into `GgufProbe::open` before displaying it.
- The workspace is not currently a git repository, so implementation should be validated locally without commit steps.
- CBRLM project index: `cbrlm+D-qwen_tts`.
