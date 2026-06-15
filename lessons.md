---
## Lesson #1 - 2026-06-14
**Trigger:** Generated OpenCode cache appeared as an untracked `${PROJECT_ROOT}/` folder before release staging.
**Rule:** Before `git add -A`, run `git status --short --ignored` and add accidental local cache folders to `.gitignore` instead of staging them.
**Source:** complete compiled version handoff

---
## Lesson #3 — 2026-06-15
**Trigger:** Used `candle_core` 0.10.x for GGUF loading but assumed `QTensor::matmul()` existed — it does not.
**Rule:** Before using a crate's API, fetch the docs.rs page to verify method signatures exist before writing code against them.
**Source:** pure-rust backend implementation

---
## Lesson #4 — 2026-06-15
**Trigger:** `candle_nn::embedding()` is a builder that constructs an `Embedding` struct via `VarBuilder`, not a forward function. Passing a weight tensor directly fails.
**Rule:** For GGUF-only models where VarBuilder isn't used, implement embedding lookup manually via `weights.gather(input_ids, 0)` instead of trying to use `candle_nn::embedding`.
**Source:** pure-rust talker implementation

---
## Lesson #5 — 2026-06-15
**Trigger:** `candle_nn::RmsNorm` takes `f64` epsilon (not `f32`), `Module::forward` requires explicit import, and `IndexOp::i()` also needs explicit import.
**Rule:** When using candle-nn types like RmsNorm, import `candle_core::{Module, IndexOp}` explicitly. Always check the exact type signature (f64 vs f32) on docs.rs before writing constructors.
**Source:** pure-rust talker implementation

---
## Lesson #2 - 2026-06-14
**Trigger:** Release copy failed because `dist/qwen-tts-gui.exe` was still running and locked on Windows.
**Rule:** Before copying release GUI binaries into `dist`, check for running `qwen-tts-gui` processes and stop the old `dist` executable if it locks the target file.
**Source:** qwentts backend implementation

---
## Lesson #3 - 2026-06-14
**Trigger:** MSVC link error LNK1181: ggml.lib not found because import libraries were in `build/ggml/src/Release/` not `build/Release/`.
**Rule:** When linking pre-built CMake shared libraries, check both the DLL directory and the subproject object directory for import libraries (.lib). Run `Get-ChildItem -Recurse -Filter "*.lib"` on the build tree rather than guessing the location.
**Source:** native-cpu-backend-m2-ffi-inference

---
## Lesson #4 - 2026-06-14
**Trigger:** GGUF F16 and Q8_0 tensor reads produced all-zero values because `f16_bits_to_f32` normal-branch bit manipulation was wrong (treated `exp` as raw field value instead of bit-shifted position).
**Rule:** When implementing IEEE 754 half-precision conversion, shift the raw bit fields right to their proper positions FIRST (raw_exp = (h>>10)&0x1F, raw_mant = h&0x3FF), compute the f32 exponent as `raw_exp - 15 + 127`, then construct the final f32 bits with `(sign<<31)|(f32_exp<<23)|(raw_mant<<13)`. Test with 0x3C00→1.0, 0x3800→0.5, 0xC000→-2.0.
**Source:** codec-decoder-rust (Task 1: GGUF loader)

---
## Lesson #5 - 2026-06-14
**Trigger:** DAC output blew up to ±47 (normal expected ±0.14) because SnakeBeta denominator was `1/beta` instead of `1/(exp(beta)+1e-9)`.
**Rule:** When porting activation functions from ggml to Rust, examine the C++ load-time pre-computation, not just the runtime formula. The C++ code pre-computes `a = exp(alpha)` and `inv_b = 1/(exp(beta)+1e-9)` and stores those, not the raw parameters. If your Rust implementation computes `exp(alpha)` at runtime, make sure you also compute `exp(beta)` and the epsilon-guarded inverse. Don't assume `beta` is used directly as a denominator.
**Source:** codec-decoder-rust (Task 1: Codec decoder RS)

---
## Lesson #6 - 2026-06-14
**Trigger:** CLI `cmd_info()` used `probe.architecture()` and `probe.description()` which don't exist on `GgufProbe` struct.
**Rule:** Before calling methods on a Rust type from a crate you didn't write today, read the struct definition and `impl` block to verify the method exists. Guess-by-name (e.g., `probe.architecture()`) is not reliable for types with limited public API. Use `grep "pub fn"` on the source file to enumerate available methods.
**Source:** codec-decoder-rust (CLI tool)

---
## Lesson #7 - 2026-06-14
**Trigger:** Adding fields to `SynthesisRequest` broke 6 construction sites (config.rs, scheduler.rs, backend tests, CLI, app, CPU backend).
**Rule:** Before adding required fields to a widely-used struct, grep for every construction site (`StructName {`) and update all of them in the same commit to avoid intermediate broken states.
**Source:** ffi-backend-completeness

---
## Lesson #8 - 2026-06-14
**Trigger:** FfiBackend and CpuBackend both needed read_wav_f32_mono but in different crates
**Rule:** Before duplicating a utility function across backends, check if it can be exported from `qwen-tts-core` (the shared core crate) instead of each backend implementing its own copy.
**Source:** feat(cpu-backend): wire instruct, ref_audio, sampling params

---
## Lesson #10 — 2026-06-15
**Trigger:** Extracting a subdirectory (`opencode-tui/`) from a mono-repo (`qwen-tts`) into a standalone git repo with a separate remote.
**Rule:** When splitting a subdirectory into its own git repo: (1) `git init` inside the subdirectory, (2) update `.gitignore` to exclude symlinks/mount-points like `${PROJECT_ROOT}/` that reference the parent, (3) use `git add -A` + `git commit` + `git remote add origin <url>` + `git push -u origin main`. Verify with `git status --short` that only the intended files are tracked before committing.
**Source:** opencode-tui standalone repo setup

---
## Lesson #11 — 2026-06-15
**Trigger:** `D:\qwen_tts\opencode-tui` directory was locked during `Move-Item` and `robocopy` because `${PROJECT_ROOT}/` is an opencode mount point containing active `memory.db` files.
**Rule:** When moving a directory that contains an opencode `${PROJECT_ROOT}` mount point, use `robocopy /E /COPY:DAT /MOVE` to copy files, then accept that the mount point itself can't be deleted until the opencode process exits. Verify the target directory has `.git` and source code intact. The empty source shell can be cleaned on next reboot.
**Source:** opencode-tui relocation to plugins directory

---
## Lesson #9 - 2026-06-14
**Trigger:** A TypeScript TUI plugin passed `tsc` but emitted an import of `@opentui/solid/jsx-runtime`, which is type-only and failed at runtime.
**Rule:** Build OpenTUI Solid TSX with `jsx: preserve` and `@opentui/solid/bun-plugin`; include a Bun runtime import or slot-registration smoke test in validation.
**Source:** OpenCode persistent status footer

---
## Lesson #10 - 2026-06-14
**Trigger:** Testing `opencode plugin --global` with `OPENCODE_CONFIG_DIR` still updated the real global OpenCode config.
**Rule:** Test OpenCode plugin installation in a temporary Git project without `--global`; reserve global installer tests for an explicitly disposable user config and verify cleanup immediately.
**Source:** OpenCode persistent status footer

---
## Lesson #11 - 2026-06-15
**Trigger:** Extracting a subdirectory (`opencode-tui/`) from a mono-repo (`qwen-tts`) into a standalone git repo with a separate remote.
**Rule:** When splitting a subdirectory into its own git repo: (1) `git init` inside the subdirectory, (2) update `.gitignore` to exclude symlinks/mount-points like `${PROJECT_ROOT}/` that reference the parent, (3) use `git add -A` + `git commit` + `git remote add origin <url>` + `git push -u origin main`. Verify with `git status --short` that only the intended files are tracked before committing.
**Source:** opencode-tui standalone repo setup

---
## Lesson #12 - 2026-06-15
**Trigger:** `D:\qwen_tts\opencode-tui` directory was locked during `Move-Item` and `robocopy` because `${PROJECT_ROOT}/` is an opencode mount point containing active `memory.db` files.
**Rule:** When moving a directory that contains an opencode `${PROJECT_ROOT}` mount point, use `robocopy /E /COPY:DAT /MOVE` to copy files, then accept that the mount point itself can't be deleted until the opencode process exits. Verify the target directory has `.git` and source code intact. The empty source shell can be cleaned on next reboot.
**Source:** opencode-tui relocation to plugins directory

---
## Lesson #13 - 2026-06-15
**Trigger:** Config metadata keys used `qwen3-tts.talker.*` prefix, not `llama.*` — ModelConfig returned defaults (24 layers, wrong vocab) instead of real values.
**Rule:** Before trusting default metadata keys in GGUF parsing, probe real model metadata with a test that prints all keys, then update lookup to try architecture-specific prefixes first (`qwen3-tts.talker.*` → `qwen3-tts.*` → `llama.*`).
**Source:** GGUF tensor naming fix

---
## Lesson #14 — 2026-06-15
**Trigger:** `Tensor::matmul()` in `candle-core` 0.10.x panicked with "shape mismatch" when given 2D weight and 3D hidden — unlike PyTorch/ggml, candle requires both operands to have the same rank.
**Rule:** Before writing matmul operations with candle, check the rank of both tensors. If ranks differ (e.g., 2D weight [out, in] @ 3D hidden [B, T, in]), flatten the higher-rank tensor to 2D first: `x_2d = x.reshape((batch*seq_len, in_dim))`, compute `x_2d.matmul(&weight.t())`, then reshape back to original rank with new last dim. Extract this as a `linear_fwd(weight, x)` helper.
**Source:** pure-rust matmul rank mismatch fix

---
## Lesson #15 — 2026-06-15
**Trigger:** After rewriting the pipeline to use KV cache, the code predictor failed with `shape mismatch in matmul, lhs: [1024, 2048], rhs: [1, 2048, 1]` — `forward_step` returned 3D `[batch, 1, d_model]` but code predictor's `predict_one_frame_sampled` expected 2D `[batch, d_model]`. The original pipeline's `i((0, seq_len-1, ..))` reduced rank from 3D to 1D, then `.unsqueeze(0)` made it 2D; the new pipeline's `forward_step` returned 3D directly.
**Rule:** When refactoring a pipeline function that changes intermediate tensor shapes, verify ALL downstream consumers accept the new shape. Use `squeeze()` / `unsqueeze()` explicitly at the call site to match the callee's expected rank rather than changing the callee's interface.
**Source:** KV cache pipeline rewrite
