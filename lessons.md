---
## Lesson #1 - 2026-06-14
**Trigger:** Generated OpenCode cache appeared as an untracked `${PROJECT_ROOT}/` folder before release staging.
**Rule:** Before `git add -A`, run `git status --short --ignored` and add accidental local cache folders to `.gitignore` instead of staging them.
**Source:** complete compiled version handoff

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
