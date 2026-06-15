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

---
## Lesson #16 — 2026-06-15
**Trigger:** CUDA pipeline tests showed only ~1× speedup over CPU for autoregressive decode with tiny matmuls `[1, 2048] @ [6144, 2048]`.
**Rule:** Before assuming GPU acceleration will help, profile the matrix sizes: if the largest matmul dimension is < 8192 and the batch dimension is 1, kernel launch overhead dominates (>95%), making GPU ~same as CPU. GPU helps only with larger matrices (prefill, batched inference, or kernel fusion).
**Source:** CUDA pipeline verification tests

---
## Lesson #17 — 2026-06-15
**Trigger:** Cross-validation test showed 0% sample match between Pure Rust (argmax) and FFI (temperature=1.0) with same seed=42 — different RNG implementations (StdRng vs Mersenne Twister) produce completely different acoustic code tokens.
**Rule:** When cross-validating two implementations that differ in code predictor architecture (simplified vs full) AND RNG, compare structural metrics (duration, RMS, peak, SDR) instead of sample-by-sample exactness. A 0% sample match is expected and valid as long as both produce reasonable audio of the same length.
**Source:** Cross-validation test suite

---
## Lesson #18 — 2026-06-15
**Trigger:** Rewrote code_predictor.rs from simplified linear predictor to full 5-layer transformer with KV cache, requiring coordinated changes across 3 source files and 3 test files.
**Rule:** When changing a struct's method signature (e.g., `&self → &mut self`), grep for ALL callers in ALL test files before editing — the compiler catches lib callers but test files are only checked at test compile time.
**Source:** Full code predictor implementation (Task 2)

---
## Lesson #19 — 2026-06-15
**Trigger:** E2E test failed with "index-select only supports contiguous tensors" after transposing a square embedding weight [2048,2048] for index_select in the new code predictor.
**Rule:** After any `Tensor::t()` (transpose) in candle, call `.contiguous()?` before `index_select()` — transpose produces a non-contiguous view, and index_select requires contiguous storage. For square matrices where transpose is a no-op, skip the transpose entirely.
**Source:** Full code predictor E2E verification

---
## Lesson #20 — 2026-06-15
**Trigger:** Investigated KV cache pre-allocation optimization using `slice_assign`, discovered candle's implementation uses `where_cond` + `pad_with_zeros`, creating 3× full-capacity copies per step — worse than `cat`'s growing allocation.
**Rule:** Before optimizing anything, read the actual library source code first (candle source is in `~/.cargo/registry/src/`). Verify that the optimization primitive (`slice_assign`, `expand`, etc.) is actually in-place. Many functional tensor libraries (including candle) return NEW tensors from every operation.
**Source:** KV cache pre-allocation analysis

---
## Lesson #21 — 2026-06-15

---
## Lesson #22 — 2026-06-15
**Trigger:** Q8_0 quantized matmul (QMatMul) was 30× slower than F32 gemm for autoregressive inference.
**Rule:** Before replacing F32 matmul with quantized matmul for autoregressive inference, benchmark the M=1 (GEMV) case. candle's `k_quants::matmul` uses Rayon parallelism via `into_par_iter()` with chunk sizes 128-512, which adds overhead that dominates for single-token autoregressive steps. Use `linear_fwd(x @ W^T)` with F32 weights (gemm crate) instead. If quantized matmul is needed, write a custom single-threaded GEMV that avoids Rayon.
**Source:** qmatmul-dequant-revert
**Trigger:** Release mode benchmark showed 21× speedup over debug mode (513s → 24s for 4-frame E2E test), confirming debug mode F32 matmul as the real bottleneck.
**Rule:** Before spending time on algorithmic optimizations (KV cache pre-alloc, F16, etc.), always benchmark in release mode first. The debug/release gap for candle F32 matmuls can be ~100× due to naive scalar loops vs SIMD auto-vectorization.
**Source:** Release mode benchmark

---
## Lesson #23 — 2026-06-15
**Trigger:** Implemented custom Q8_0 GEMV and discovered that `BlockQ8_0` fields (`d`, `qs`) are `pub(crate)` in candle-core 0.10.2, making them inaccessible from external crates.
**Rule:** When using quantized block types from candle-core, create and fill them via the public `GgmlType` trait methods (`zeros()` for allocation, `from_float()` for quantization) rather than trying to access fields directly or use raw byte manipulation. The `vec_dot()` method also requires the element count `n` to be a multiple of `QK8_0` (32); always pad input to `ceil(k/32)*32` before quantizing.
**Source:** qgemv implementation

---
## Lesson #24 — 2026-06-15
**Trigger:** `ast_grep_replace` tool reported "APPLIED 28 replacements" but the file content on disk was never actually changed.
**Rule:** When `ast_grep_replace` does not persist changes to disk, fall back to using the `edit` tool with `replaceAll=true` for bulk string replacements. Verify file content after every replace operation by reading a line.
**Source:** q8_linear ws-parameter migration

---
## Lesson #27 — 2026-06-15
**Trigger:** The codec decode stage (~35s for 8 calls) dominates total pipeline time, not the talker or predictor.
**Rule:** Always add end-to-end timing before optimizing. The perceived bottleneck (talker/predictor) may not be the real one — the codec decoder can dominate. Use per-stage timing to guide optimization priority.
**Source:** stage timing implementation (sub-task 1)

---
## Lesson #26 — 2026-06-15
**Trigger:** The predictor's `head_dim_sum` (= n_q_heads * head_dim = 2048) was mistakenly used as both the attention output dimension AND the FFN hidden intermediate dimension, but FFN hidden was 3072 (ffn_gate.out_features()).
**Rule:** In GQA + SwiGLU architectures, do not assume attention and FFN dimensions are the same. Compute `attn_dim = n_q_heads * head_dim` for attention output, and `ffn_dim = ffn_gate.out_features()` for the SwiGLU intermediate — these can differ (2048 vs 3072 in Qwen2-TTS 1.7B predictor).
**Source:** code_predictor dimension fix

---
## Lesson #28 — 2026-06-15
**Trigger:** Ported 778-line C++ prompt-builder.h to Rust prompt.rs with 5 modes, prefix cache, and GGUF metadata parsing. Required adding accessor methods to Talker and a new `embed_frame()` to CodePredictor.
**Rule:** When porting a complex C++ builder to Rust, implement it as a free function (not a struct with lifetime-bound borrow) when it only needs read-only access to config data. Pass `&Talker` as parameter to avoid Pipeline lifetime complications. Store pre-computed special embeddings in a `PromptCache` to avoid redundant projections on repeated calls with the same parameters.
**Source:** feat(prompt): full C++ prompt-builder port

---
## Lesson #25 — 2026-06-15
**Trigger:** Custom attention_tensor at small cache sizes (kv_len<5) was slower than candle's native matmul due to to_vec1 copy overhead dominating the tiny computation.
**Rule:** When replacing tensor ops with custom f32 ops, benchmark at representative sizes (not just microbenchmark). For tiny tensors (<10KB), tensor dispatch overhead is negligible and vanilla candle ops may be faster. For larger tensors (>100KB) or operations that scale with data size (attention grows with kv_len), custom f32 ops win due to fusion eliminating intermediate allocations.
**Source:** qgemv GQA custom attention

---
## Lesson #29 — 2026-06-15
**Trigger:** KV cache hot path used Tensor::cat (O(n²) cumulative copies). Replaced with pre-allocated Vec<f32> flat buffers + strided attention_f32 to avoid per-step Tensor→f32 cache copies.
**Rule:** To eliminate O(n²) KV cache append cost: (1) allocate max-sized flat f32 buffers per layer, memcpy new K/V at current position (O(1) per step per head); (2) add `head_stride` param to attention_f32 so it can index into pre-allocated (non-compact) buffers with gaps; (3) persist Q8Workspace as a Talker field to avoid Vec::new per forward_step call. The key saving is avoiding `Tensor::to_vec1` on the growing KV cache (one full-cache copy per layer per step).
**Source:** feat(talker): pre-allocated KV cache + persistent Q8Workspace

---
## Lesson #30 — 2026-06-15
**Trigger:** forward_step had ~10 Tensor round-trips per layer (reshape, permute, narrow, from_slice, to_vec1). The fused f32 path replaced them all with direct f32 ops on pre-allocated buffers.
**Rule:** When implementing a fused CPU forward path for M=1 inference: (1) store norm weight data as `Vec<f32>` at load time (from `RmsNorm.weight().to_vec1()`) to avoid Tensor indirection on the hot path; (2) pre-allocate scratch buffers in a `FusedScratch` struct — `normed` (d_model, reused as main state), `residual` (d_model), and `ffn_mid` (ffn_dim); (3) call `Q8Weights::gemv` directly on raw `&[f32]` slices instead of going through the Tensor-wrapping `q8_linear`; (4) compute RoPE table as a flat `Vec<f32>` with `[max_seq, head_dim]` layout, indexed by position — avoid Tensor `{cos,sin}` calls by computing f64 angles directly. The remaining allocation on the fused path is the final output Vec (one `rms_norm_f32` call per step) plus each `gemv` result Vec (inevitable O(output_dim) allocation).
**Source:** feat(talker): add fused M=1 CPU forward path (forward_step_fused)

---
## Lesson #31 — 2026-06-15
**Trigger:** `forward_step_fused` used `cfg.max_seq_len` (model config, 32768) as `head_stride` for `attention_f32`, but the KV cache was created with a smaller `max_seq` (2048), causing out-of-bounds access at `k[k_base + t*hd]` where `k_base = kv_h * head_stride * hd` overflowed the cache buffer.
**Rule:** When passing a pre-allocated flat buffer's stride/layout parameter to a downstream function, derive the stride from the buffer itself (via a getter method), not from a configuration value that could diverge. For `KvCacheFlat`, add a `head_stride()` method that returns `self.max_seq` and use that everywhere instead of a separately-obtained `max_seq_len`.
**Source:** feat: benchmark fused f32 path

---
## Lesson #32 — 2026-06-15
**Trigger:** `PredScratch::q_buf` was sized `pred_hidden` (1024) and reused across `attn_q` (output `attn_dim` = 2048), `attn_o` (output `pred_hidden` = 1024), and `ffn_down` (output `pred_hidden` = 1024). But `gemv_into` requires `dst.len() == self.n` exactly — cannot pass a larger buffer.
**Rule:** When a scratch buffer must serve multiple `gemv_into` calls with different output dimensions, allocate separate buffers for each distinct size. Do not try to size `max(n1, n2)` unless every caller writes with the same `n`. Prefer `attn_q_buf: Vec<f32>` (sized attn_dim) vs `q_buf: Vec<f32>` (sized pred_hidden) over a single unified buffer.
**Source:** feat: benchmark fused f32 path

---
## Lesson #33 — 2026-06-15
**Trigger:** Multi-level parallel optimization (QKV fusion, head-level par, element-wise par) improved talker by 8.8% (68→63ms/step) but predictor stayed unchanged. The bottleneck was f32 matmul (project_f32, apply_lm_head_f32), not Q8 ops.
**Rule:** Before adding parallelism, profile to identify the actual bottleneck. Q8 GEMV is memory-bandwidth-bound — adding more CPU cores to a memory-bound op yields diminishing returns. Head-level parallelism helps when head_dim ops are compute-bound (norm, RoPE, attention) but does not help when the bottleneck is reading weight data from DRAM (gemv). Check compute-to-memory ratio before deciding where to add threads.
**Source:** 2026-06-15-multi-level-parallel-optimization

---
## Lesson #34 — 2026-06-15
**Trigger:** Predictor's f32 matmul layers (project_f32, apply_lm_head_f32, embed_codec_f32) were bottlenecked at 250ms/frame. Adding parallelism (Lesson #33) didn't help. Converting the three weight matrices (mtp_proj: 1024×2048, lm_heads: 15×2048×1024, codec_embd: 14×2048×2048) from f32 to Q8_0 via `Q8Weights::from_f32_data` and using `gemv_into` / `gemv_into_quantized` cut predictor time 2.9× (250ms→85ms/frame).
**Rule:** For memory-bandwidth-bound f32 matmul where each output element reads every input element once, Q8_0 quantization (4 bytes/weight → ~1 byte/weight) reduces DRAM traffic by ~75%, translating to ~3× speedup on CPU. Combine with codec_embd Q8 row lookup fused into mtp_proj's `gemv_into_quantized` to skip the f32 dequantize+project round-trip entirely. GGML's design principle — solve the memory bandwidth wall — applies equally to custom Rust Q8 GEMV implementations.
**Source:** perf(predictor): Q8_0 quantize mtp_proj, lm_heads, codec_embd
---
## Lesson #1 — 2026-06-15
**Trigger:** Batched predictor M=1 produced different output than sequential M=1. Debugging revealed k_cache_batched was used for both key and value slices.
**Rule:** When building multi-cache structures for batched attention, double-check each slice variable reads from the correct cache. A copy-paste error (k → v) silently produces wrong attention output with no runtime error.
**Source:** Phase 5 bug fix — forward_at_pos_batched v_slices

---
## Lesson #2 — 2026-06-15
**Trigger:** Believed `from_gguf` had a shape bug (n/k swapped) for non-square Q8 weights. Spent hours analyzing and "fixing", but the original code `n=shape[0]; k=shape[1]` was always correct. candle_core reverses GGUF dimensions on read (gguf_file.rs:438): GGUF file stores `[n_in, n_out]` (innermost=n_in=k), after reversal → `[n_out, n_in]`. So shape[0] = n_out (rows), shape[1] = n_in (cols, contiguous for Q8 blocks). The assertion failures were from NEW code (forward_at_pos_batched using wrong bpr), not the old from_gguf.
**Rule:** Before "fixing" a suspected bug in working code, verify by reading the ACTUAL candle source (gguf_file.rs write path writes `dims.iter().rev()`; read path reverses back). Then run a WITH-model equivalence test BEFORE making code changes. If the existing code loads models correctly (model run doesn't panic), the load-time shape reading is likely correct — the bug is probably in NEW code that consumes the values. Never assume a working function is buggy without first proving the bug with a model-grounded test.
**Source:** Phase 6 shape fix revert
