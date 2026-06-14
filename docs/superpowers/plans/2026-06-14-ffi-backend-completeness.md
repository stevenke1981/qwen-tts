# FfiBackend Feature Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the feature gap between `FfiBackend` and `ExternalQwenTtsBackend` by wiring all C API synthesis parameters through the safe wrapper, runtime, and CLI.

**Architecture:** Incremental — each task adds one logical group of parameters. The stack is `qwen.h` C API → `raw.rs` (FFI) → `safe.rs` (wrapper) → `ffi_backend.rs` (runtime) → `main.rs` (CLI). For each group we add typing in `SynthesisRequest` as needed, pipe through `FfiBackend::synthesize`, and expose CLI flags.

**Tech Stack:** Rust, FFI, clap, qwentts.cpp v2 ABI

---

### Task 1: Wire `speaker` through FfiBackend

**Background:** `SynthesisRequest.speaker: Option<String>` already exists. `ExternalQwenTtsBackend` uses it via `--speaker`. The CLI already has `--speaker`. But `FfiBackend::synthesize` ignores it entirely.

**Files:**
- Modify: `crates/runtime/src/ffi_backend.rs` — lines ~100-106

- [ ] **Step 1: Add speaker wiring in FfiBackend::synthesize**

After the `lang_cstr` block and before setting params, add:

```rust
let speaker_cstr = match &request.speaker {
    Some(s) => Some(
        CString::new(s.as_str())
            .map_err(|_| BackendError::InvalidRequest("speaker contains NUL".into()))?,
    ),
    None => None,
};
```

Then set `params.speaker`:

```rust
params.speaker = speaker_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p qwen-tts-runtime --features ffi`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add crates/runtime/src/ffi_backend.rs
git commit -m "feat(ffi): wire speaker param through FfiBackend"
```

---

### Task 2: Add `instruct` parameter across the stack

**Background:** The C API has `qt_tts_params.instruct: *const c_char`. `ExternalQwenTtsBackend` does NOT support it (the `qwen-tts` CLI binary may not have `--instruct`), but the C API does. Adding it enables talker instruction/guidance.

**Files:**
- Add to: `crates/runtime/src/backend.rs` — `SynthesisRequest` struct
- Modify: `crates/runtime/src/ffi_backend.rs` — wire through
- Modify: `crates/cli/src/main.rs` — add `--instruct` flag, pass in request

- [ ] **Step 1: Add `instruct` field to SynthesisRequest**

In `crates/runtime/src/backend.rs`, add to `SynthesisRequest`:

```rust
pub instruct: Option<String>,
```

- [ ] **Step 2: Wire `instruct` in FfiBackend**

In `ffi_backend.rs`, after the `speaker_cstr` block, add:

```rust
let instruct_cstr = match &request.instruct {
    Some(s) => Some(
        CString::new(s.as_str())
            .map_err(|_| BackendError::InvalidRequest("instruct contains NUL".into()))?,
    ),
    None => None,
};
```

Then set params:

```rust
params.instruct = instruct_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
```

- [ ] **Step 3: Add `--instruct` CLI flag**

In `SynthArgs` in `main.rs`:

```rust
#[arg(long)]
instruct: Option<String>,
```

- [ ] **Step 4: Pass instruct in request**

In the `synth()` function, add to `SynthesisRequest` construction:

```rust
instruct: args.instruct.clone(),
```

- [ ] **Step 5: Compile check**

Run: `cargo check -p qwen-tts-runtime -p qwen-tts-cli --features ffi`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add crates/runtime/src/backend.rs crates/runtime/src/ffi_backend.rs crates/cli/src/main.rs
git commit -m "feat(ffi): add instruct param through runtime and CLI"
```

---

### Task 3: Add init-time `use_fa` and `clamp_fp16` params

**Background:** The C init params struct has two boolean flags: `use_fa` (enable flash attention for GPU backends) and `clamp_fp16` (clamp fp16 values to avoid NaN artifacts). Currently FfiBackend uses the C defaults (zero-initialized = false for both).

**Files:**
- Modify: `crates/runtime/src/ffi_backend.rs` — `FfiBackend` struct, `new()`, `synthesize()`
- Modify: `crates/cli/src/main.rs` — SynthArgs

- [ ] **Step 1: Add fields to FfiBackend struct**

```rust
pub struct FfiBackend {
    pub talker_path: PathBuf,
    pub codec_path: PathBuf,
    pub device: DeviceKind,
    pub use_flash_attn: bool,
    pub clamp_fp16: bool,
}
```

- [ ] **Step 2: Update FfiBackend::new()**

```rust
pub fn new(
    talker_path: impl Into<PathBuf>,
    codec_path: impl Into<PathBuf>,
    device: DeviceKind,
) -> Self {
    Self {
        talker_path: talker_path.into(),
        codec_path: codec_path.into(),
        device,
        use_flash_attn: false,
        clamp_fp16: false,
    }
}
```

- [ ] **Step 3: Use the fields in synthesize()**

Before `QwenTts::new`, set:

```rust
init.use_fa = self.use_flash_attn;
init.clamp_fp16 = self.clamp_fp16;
```

- [ ] **Step 4: Add CLI flags**

In `SynthArgs`:

```rust
#[arg(long)]
flash_attention: bool,

#[arg(long)]
clamp_fp16: bool,
```

- [ ] **Step 5: Wire through CLI synth()**

When constructing `FfiBackend`:

```rust
let mut ffi_bk = FfiBackend::new(talker.clone(), codec.clone(), args.device);
ffi_bk.use_flash_attn = args.flash_attention;
ffi_bk.clamp_fp16 = args.clamp_fp16;
scheduler.register(ffi_bk);
```

- [ ] **Step 6: Compile check**

```bash
cargo check -p qwen-tts-runtime -p qwen-tts-cli --features ffi
```

- [ ] **Step 7: Commit**

```bash
git add crates/runtime/src/ffi_backend.rs crates/cli/src/main.rs
git commit -m "feat(ffi): add use_flash_attn and clamp_fp16 init params"
```

---

### Task 4: Add sampling parameters (seed, temperature, top_k, top_p, etc.)

**Background:** The C API exposes rich sampling controls. These are advanced but common TTS tuning knobs. We'll add optional fields to `SynthesisRequest` and pipe through.

**Parameters to add:**
- `seed: Option<i64>` (deterministic generation)
- `max_new_tokens: Option<i32>` (max output length)
- `temperature: Option<f32>` (main talker randomness)
- `top_k: Option<i32>` (main talker top-k filtering)
- `top_p: Option<f32>` (main talker nucleus sampling)
- `repetition_penalty: Option<f32>`
- `do_sample: Option<bool>` (default true in C)

**Files:**
- Modify: `crates/runtime/src/backend.rs` — SynthesisRequest
- Modify: `crates/runtime/src/ffi_backend.rs` — wiring
- Modify: `crates/cli/src/main.rs` — CLI flags

- [ ] **Step 1: Add sampling fields to SynthesisRequest**

```rust
pub seed: Option<i64>,
pub max_new_tokens: Option<i32>,
pub temperature: Option<f32>,
pub top_k: Option<i32>,
pub top_p: Option<f32>,
pub repetition_penalty: Option<f32>,
pub do_sample: Option<bool>,
```

- [ ] **Step 2: Wire in FfiBackend**

After the CString setup block, add a helper function or inline code to apply optional params:

```rust
if let Some(seed) = request.seed {
    params.seed = seed;
}
if let Some(tokens) = request.max_new_tokens {
    params.max_new_tokens = tokens;
}
if let Some(temp) = request.temperature {
    params.temperature = temp;
}
if let Some(k) = request.top_k {
    params.top_k = k;
}
if let Some(p) = request.top_p {
    params.top_p = p;
}
if let Some(rp) = request.repetition_penalty {
    params.repetition_penalty = rp;
}
if let Some(sample) = request.do_sample {
    params.do_sample = sample;
}
```

**Note:** The C API `qt_tts_default_params` sets sensible defaults (`do_sample=true`, `temperature=0.75`, `top_k=40`, `top_p=0.95`, etc.), so we only overwrite fields the user explicitly sets.

- [ ] **Step 3: Add CLI flags to SynthArgs**

```rust
#[arg(long)]
seed: Option<i64>,
#[arg(long)]
max_tokens: Option<i32>,
#[arg(long)]
temperature: Option<f32>,
#[arg(long)]
top_k: Option<i32>,
#[arg(long)]
top_p: Option<f32>,
#[arg(long)]
repetition_penalty: Option<f32>,
#[arg(long)]
no_sample: bool,  // negated: --no-sample sets do_sample=false
```

- [ ] **Step 4: Wire CLI to request**

```rust
seed: args.seed,
max_new_tokens: args.max_tokens,
temperature: args.temperature,
top_k: args.top_k,
top_p: args.top_p,
repetition_penalty: args.repetition_penalty,
do_sample: if args.no_sample { Some(false) } else { None },
```

Also add to `ExternalQwenTtsBackend` construction in the non-FFI path so tests still compile. Actually, `ExternalQwenTtsBackend` uses a builder pattern for request, so new fields need defaults in all match arms.

Wait — looking at the code again, `SynthesisRequest` is constructed in `synth()` function only, not by ExternalQwenTtsBackend. The request is created once and passed to whichever backend. So we just need to add the fields with appropriate defaults.

For the existing test that constructs a request — we need to update it. Let me check if there are tests that construct `SynthesisRequest` directly.

Actually, `SynthesisRequest` doesn't have tests that construct it. It's only constructed in `synth()` in main.rs. So adding fields with `None` defaults in the struct definition should be fine, and we just need to update the one construction site.

- [ ] **Step 5: Compile check**

```bash
cargo check -p qwen-tts-runtime -p qwen-tts-cli --features ffi
```

- [ ] **Step 6: Commit**

```bash
git add crates/runtime/src/backend.rs crates/runtime/src/ffi_backend.rs crates/cli/src/main.rs
git commit -m "feat(ffi): add sampling params (seed, temp, top-k/p, rep-penalty)"
```

---

### Task 5: Tests

**Background:** Ensure the new parameters parse correctly from CLI and propagate correctly.

**Files:**
- Modify: `crates/cli/src/main.rs` — add test for all new flags

- [ ] **Step 1: Add CLI test for new Ffi flags**

After `parses_synth_backend_ffi`, add:

```rust
#[cfg(feature = "ffi")]
#[test]
fn parses_synth_ffi_advanced_params() {
    let cli = parse([
        "qwen-tts", "synth",
        "--text", "hello",
        "--backend", "ffi",
        "--speaker", "ann",
        "--instruct", "speak slowly",
        "--seed", "42",
        "--temperature", "0.9",
        "--top-k", "50",
        "--top-p", "0.9",
        "--repetition-penalty", "1.2",
        "--max-tokens", "512",
        "--no-sample",
        "--flash-attention",
        "--clamp-fp16",
    ]);

    let Command::Synth(args) = cli.command else {
        panic!("expected synth command");
    };
    assert_eq!(args.backend, BackendMode::Ffi);
    assert_eq!(args.speaker, Some(String::from("ann")));
    assert_eq!(args.instruct, Some(String::from("speak slowly")));
    assert_eq!(args.seed, Some(42));
    assert_eq!(args.temperature, Some(0.9));
    assert_eq!(args.top_k, Some(50));
    assert_eq!(args.top_p, Some(0.9));
    assert_eq!(args.repetition_penalty, Some(1.2));
    assert_eq!(args.max_tokens, Some(512));
    assert!(args.no_sample);
    assert!(args.flash_attention);
    assert!(args.clamp_fp16);
}
```

- [ ] **Step 2: Run test**

```bash
cargo test -p qwen-tts-cli --features ffi -- parses_synth_ffi_advanced_params --nocapture
```

Expected: PASS

- [ ] **Step 3: Full test pass**

```bash
cargo test -p qwen-tts-cli --features ffi
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/cli/src/main.rs
git commit -m "test(ffi): add CLI tests for advanced FFI params"
```

---

## Definition of Done
- [ ] `FfiBackend` passes `speaker` from `SynthesisRequest` to C API
- [ ] `instruct` added to `SynthesisRequest`, wired through `FfiBackend`, exposed as `--instruct`
- [ ] `use_flash_attn` and `clamp_fp16` configurable on `FfiBackend` and via `--flash-attention` / `--clamp-fp16`
- [ ] Sampling params (seed, temperature, top_k, top_p, repetition_penalty, do_sample, max_new_tokens) added to `SynthesisRequest`, wired through `FfiBackend`, exposed via CLI flags
- [ ] CLI test validates all new flags parse correctly
- [ ] `cargo check --workspace --features ffi` passes
