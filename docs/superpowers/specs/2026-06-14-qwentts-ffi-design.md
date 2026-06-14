# qwentts.cpp FFI — Design Spec

## Goal

Replace the subprocess-based `ExternalQwenTtsBackend` with direct in-process FFI
calls into the qwentts.cpp shared library via a safe Rust wrapper.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  crates/qwentts-sys                                         │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │ build.rs         │  │ src/                             │ │
│  │  cmake auto-build│  │  ├── lib.rs  (bindgen)           │ │
│  │  ← prebuilt DLL  │  │  └── safe.rs (safe wrapper)     │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
│  ┌──────────────────┐                                       │
│  │ wrapper.h        │  #include "src/qwen.h"               │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
          │  (safe Rust API)
          ▼
┌─────────────────────────────────────────────────────────────┐
│  crates/runtime                                   (new)     │
│  ┌────────────────────────────────────┐                     │
│  │ src/ffi_backend.rs                │                     │
│  │  struct FfiBackend                │                     │
│  │  impl RuntimeBackend              │                     │
│  └────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  crates/cli                                    (modified)   │
│  ┌────────────────────────────────────┐                     │
│  │ --backend {native-cpu,qwentts,ffi}│                     │
│  │ FfiBackend 註冊進 Scheduler       │                     │
│  └────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## C ABI Surface (from `qwen.h`)

All symbols are `extern "C"` with `QT_API` visibility.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `qt_version` | `const char *` | Build identity string |
| `qt_last_error` | `const char *` | Thread-local error message |
| `qt_init` | `qt_context * (const qt_init_params *)` | Create context |
| `qt_free` | `void (qt_context *)` | Destroy context |
| `qt_init_default_params` | `void (qt_init_params *)` | Fill default init params |
| `qt_tts_default_params` | `void (qt_tts_params *)` | Fill default tts params |
| `qt_synthesize` | `qt_status (qt_context *, const qt_tts_params *, qt_audio *)` | Run TTS |
| `qt_audio_free` | `void (qt_audio *)` | Free audio buffer |
| `qt_num_codebooks` | `int (const qt_context *)` | Get K for codec |
| `qt_n_speakers` | `int (const qt_context *)` | Speaker count |
| `qt_speaker_name` | `const char * (const qt_context *, int)` | Speaker name by index |
| `qt_duration_sec_to_tokens` | `int (const qt_context *, float)` | Seconds → frames |
| `qt_log_set` | `void (qt_log_cb, void *)` | Install log callback |

Structs: `qt_init_params`, `qt_tts_params`, `qt_audio`, `qt_context` (opaque).

Enum: `qt_status` (0 = OK, negative = error).

Callbacks: `qt_cancel_cb`, `qt_audio_chunk_cb`, `qt_log_cb`.

**Convention**: all fields zero-initialised before first use; `abi_version`
stays first in every struct for forward-compat.

---

## Hybrid Build Strategy (`build.rs`)

```
1. Prebuilt DLL search (in order):
   a. dist/qwen.dll               (Windows)
   b. vendor/qwentts.cpp/build/Release/qwen.dll
   c. vendor/qwentts.cpp/build/libqwen.so      (Linux)
   d. vendor/qwentts.cpp/build/libqwen.dylib   (macOS)

2. If not found → CMake auto-build:
   cmake -S vendor/qwentts.cpp -B vendor/qwentts.cpp/build
         -DCMAKE_BUILD_TYPE=Release -DQWEN_SHARED=ON
   cmake --build vendor/qwentts.cpp/build --config Release --target qwen

3. If CMake also fails → emit cargo:warning=, skip link step.
   safe wrapper returns RuntimeBackend::is_available() = false.
```

`build.rs` emits:
- `cargo:rustc-link-lib=qwen`
- `cargo:rustc-link-search=<dir>`
- `cargo:rerun-if-changed=vendor/qwentts.cpp/src/qwen.h`

---

## Raw FFI (`src/lib.rs`)

Generated via `bindgen` from `wrapper.h` (single `#include "src/qwen.h"`).

Output: all extern functions + POD struct definitions + enum constants.

Re-export everything publicly so safe wrapper can reference.

---

## Safe Wrapper (`src/safe.rs`)

```rust
/// Opaque handle to a fully initialised TTS engine.
pub struct QwenTts { inner: NonNull<qt_context> }

/// Output of a synthesis call.
pub struct SynthesisOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Parameters mirroring qt_tts_params with Rust-friendly types.
pub struct TtsParams {
    pub text: String,
    pub lang: Option<String>,        // None → "auto"
    pub speaker: Option<String>,
    pub seed: i64,                    // -1 → random
    pub max_new_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl QwenTts {
    pub fn new(talker_path: &str, codec_path: &str) -> Result<Self, String>;
    pub fn synthesize(&self, params: &TtsParams) -> Result<SynthesisOutput, String>;
    pub fn num_codebooks(&self) -> i32;
    pub fn n_speakers(&self) -> i32;
    pub fn speaker_name(&self, index: i32) -> Option<String>;
}

impl Drop for QwenTts { fn drop(&mut self) { qt_free(self.inner.as_ptr()); } }
```

### Error handling

All `qt_status` → negative → `Err(qt_last_error())`.
`qt_init` returning NULL → `Err(qt_last_error())`.

---

## FfiBackend (`crates/runtime/src/ffi_backend.rs`)

```rust
pub struct FfiBackend {
    qwen_tts: QwenTts,       // created once at construction
    device: DeviceKind,
}

impl RuntimeBackend for FfiBackend {
    fn name(&self) -> &'static str { "qwentts-ffi" }
    fn device_kind(&self) -> DeviceKind { self.device }
    fn is_available(&self) -> bool { true }  // constructed → available

    fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        1. Convert SynthesisRequest → TtsParams
        2. self.qwen_tts.synthesize(params) → Vec<f32>
        3. Write WAV to request.out_path via hound
        4. validate_wav_file(...)
        5. Return SynthesisResponse
    }
}
```

**Construction**: `FfiBackend::new(talker_path, codec_path, device)` calls
`QwenTts::new(...)` once; fail-fast if DLL missing or init fails.

---

## CLI Integration

Add variant to `BackendMode`:

```rust
#[derive(ValueEnum)]
enum BackendMode {
    NativeCpu,
    Qwentts,
    Ffi,
}
```

`--backend ffi` path:

```rust
BackendMode::Ffi => {
    let ffi = FfiBackend::new(&talker, &codec, args.device)
        .map_err(|e| format!("FFI init failed: {e}"))?;
    scheduler.register(ffi);
}
```

Default stays `native-cpu`; the user explicitly opts in with `--backend ffi`.

---

## Files

### New
| File | Purpose |
|------|---------|
| `crates/qwentts-sys/Cargo.toml` | Crate manifest, dep: bindgen (build) |
| `crates/qwentts-sys/build.rs` | Hybrid cmake/prebuilt DLL |
| `crates/qwentts-sys/wrapper.h` | `#include "src/qwen.h"` |
| `crates/qwentts-sys/src/lib.rs` | bindgen raw FFI |
| `crates/qwentts-sys/src/safe.rs` | Safe Rust wrapper |
| `crates/runtime/src/ffi_backend.rs` | FfiBackend impl RuntimeBackend |

### Modified
| File | Change |
|------|--------|
| `Cargo.toml` | Add `crates/qwentts-sys` to workspace members |
| `crates/runtime/Cargo.toml` | Add dep: `qwen-tts-sys` |
| `crates/runtime/src/lib.rs` | Re-export `FfiBackend` |
| `crates/cli/Cargo.toml` | Add dep on `qwen-tts-runtime` (already) |
| `crates/cli/src/main.rs` | New `BackendMode::Ffi`, register `FfiBackend` |

---

## Error Handling & Edge Cases

| Situation | Behaviour |
|-----------|-----------|
| DLL not found at build time | `build.rs` warning, safe wrapper `new()` returns `Err` |
| DLL not found at runtime | `QwenTts::new()` returns `Err("...")` |
| `qt_init` returns NULL | `QwenTts::new()` returns `Err(qt_last_error())` |
| `qt_synthesize` fails | `Err` with `qt_last_error()` message |
| Audio output empty | still writes WAV (0-length), validation surfaces it |
| Null speaker/text/lang | Safe wrapper substitutes empty string / "auto" |

---

## Testing

| Level | What | How |
|-------|------|-----|
| Unit | safe wrapper params conversion | `#[cfg(test)]` in `safe.rs` |
| Unit | error path: NULL init, bad paths | Skip if no DLL; gated by `#[cfg(feature = "integration_test")]` |
| Integration | FfiBackend synthesize | Needs real GGUF + DLL; manual only |
| CI | workspace check | `cargo check --workspace` (no DLL needed) |
| CI | workspace test | `cargo test --workspace --exclude qwen-tts-sys` |

---

## Non-Goals

- No streaming audio callback support in v1 (always buffered).
- No cancellation support in v1.
- No standalone codec decode (the Rust crate `qwen-tts-codec` already covers that).
- No replacement of `ExternalQwenTtsBackend` (both coexist).
