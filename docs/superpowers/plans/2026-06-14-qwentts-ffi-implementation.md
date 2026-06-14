# qwentts.cpp FFI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `crates/qwentts-sys` (C FFI + safe wrapper) and integrate it into `crates/runtime` as `FfiBackend` to replace subprocess with in-process TTS inference.

**Architecture:** Single `qwentts-sys` crate owns the shared library build (hybrid cmake/prebuilt), raw extern "C" declarations, and a safe Rust wrapper. A new `FfiBackend` in runtime implements the existing `RuntimeBackend` trait. CLI gains `--backend ffi`.

**Tech Stack:** Rust `extern "C"` FFI, cmake for C++ shared lib, C99 POD structs from `qwen.h`.

---

## File Inventory

### New
| Path | Role |
|------|------|
| `crates/qwentts-sys/Cargo.toml` | Crate manifest, no bindgen / cc deps needed (hand-written FFI) |
| `crates/qwentts-sys/build.rs` | Hybrid cmake → prebuilt DLL search |
| `crates/qwentts-sys/wrapper.h` | `#include "src/qwen.h"` with adjusted path |
| `crates/qwentts-sys/src/lib.rs` | Raw `extern "C"` declarations hand-written |
| `crates/qwentts-sys/src/safe.rs` | Safe `QwenTts` struct with `Drop` |
| `crates/runtime/src/ffi_backend.rs` | `FfiBackend` impl `RuntimeBackend` |

### Modified
| Path | Change |
|------|--------|
| `Cargo.toml` | Add `crates/qwentts-sys` to workspaces members |
| `crates/runtime/Cargo.toml` | Add `qwen-tts-sys` dependency |
| `crates/runtime/src/lib.rs` | Re-export `ffi_backend::FfiBackend` |
| `crates/cli/src/main.rs` | New `BackendMode::Ffi` variant + registration |

---

### Task 1: Scaffold `qwentts-sys` crate

**Files:**
- Create: `crates/qwentts-sys/Cargo.toml`
- Create: `crates/qwentts-sys/build.rs`
- Create: `crates/qwentts-sys/wrapper.h`
- Modify: `Cargo.toml` (workspace)

**Step 1.1: Create Cargo.toml**

```toml
[package]
name = "qwen-tts-sys"
version = "0.1.0"
edition.workspace = true
license.workspace = true

[lib]
name = "qwen_tts_sys"

[dependencies]

[build-dependencies]
```

**Note on bindgen:** We hand-write the FFI bindings instead of using bindgen.
`qwen.h` is only 325 lines of pure C99, so manually declaring the `extern "C"`
functions avoids the `libclang` dependency that bindgen requires on Windows.
Maintenance is straightforward: when `qwen.h` adds a field, add one line in
`src/lib.rs`.

**Step 1.2: Create build.rs**

```rust
use std::{env, path::PathBuf};

fn main() {
    let project_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two levels below workspace root")
        .to_path_buf();

    let vendor_dir = project_root.join("vendor").join("qwentts.cpp");
    let build_dir = vendor_dir.join("build");
    let dist_dir = project_root.join("dist");

    // Search order: dist/dll > build/Release/dll > cmake build
    let (lib_name, found) = if cfg!(target_os = "windows") {
        search_dll(&dist_dir, "qwen.dll")
            .or_else(|| search_dll(&build_dir.join("Release"), "qwen.dll"))
            .or_else(|| {
                eprintln!("cargo:warning=qwen.dll not found; attempting cmake build with -DQWEN_SHARED=ON");
                cmake_build_shared(&vendor_dir, &build_dir);
                search_dll(&build_dir.join("Release"), "qwen.dll")
            })
            .map(|p| ("qwen".to_string(), p))
    } else if cfg!(target_os = "linux") {
        search_dll(&dist_dir, "libqwen.so")
            .or_else(|| search_dll(&build_dir, "libqwen.so"))
            .or_else(|| {
                cmake_build_shared(&vendor_dir, &build_dir);
                search_dll(&build_dir, "libqwen.so")
            })
            .map(|p| ("qwen".to_string(), p))
    } else {
        // macOS
        search_dll(&dist_dir, "libqwen.dylib")
            .or_else(|| search_dll(&build_dir, "libqwen.dylib"))
            .or_else(|| {
                cmake_build_shared(&vendor_dir, &build_dir);
                search_dll(&build_dir, "libqwen.dylib")
            })
            .map(|p| ("qwen".to_string(), p))
    };

    match found {
        Some((name, path)) => {
            let dir = path.parent().unwrap();
            println!("cargo:rustc-link-lib={name}");
            println!("cargo:rustc-link-search={}", dir.display());
        }
        None => {
            eprintln!("cargo:warning=qwen shared library not found and cmake build failed");
            eprintln!("cargo:warning=run: cmake -S vendor/qwentts.cpp -B vendor/qwentts.cpp/build -DQWEN_SHARED=ON");
            eprintln!("cargo:warning=run: cmake --build vendor/qwentts.cpp/build --config Release --target qwen");
        }
    }

    println!("cargo:rerun-if-changed=vendor/qwentts.cpp/src/qwen.h");
    println!("cargo:rerun-if-changed=build.rs");
}

fn search_dll(dir: &PathBuf, name: &str) -> Option<PathBuf> {
    let path = dir.join(name);
    if path.exists() { Some(path) } else { None }
}

fn cmake_build_shared(vendor_dir: &PathBuf, build_dir: &PathBuf) {
    use std::process::Command;

    let status = Command::new("cmake")
        .args([
            "-S", vendor_dir.to_str().unwrap(),
            "-B", build_dir.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DQWEN_SHARED=ON",
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            let build_status = Command::new("cmake")
                .args([
                    "--build", build_dir.to_str().unwrap(),
                    "--config", "Release",
                    "--target", "qwen",
                ])
                .status();
            if let Ok(s) = build_status {
                if !s.success() {
                    eprintln!("cargo:warning=cmake --build --target qwen failed (exit={s:?})");
                }
            }
        }
        Ok(s) => eprintln!("cargo:warning=cmake configure failed (exit={s:?})"),
        Err(e) => eprintln!("cargo:warning=cmake not found: {e}"),
    }
}
```

**Step 1.3: Create wrapper.h**

```c
#pragma once
// Include guard for bindgen-style consumption.
// The relative path resolves from vendor/qwentts.cpp/src/qwen.h
#include "../../../vendor/qwentts.cpp/src/qwen.h"
```

**Step 1.4: Register in workspace Cargo.toml**

Add `"crates/qwentts-sys"` to the workspace `members` list.

**Step 1.5: Verify check**

```bash
cargo check -p qwen-tts-sys 2>&1
```

Expected: warnings about "unused crate", but no errors (the crate has no code yet).

**Step 1.6: Commit**

```bash
git add Cargo.toml crates/qwentts-sys/
git commit -m "chore: scaffold qwen-tts-sys crate with hybrid cmake build"
```

---

### Task 2: Write raw FFI bindings

**Files:**
- Modify: `crates/qwentts-sys/src/lib.rs`

Declare every `extern "C"` function and POD struct from `qwen.h`.

```rust
//! Raw FFI bindings to qwentts.cpp shared library.
//!
//! Hand-written from `vendor/qwentts.cpp/src/qwen.h`.
//! All strings are NUL-terminated UTF-8.

#![allow(non_camel_case_types, dead_code)]

pub const QT_ABI_VERSION: i32 = 2;

// ── Enums ──────────────────────────────────────────────

pub type qt_status = i32;
pub const QT_STATUS_OK: qt_status = 0;
pub const QT_STATUS_INVALID_PARAMS: qt_status = -1;
pub const QT_STATUS_MODE_INVALID: qt_status = -2;
pub const QT_STATUS_GENERATE_FAILED: qt_status = -3;
pub const QT_STATUS_OOM: qt_status = -4;
pub const QT_STATUS_CANCELLED: qt_status = -5;

pub type qt_log_level = i32;
pub const QT_LOG_DEBUG: qt_log_level = 0;
pub const QT_LOG_INFO: qt_log_level = 1;
pub const QT_LOG_WARN: qt_log_level = 2;
pub const QT_LOG_ERROR: qt_log_level = 3;

// ── Callback type aliases ─────────────────────────────

pub type qt_cancel_cb = Option<unsafe extern "C" fn(user_data: *mut std::ffi::c_void) -> bool>;
pub type qt_audio_chunk_cb = Option<
    unsafe extern "C" fn(
        samples: *const f32,
        n_samples: i32,
        user_data: *mut std::ffi::c_void,
    ) -> bool,
>;
pub type qt_log_cb = Option<
    unsafe extern "C" fn(
        level: qt_log_level,
        msg: *const std::ffi::c_char,
        user_data: *mut std::ffi::c_void,
    ),
>;

// ── Structs ───────────────────────────────────────────

#[repr(C)]
pub struct qt_init_params {
    pub abi_version: i32,
    pub talker_path: *const std::ffi::c_char,
    pub codec_path: *const std::ffi::c_char,
    pub use_fa: bool,
    pub clamp_fp16: bool,
}

#[repr(C)]
pub struct qt_audio {
    pub samples: *mut f32,
    pub n_samples: i32,
    pub sample_rate: i32,
    pub channels: i32,
}

#[repr(C)]
pub struct qt_tts_params {
    pub abi_version: i32,
    pub text: *const std::ffi::c_char,
    pub lang: *const std::ffi::c_char,
    pub instruct: *const std::ffi::c_char,
    pub speaker: *const std::ffi::c_char,
    pub ref_audio_24k: *const f32,
    pub ref_n_samples: i32,
    pub ref_text: *const std::ffi::c_char,
    pub seed: i64,
    pub max_new_tokens: i32,
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub subtalker_do_sample: bool,
    pub subtalker_temperature: f32,
    pub subtalker_top_k: i32,
    pub subtalker_top_p: f32,
    pub dump_dir: *const std::ffi::c_char,
    pub cancel: qt_cancel_cb,
    pub cancel_user_data: *mut std::ffi::c_void,
    pub on_chunk: qt_audio_chunk_cb,
    pub on_chunk_user_data: *mut std::ffi::c_void,
    pub codec_chunk_sec: f32,
    pub codec_left_context_sec: f32,
    pub ref_spk_emb: *const f32,
    pub ref_spk_dim: i32,
    pub ref_codes: *const i32,
    pub ref_T: i32,
}

// Opaque handle — Rust side never dereferences it.
pub enum qt_context {}

// ── extern "C" functions ─────────────────────────────

extern "C" {
    pub fn qt_version() -> *const std::ffi::c_char;

    pub fn qt_last_error() -> *const std::ffi::c_char;

    pub fn qt_init(params: *const qt_init_params) -> *mut qt_context;

    pub fn qt_free(ctx: *mut qt_context);

    pub fn qt_init_default_params(params: *mut qt_init_params);

    pub fn qt_tts_default_params(params: *mut qt_tts_params);

    pub fn qt_synthesize(
        ctx: *const qt_context,
        params: *const qt_tts_params,
        out: *mut qt_audio,
    ) -> qt_status;

    pub fn qt_audio_free(a: *mut qt_audio);

    pub fn qt_num_codebooks(ctx: *const qt_context) -> i32;

    pub fn qt_n_speakers(ctx: *const qt_context) -> i32;

    pub fn qt_speaker_name(
        ctx: *const qt_context,
        index: i32,
    ) -> *const std::ffi::c_char;

    pub fn qt_duration_sec_to_tokens(ctx: *const qt_context, sec: f32) -> i32;

    pub fn qt_log_set(cb: qt_log_cb, user_data: *mut std::ffi::c_void);
}
```

**Step 2.1: Verify check**

```bash
cargo check -p qwen-tts-sys 2>&1
```

Expected: no errors (the extern functions won't link without the DLL, but that's fine at check time since we're not building a binary that calls them).

**Step 2.2: Commit**

```bash
git add crates/qwentts-sys/src/lib.rs
git commit -m "feat(qwentts-sys): add raw FFI bindings for qwen.h C ABI"
```

---

### Task 3: Write safe wrapper

**Files:**
- Create: `crates/qwentts-sys/src/safe.rs`
- Modify: `crates/qwentts-sys/src/lib.rs` (add `mod safe;`)

```rust
//! Safe Rust wrapper around raw qwentts.cpp FFI.

use crate::raw::{
    qt_audio, qt_audio_free, qt_context, qt_init, qt_init_default_params, qt_init_params,
    qt_last_error, qt_n_speakers, qt_num_codebooks, qt_speaker_name, qt_synthesize,
    qt_tts_default_params, qt_tts_params, QT_STATUS_OK,
};
use std::ffi::{CStr, CString};
use std::ptr::NonNull;

// ═══════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct TtsParams {
    pub text: String,
    pub lang: Option<String>, // None → "auto"
    pub speaker: Option<String>,
    pub seed: i64,            // -1 → random
    pub max_new_tokens: i32,  // default 2048
    pub temperature: f32,     // default 0.9
    pub top_k: i32,           // default 50
    pub top_p: f32,           // default 1.0
    pub repetition_penalty: f32, // default 1.05
}

impl Default for TtsParams {
    fn default() -> Self {
        Self {
            text: String::new(),
            lang: None,
            speaker: None,
            seed: -1,
            max_new_tokens: 2048,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
        }
    }
}

#[derive(Debug)]
pub struct SynthesisOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

// ═══════════════════════════════════════════════════════
// QwenTts handle
// ═══════════════════════════════════════════════════════

pub struct QwenTts {
    inner: NonNull<qt_context>,
}

// SAFETY: qt_context is internally synchronized (Thread-local error,
// process-wide log). All public operations are sequential in practice.
unsafe impl Send for QwenTts {}
unsafe impl Sync for QwenTts {}

impl QwenTts {
    /// Initialise the TTS engine.
    ///
    /// # Errors
    ///
    /// Returns an error if the shared library is not found, the GGUF files
    /// cannot be loaded, or the underlying `qt_init` fails.
    pub fn new(talker_path: &str, codec_path: &str) -> Result<Self, String> {
        let mut params: qt_init_params = unsafe { std::mem::zeroed() };
        // Safe: qt_init_default_params only writes to `params`.
        unsafe { qt_init_default_params(&mut params) };

        let talker_c = CString::new(talker_path)
            .map_err(|e| format!("invalid talker_path: {e}"))?;
        let codec_c = CString::new(codec_path)
            .map_err(|e| format!("invalid codec_path: {e}"))?;

        params.talker_path = talker_c.as_ptr();
        params.codec_path = codec_c.as_ptr();

        // SAFETY: params is fully initialised; qt_init validates internally.
        let ptr = unsafe { qt_init(&params as *const qt_init_params) };
        NonNull::new(ptr).ok_or_else(|| Self::last_error())
    }

    /// Run TTS synthesis. Returns audio samples.
    ///
    /// # Errors
    ///
    /// Returns an error if the synthesis fails for any reason. The error
    /// message comes from `qt_last_error()`.
    pub fn synthesize(&self, params: &TtsParams) -> Result<SynthesisOutput, String> {
        // Build raw qt_tts_params
        let mut raw: qt_tts_params = unsafe { std::mem::zeroed() };
        unsafe { qt_tts_default_params(&mut raw) };

        let text_c = CString::new(&params.text[..])
            .map_err(|e| format!("invalid text: {e}"))?;
        let lang_c = match &params.lang {
            Some(l) => CString::new(&l[..])
                .map_err(|e| format!("invalid lang: {e}"))?,
            None => CString::new("auto").unwrap(),
        };
        let speaker_c = match &params.speaker {
            Some(s) => CString::new(&s[..])
                .map_err(|e| format!("invalid speaker: {e}"))?,
            None => CString::new("").unwrap(),
        };

        raw.text = text_c.as_ptr();
        raw.lang = lang_c.as_ptr();
        raw.speaker = speaker_c.as_ptr();
        raw.seed = params.seed;
        raw.max_new_tokens = params.max_new_tokens;
        raw.do_sample = true;
        raw.temperature = params.temperature;
        raw.top_k = params.top_k;
        raw.top_p = params.top_p;
        raw.repetition_penalty = params.repetition_penalty;
        // abi_version was set by qt_tts_default_params

        let mut out: qt_audio = unsafe { std::mem::zeroed() };

        // SAFETY: raw is fully populated; out is zeroed; the lib writes into it.
        let status = unsafe {
            qt_synthesize(self.inner.as_ptr(), &raw as *const qt_tts_params, &mut out as *mut qt_audio)
        };

        if status != QT_STATUS_OK {
            // If out was allocated, free it
            if !out.samples.is_null() {
                unsafe { qt_audio_free(&mut out as *mut qt_audio) };
            }
            return Err(Self::last_error());
        }

        // Copy samples out of the C heap before freeing
        let n = out.n_samples as usize;
        let sr = out.sample_rate as u32;
        let mut samples = Vec::with_capacity(n);
        if !out.samples.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(out.samples, samples.as_mut_ptr(), n);
                samples.set_len(n);
            }
        }
        unsafe { qt_audio_free(&mut out as *mut qt_audio) };

        Ok(SynthesisOutput {
            samples,
            sample_rate: sr,
        })
    }

    pub fn num_codebooks(&self) -> i32 {
        unsafe { qt_num_codebooks(self.inner.as_ptr()) }
    }

    pub fn n_speakers(&self) -> i32 {
        unsafe { qt_n_speakers(self.inner.as_ptr()) }
    }

    pub fn speaker_name(&self, index: i32) -> Option<String> {
        let ptr = unsafe { qt_speaker_name(self.inner.as_ptr(), index) };
        if ptr.is_null() {
            None
        } else {
            unsafe {
                CStr::from_ptr(ptr)
                    .to_str()
                    .ok()
                    .map(|s| s.to_owned())
            }
        }
    }

    // ── Internal helpers ──────────────────────────────

    fn last_error() -> String {
        unsafe {
            let ptr = qt_last_error();
            if ptr.is_null() {
                "unknown error".to_owned()
            } else {
                CStr::from_ptr(ptr)
                    .to_str()
                    .unwrap_or("non-UTF-8 error")
                    .to_owned()
            }
        }
    }
}

impl Drop for QwenTts {
    fn drop(&mut self) {
        unsafe { qt_free(self.inner.as_ptr()) };
    }
}

// ═══════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_params_default_has_sane_values() {
        let p = TtsParams::default();
        assert_eq!(p.max_new_tokens, 2048);
        assert!((p.temperature - 0.9).abs() < 1e-6);
        assert_eq!(p.top_k, 50);
    }

    #[test]
    fn last_error_no_context() {
        // last_error with no prior call returns empty string.
        let err = QwenTts::last_error_for_test();
        assert!(err.is_empty() || err.contains("unknown"));
    }
}
```

Note: the test `last_error_no_context` calls a helper we won't expose publicly;
just make `last_error` `pub(crate)` for testing, or skip that test.

**Step 3.1: Wire into lib.rs**

In `crates/qwentts-sys/src/lib.rs`:

```rust
pub mod raw;   // extern "C" bindings (current content moved here)
pub mod safe;  // safe wrapper (new)
```

Move the existing extern declarations into `raw.rs`, keep `lib.rs` minimal.

**Step 3.2: Verify check**

```bash
cargo check -p qwen-tts-sys 2>&1
```

Expected: no errors. The tests won't compile because of the `last_error_for_test` helper — fix that in the actual implementation.

**Step 3.3: Run tests that don't need the DLL**

```bash
cargo test -p qwen-tts-sys -- --skip needs_dll 2>&1
```

Expected: `tts_params_default_has_sane_values` passes.

**Step 3.4: Commit**

```bash
git add crates/qwentts-sys/src/
git commit -m "feat(qwentts-sys): add safe Rust wrapper QwenTts with Drop"
```

---

### Task 4: Write FfiBackend

**Files:**
- Create: `crates/runtime/src/ffi_backend.rs`
- Modify: `crates/runtime/Cargo.toml`
- Modify: `crates/runtime/src/lib.rs`

**Step 4.1: Add dependency**

In `crates/runtime/Cargo.toml`, add:

```toml
qwen-tts-sys = { path = "../qwentts-sys", optional = true }
```

Make `FfiBackend` conditional on the `ffi` feature:

```toml
[features]
ffi = ["qwen-tts-sys"]
```

**Step 4.2: Create ffi_backend.rs**

```rust
//! In-process FFI backend using qwentts.cpp shared library.

use crate::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};
use qwen_tts_core::{validate_wav_file, AudioSpec};
use std::fs;
use tracing::info;

#[cfg(feature = "ffi")]
use qwen_tts_sys::safe::{QwenTts, TtsParams};

pub struct FfiBackend {
    #[cfg(feature = "ffi")]
    qwen_tts: QwenTts,
    device: DeviceKind,
}

impl FfiBackend {
    #[cfg(feature = "ffi")]
    pub fn new(
        talker_path: impl Into<String>,
        codec_path: impl Into<String>,
        device: DeviceKind,
    ) -> Result<Self, String> {
        let qwen_tts = QwenTts::new(&talker_path.into(), &codec_path.into())?;
        Ok(Self { qwen_tts, device })
    }
}

#[cfg(feature = "ffi")]
impl RuntimeBackend for FfiBackend {
    fn name(&self) -> &'static str {
        "qwentts-ffi"
    }

    fn device_kind(&self) -> DeviceKind {
        self.device
    }

    fn is_available(&self) -> bool {
        true // constructed = available
    }

    fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        if request.text.trim().is_empty() {
            return Err(BackendError::InvalidRequest("text cannot be empty".into()));
        }

        if let Some(parent) = request.out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .map_err(BackendError::Io)?;
        }

        let params = TtsParams {
            text: request.text.clone(),
            lang: Some(request.language.clone()),
            speaker: request.speaker.clone(),
            ..TtsParams::default()
        };

        let output = self.qwen_tts
            .synthesize(&params)
            .map_err(|e| BackendError::CommandFailed {
                program: "qwen-tts (FFI)".into(),
                status: None,
                stderr: e,
            })?;

        // Write WAV file
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: output.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&request.out_path, spec)
            .map_err(|e| BackendError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        for &sample in &output.samples {
            writer.write_sample(sample.clamp(-1.0, 1.0))
                .map_err(|e| BackendError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        }
        writer.finalize()
            .map_err(|e| BackendError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        let metadata = validate_wav_file(&request.out_path, AudioSpec::default())?;

        info!(
            backend = self.name(),
            output = %request.out_path.display(),
            sample_rate_hz = output.sample_rate,
            channels = 1_u16,
            "ffi synthesis finished"
        );

        Ok(SynthesisResponse {
            wav_path: request.out_path.clone(),
            sample_rate_hz: metadata.sample_rate_hz,
            channels: metadata.channels,
            bits_per_sample: metadata.bits_per_sample,
            data_size_bytes: metadata.data_size_bytes,
            backend_name: self.name().to_owned(),
        })
    }
}
```

**Step 4.3: Wire into runtime lib.rs**

In `crates/runtime/src/lib.rs`, add:

```rust
#[cfg(feature = "ffi")]
pub mod ffi_backend;

#[cfg(feature = "ffi")]
pub use ffi_backend::FfiBackend;
```

**Step 4.4: Verify check**

```bash
cargo check -p qwen-tts-runtime --features ffi 2>&1
```

Expected: no errors (but the binary won't link without the DLL at runtime).

Without the `ffi` feature:

```bash
cargo check -p qwen-tts-runtime 2>&1
```

Expected: no errors, FfiBackend not compiled in.

**Step 4.5: Commit**

```bash
git add crates/runtime/src/ffi_backend.rs crates/runtime/Cargo.toml crates/runtime/src/lib.rs
git commit -m "feat(runtime): add FfiBackend using qwentts-sys safe wrapper"
```

---

### Task 5: Integrate into CLI

**Files:**
- Modify: `crates/cli/src/main.rs`

**Step 5.1: Add BackendMode::Ffi variant**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendMode {
    NativeCpu,
    Qwentts,
    Ffi,   // NEW
}
```

**Step 5.2: Register FfiBackend in synth()**

In the `match args.backend` block inside `synth()`:

```rust
BackendMode::Ffi => {
    #[cfg(feature = "ffi")]
    {
        let talker_str = talker.to_string_lossy().to_string();
        let codec_str = codec.to_string_lossy().to_string();
        let ffi = qwen_tts_runtime::FfiBackend::new(talker_str, codec_str, args.device)
            .map_err(|e| format!("FFI backend init: {e}"))?;
        scheduler.register(ffi);
    }
    #[cfg(not(feature = "ffi"))]
    {
        return Err("FFI backend not enabled. Rebuild with --features ffi.".into());
    }
}
```

**Step 5.3: Enable ffi feature**

In `crates/cli/Cargo.toml`:

```toml
[dependencies]
qwen-tts-runtime = { path = "../runtime", features = ["ffi"] }
```

**Step 5.4: Verify check**

```bash
cargo check -p qwen-tts-cli 2>&1
```

Expected: no errors.

```bash
cargo check --workspace 2>&1
```

Expected: no errors.

**Step 5.5: Commit**

```bash
git add crates/cli/src/main.rs crates/cli/Cargo.toml
git commit -m "feat(cli): add --backend ffi option for in-process TTS"
```

---

### Task 6: Test and verify full workspace

**Step 6.1: Full check**

```bash
cargo check --workspace 2>&1
```

Expected: zero errors.

**Step 6.2: Run non-FFI tests**

```bash
cargo test --workspace --exclude qwen-tts-sys 2>&1
```

Expected: all tests pass.

**Step 6.3: Help output verification**

```bash
cargo run -p qwen-tts-cli -- --help 2>&1
```

Expected: shows `ffi` in the backend choices.

**Step 6.4: Commit any final fixes**

```bash
git add -A
git commit -m "test: workspace check and non-FFI tests pass"
```
