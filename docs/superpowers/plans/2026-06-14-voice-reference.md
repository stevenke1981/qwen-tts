# Voice Reference (Voice Cloning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable voice cloning by piping reference audio and transcript through the C API's `ref_audio_24k` / `ref_text` fields.

**Architecture:** 
- `ffi_backend.rs` gains a `read_wav_f32_mono()` helper (no new dependencies — hand-parses 16-bit PCM WAV)
- `SynthesisRequest` gets `ref_audio_path: Option<PathBuf>` and `ref_text: Option<String>`
- The f32 sample buffer is kept alive on the stack during the unsafe `qt_synthesize` call
- CLI adds `--ref-audio` and `--ref-text` flags

**Tech Stack:** Rust, FFI, WAV format parsing, qwentts.cpp v2 ABI

---

### Task 1: Add ref_audio/ref_text to SynthesisRequest

**Files:**
- Modify: `crates/runtime/src/backend.rs` — SynthesisRequest
- Modify: `crates/runtime/src/config.rs` — constructor
- Modify: `crates/runtime/src/scheduler.rs` — test constructor
- Modify: `crates/backends/cpu/src/lib.rs` — test constructor
- Modify: `crates/cli/src/main.rs` — constructor
- Modify: `crates/app/src/main.rs` — constructor

- [ ] **Step 1: Add fields to SynthesisRequest**

```rust
pub ref_audio_path: Option<PathBuf>,
pub ref_text: Option<String>,
```

- [ ] **Step 2: Update all 6 construction sites** with `ref_audio_path: None, ref_text: None,`

- [ ] **Step 3: Compile check**

```bash
cargo check --workspace --features ffi
```

- [ ] **Step 4: Commit**

```bash
git add crates/runtime/src/backend.rs crates/runtime/src/config.rs crates/runtime/src/scheduler.rs crates/backends/cpu/src/lib.rs crates/cli/src/main.rs crates/app/src/main.rs
git commit -m "feat: add ref_audio_path and ref_text to SynthesisRequest"
```

---

### Task 2: Wire through FfiBackend

**Files:**
- Modify: `crates/runtime/src/ffi_backend.rs`

- [ ] **Step 1: Add `read_wav_f32_mono` helper**

```rust
/// Read a 16-bit PCM mono WAV file into f32 samples normalized to [-1, 1].
///
/// Only supports the subset needed for voice reference: 16-bit PCM, 1 channel,
/// any sample rate (caller should ensure 24kHz for best results).
fn read_wav_f32_mono(path: &Path) -> BackendResult<Vec<f32>> {
    use std::fs;
    let data = fs::read(path).map_err(|e| {
        BackendError::InvalidRequest(format!("cannot read ref audio WAV: {e}"))
    })?;

    if data.len() < 44 {
        return Err(BackendError::InvalidRequest("ref audio WAV too small".into()));
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(BackendError::InvalidRequest("ref audio: not a WAV file".into()));
    }

    // Parse fmt chunk
    let mut offset = 12;
    let mut audio_format = None;
    let mut num_channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut data_offset = None;
    let mut data_size = None;

    loop {
        if offset + 8 > data.len() {
            return Err(BackendError::InvalidRequest("ref audio: truncated WAV header".into()));
        }
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        match chunk_id {
            b"fmt " => {
                if offset + 16 > data.len() {
                    return Err(BackendError::InvalidRequest("ref audio: truncated fmt chunk".into()));
                }
                audio_format = Some(u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()));
                num_channels = Some(u16::from_le_bytes(data[offset + 2..offset + 4].try_into().unwrap()));
                sample_rate = Some(u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()));
                bits_per_sample = Some(u16::from_le_bytes(data[offset + 14..offset + 16].try_into().unwrap()));
            }
            b"data" => {
                data_offset = Some(offset);
                data_size = Some(chunk_size);
            }
            _ => {}
        }
        offset += chunk_size;
        if chunk_id == b"data" {
            break;
        }
        if offset >= data.len() {
            return Err(BackendError::InvalidRequest("ref audio: data chunk not found".into()));
        }
    }

    let af = audio_format.ok_or_else(|| BackendError::InvalidRequest("ref audio: missing fmt chunk".into()))?;
    let ch = num_channels.ok_or_else(|| BackendError::InvalidRequest("ref audio: missing channels".into()))?;
    let bps = bits_per_sample.ok_or_else(|| BackendError::InvalidRequest("ref audio: missing bits_per_sample".into()))?;
    let d_off = data_offset.ok_or_else(|| BackendError::InvalidRequest("ref audio: missing data chunk".into()))?;
    let d_sz = data_size.unwrap_or(0);

    if af != 1 {
        return Err(BackendError::InvalidRequest(format!("ref audio: only PCM supported, got format {af}")));
    }
    if ch != 1 {
        return Err(BackendError::InvalidRequest(format!("ref audio: only mono supported, got {ch} channels")));
    }
    if bps != 16 {
        return Err(BackendError::InvalidRequest(format!("ref audio: only 16-bit PCM supported, got {bps} bits")));
    }

    let sample_bytes = (bps / 8) as usize;
    let total_samples = d_sz / sample_bytes;
    let mut samples = Vec::with_capacity(total_samples);

    for i in 0..total_samples {
        let byte_start = d_off + i * sample_bytes;
        if byte_start + sample_bytes > data.len() {
            break;
        }
        let raw = i16::from_le_bytes(data[byte_start..byte_start + 2].try_into().unwrap());
        samples.push(f32::from(raw) / i16::MAX as f32);
    }

    if samples.is_empty() {
        return Err(BackendError::InvalidRequest("ref audio: no samples found".into()));
    }

    info!(
        "loaded ref audio: {} samples, {} Hz",
        samples.len(),
        sample_rate.unwrap_or(0)
    );
    Ok(samples)
}
```

- [ ] **Step 2: Wire ref_audio and ref_text in synthesize()**

After the `instruct_cstr` block, add:

```rust
let ref_audio_samples: Option<Vec<f32>> = match &request.ref_audio_path {
    Some(path) => Some(read_wav_f32_mono(path)?),
    None => None,
};
let ref_text_cstr = match &request.ref_text {
    Some(s) => Some(
        CString::new(s.as_str())
            .map_err(|_| BackendError::InvalidRequest("ref_text contains NUL".into()))?,
    ),
    None => None,
};
```

Then set params after the other param assignments:

```rust
params.ref_audio_24k = ref_audio_samples.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
params.ref_n_samples = ref_audio_samples.as_ref().map_or(0, |v| v.len() as i32);
params.ref_text = ref_text_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
```

- [ ] **Step 3: Compile check**

```bash
cargo check -p qwen-tts-runtime --features ffi
```

- [ ] **Step 4: Commit**

```bash
git add crates/runtime/src/ffi_backend.rs
git commit -m "feat(ffi): wire ref_audio / ref_text for voice cloning"
```

---

### Task 3: Add CLI flags

**Files:**
- Modify: `crates/cli/src/main.rs`

- [ ] **Step 1: Add CLI flags to SynthArgs**

```rust
#[arg(long)]
ref_audio: Option<PathBuf>,
#[arg(long)]
ref_text: Option<String>,
```

- [ ] **Step 2: Wire in synth()**

```rust
ref_audio_path: args.ref_audio.clone(),
ref_text: args.ref_text.clone(),
```

- [ ] **Step 3: Compile & test**

```bash
cargo test -p qwen-tts-cli
```

- [ ] **Step 4: Commit**

```bash
git add crates/cli/src/main.rs
git commit -m "feat(cli): add --ref-audio and --ref-text flags"
```

---

### Task 4: Test

- [ ] **Step 1: Add CLI test for voice reference flags**

```rust
#[cfg(feature = "ffi")]
#[test]
fn parses_synth_ffi_voice_ref() {
    let cli = parse([
        "qwen-tts", "synth",
        "--text", "hello",
        "--backend", "ffi",
        "--ref-audio", "speaker.wav",
        "--ref-text", "original utterance",
    ]);

    let Command::Synth(args) = cli.command else {
        panic!("expected synth command");
    };
    assert_eq!(args.ref_audio, Some(PathBuf::from("speaker.wav")));
    assert_eq!(args.ref_text, Some(String::from("original utterance")));
}
```

- [ ] **Step 2: Run test**

```bash
cargo test -p qwen-tts-cli -- parses_synth_ffi_voice_ref --nocapture
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/cli/src/main.rs
git commit -m "test: add CLI test for voice reference flags"
```

---

## Definition of Done
- [ ] `SynthesisRequest` has `ref_audio_path` and `ref_text` fields
- [ ] `FfiBackend::synthesize` loads WAV, sets `ref_audio_24k`/`ref_n_samples`/`ref_text`
- [ ] `--ref-audio` and `--ref-text` CLI flags
- [ ] CLI test validates new flags
- [ ] All 10 CLI tests pass
