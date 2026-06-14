# GUI Audio Playback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Play the synthesised WAV file inside the egui GUI using rodio, with auto-play after synthesis and manual play/pause/stop controls.

**Architecture:** `rodio::OutputStream` lives in `QwenTtsApp` (main thread). After synthesis completes, create a `rodio::Sink` from the output file. Update progress bar every frame via `sink.get_pos()`. On Stop / new synthesis, call `sink.stop()`.

**Tech Stack:** rodio 0.19, egui 0.27, Rust 2021

**Files:**
- Modify: `crates/app/Cargo.toml` — add rodio dependency
- Modify: `crates/app/src/main.rs` — add fields, UI, playback logic

---

### Task 1: Add rodio dependency

- [ ] **Step 1: Edit Cargo.toml**

Insert rodio after the existing dependencies:

```toml
[dependencies]
eframe = "0.27.2"
rodio = { version = "0.19", default-features = false, features = ["wav"] }
qwen-tts-backend-cpu = { path = "../backends/cpu" }
# ... rest unchanged
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p qwen-tts-app`
Expected: success (rodio resolves and links)

- [ ] **Step 3: Commit**

```bash
git add crates/app/Cargo.toml Cargo.lock
git commit -m "deps(app): add rodio 0.19 for WAV playback"
```

---

### Task 2: Add playback fields to QwenTtsApp

- [ ] **Step 1: Add imports and new fields**

At the top of `main.rs`, add the rodio import:

```rust
use std::time::Duration;
```

Add these fields to `struct QwenTtsApp` (after `clamp_fp16: bool,`):

```rust
    /// Path of the most recently synthesised WAV.
    last_wav_path: Option<PathBuf>,
    /// rodio output stream handle — must outlive every Sink.
    _audio_stream: Option<rodio::OutputStream>,
    /// Active playback sink (None when idle or stopped).
    audio_sink: Option<rodio::Sink>,
    /// Duration of the loaded WAV (used for progress bar).
    audio_duration_secs: f64,
```

- [ ] **Step 2: Add defaults**

In the `Default` impl, after `clamp_fp16: false,`:

```rust
            last_wav_path: None,
            _audio_stream: None,
            audio_sink: None,
            audio_duration_secs: 0.0,
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p qwen-tts-app`
Expected: success (new fields added, no unused warnings)

- [ ] **Step 4: Commit**

```bash
git add crates/app/src/main.rs
git commit -m "feat(gui): add playback state fields to QwenTtsApp"
```

---

### Task 3: Add playback control UI

- [ ] **Step 1: Add playback section in update()**

In the `update()` method of `QwenTtsApp`, after the synthesis form and before the status bar, add a playback section that only renders when `last_wav_path` is set:

```rust
        // ── Playback controls ──────────────────────────────────────────
        if self.last_wav_path.is_some() {
            ui.add_space(6.0);
            ui.separator();
            ui.horizontal(|ui| {
                // Play / Pause toggle
                if let Some(ref sink) = self.audio_sink {
                    if sink.is_paused() {
                        if ui.button("▶ 播放").clicked() {
                            sink.play();
                        }
                    } else {
                        if ui.button("⏸ 暫停").clicked() {
                            sink.pause();
                        }
                    }
                    if ui.button("⏹ 停止").clicked() {
                        self.stop_playback();
                    }
                } else {
                    // Idle — show a single play button for replay
                    if ui.button("播放音檔").clicked() {
                        self.start_playback();
                    }
                }

                // ── Progress bar ──────────────────────────────────────
                if self.audio_duration_secs > 0.0 {
                    let progress = if let Some(ref sink) = self.audio_sink {
                        let pos = sink.get_pos().as_secs_f64();
                        (pos / self.audio_duration_secs).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let bar = egui::ProgressBar::new(progress as f32)
                        .desired_width(200.0)
                        .show_percentage();
                    ui.add(bar);

                    // Time label
                    let pos_str = if let Some(ref sink) = self.audio_sink {
                        format_duration(sink.get_pos())
                    } else {
                        "00:00".to_owned()
                    };
                    let total_str = format_duration(Duration::from_secs_f64(self.audio_duration_secs));
                    ui.label(format!("{pos_str} / {total_str}"));
                }
            });
        }
```

And at the bottom of the file, add the helper function:

```rust
/// Format a Duration as MM:SS.
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let mins = secs / 60;
    let secs = secs % 60;
    format!("{mins:02}:{secs:02}")
}
```

- [ ] **Step 2: Add start_playback and stop_playback methods**

Add to `impl QwenTtsApp` block (before `start_synthesis`):

```rust
    fn start_playback(&mut self) {
        let path = match self.last_wav_path.as_ref() {
            Some(p) => p.clone(),
            None => return,
        };

        // Stop any previous playback first.
        self.stop_playback();

        // Create OutputStream + Sink
        let (_stream, handle) = match rodio::OutputStream::try_default() {
            Ok(pair) => pair,
            Err(err) => {
                self.set_status(format!("無法開啟音訊裝置: {err}"));
                return;
            }
        };

        // Read the WAV file
        let file = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(err) => {
                self.set_status(format!("無法讀取音檔: {err}"));
                return;
            }
        };

        // Decode WAV and compute duration
        let source = match rodio::Decoder::new_wav(file) {
            Ok(src) => src,
            Err(err) => {
                self.set_status(format!("無法解碼 WAV: {err}"));
                return;
            }
        };
        self.audio_duration_secs = source
            .total_duration()
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let sink = rodio::Sink::connect(&handle);
        sink.append(source);
        // sink.play() is the default state — auto-play.

        self._audio_stream = Some(_stream);
        self.audio_sink = Some(sink);
        self.set_status("正在播放...");
    }

    fn stop_playback(&mut self) {
        if let Some(sink) = self.audio_sink.take() {
            sink.stop();
        }
        // Keep _audio_stream alive (drops with self).
    }
```

- [ ] **Step 3: Call stop_playback when starting new synthesis**

In `start_synthesis`, at the very beginning (after the empty-text check), add:

```rust
        // Stop any in-progress playback before starting synthesis.
        self.stop_playback();
```

- [ ] **Step 4: Auto-play after synthesis succeeds**

In the `update()` message handler for `WorkerMessage::SynthesisFinished(result)` (look for the match arm around line 227), after setting the status, add:

```rust
            Ok(WorkerMessage::SynthesisFinished(result)) => {
                match result {
                    Ok(msg) => {
                        self.set_status(&msg);
                        // Extract path from the "已產生 ..." message
                        // Message format: "已產生 path（24000 Hz，1 聲道）"
                        if let Some(path_str) = msg
                            .split('（')
                            .next()
                            .and_then(|s| s.strip_prefix("已產生 "))
                        {
                            let path = PathBuf::from(path_str);
                            if path.exists() {
                                self.last_wav_path = Some(path);
                                self.start_playback();
                                return; // start_playback already called set_status
                            }
                        }
                    }
                    Err(err) => self.set_status(&err),
                }
            }
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p qwen-tts-app`
Expected: success

- [ ] **Step 6: Commit**

```bash
git add crates/app/src/main.rs
git commit -m "feat(gui): add audio playback UI with auto-play on synthesis"
```

---

### Task 4: Edge cases and cleanup

- [ ] **Step 1: Reset last_wav_path on new synthesis**

At the start of `start_synthesis`, after `self.stop_playback();`, add:

```rust
        self.last_wav_path = None;
```

- [ ] **Step 2: Handle playback end detection**

In the playback section of `update()`, after rendering controls, check if the sink has finished:

```rust
        // ── Auto-detect playback end ──────────────────────────
        if self.audio_sink.is_some() {
            if let Some(ref sink) = self.audio_sink {
                if sink.empty() {
                    self.audio_sink = None; // allow replay
                    self.set_status("播放完成");
                }
            }
        }
```

This should be placed right after the playback controls block (before the status bar).

- [ ] **Step 3: Re-check compilation**

Run: `cargo check -p qwen-tts-app`
Expected: success

- [ ] **Step 4: Run CLI tests to confirm no breakage**

Run: `cargo test --features ffi -p qwen-tts-cli`
Expected: 10 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/app/src/main.rs
git commit -m "fix(gui): reset playback state on new synthesis, detect playback end"
```

---

### Task 5: Final verification

- [ ] **Step 1: Full workspace check**

Run: `cargo check --workspace --features ffi`
Expected: success, zero app-layer warnings

- [ ] **Step 2: Verify the playback bar renders correctly**

Run: `cargo run --features ffi -p qwen-tts-app`
Open the GUI, complete a synthesis, verify:
1. Auto-play starts
2. Progress bar moves
3. Pause/Play toggle works
4. Stop stops playback
5. "播放音檔" button replays
6. New synthesis stops current playback

- [ ] **Step 3: Final commit if any fixes needed during verification**

---

## Self-Review

**Spec coverage:**
- ✅ Auto-play after synthesis: Task 3 Step 4
- ✅ Manual replay button: Task 3 Step 1
- ✅ Play/Pause/Stop: Task 3 Step 1
- ✅ Progress bar: Task 3 Step 1
- ✅ Rodio integration: Task 2 + Task 3 Step 2
- ✅ Stop on new synthesis: Task 3 Step 3 + Task 4 Step 1
- ✅ Playback end detection: Task 4 Step 2
- ✅ Error handling (no device, corrupt WAV): Task 3 Step 2

**Placeholder scan:** All steps contain actual code. No TBD/TODO.

**Type consistency:** Field names match between `struct`, `Default`, `start_playback`, `stop_playback`, and `update()` usage.
