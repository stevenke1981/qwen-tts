# GUI Audio Playback — Design Spec

## Goal

Add in-app WAV audio playback to the egui GUI so the user can hear synthesis
results immediately without opening an external player.

---

## Behaviour

| Scenario | Action |
|----------|--------|
| Synthesis completes successfully | Auto-play the output WAV |
| User clicks "播放音檔" button | Re-play the last synthesised WAV (manual) |
| User presses Stop / starts a new synthesis | Stop current playback |
| No WAV available yet | Play button disabled |

Playback is **non-blocking** — the UI stays responsive and the progress bar
updates every frame.

---

## Architecture

```
rodio::OutputStream (lives in QwenTtsApp)
         │
         ▼
rodio::Sink (play/pause/stop)
         │
         ▼
WAV file on disk ← already written by synthesis worker
```

### Crate

`rodio` 0.19 — cross-platform, non-blocking, built-in WAV support.

```toml
# crates/app/Cargo.toml
rodio = { version = "0.19", default-features = false, features = ["wav"] }
```

No additional platform-specific setup.

---

## Data model — new fields on `QwenTtsApp`

```rust
struct QwenTtsApp {
    // … existing fields …

    /// Path of the most recently synthesised WAV.
    last_wav_path: Option<PathBuf>,

    /// rodio output stream — must outlive every Sink.
    #[allow(dead_code)]
    _audio_stream: Option<rodio::OutputStream>,

    /// Active playback handle (None when idle or stopped).
    audio_sink: Option<rodio::Sink>,

    /// Duration of the loaded WAV (used for progress calculation).
    audio_duration_secs: f64,
}
```

### Why `Option<OutputStream>`?

`rodio::OutputStreamHandle` is `!Send`, so we keep the `OutputStream` alive in
the main thread and construct the `Sink` from it after synthesis completes.

### Progress tracking

Every `update()` frame:
1. If `audio_sink.is_some()` → call `sink.get_pos()` (returns `Duration`).
2. Compute `progress = pos.as_secs_f64() / audio_duration_secs` (clamped 0–1).
3. Render an `egui::ProgressBar`.
4. When `sink.empty()` → playback finished; set `audio_sink = None`.

---

## UI layout

```
┌─────────────────────────────────────────────────────────┐
│  語音合成                                                │
│  [文字輸入框]                                            │
│  ...                                                    │
│  [開始合成]                                     [播放]   │
│                                                         │
│  ── playback ─────────────────────────────────────      │
│  [▶/⏸] [⏹]  ▓▓▓▓▓▓▓▓░░░░░░░░░░  45%  (00:12 / 00:27)  │
│                                                         │
│  狀態: 已產生 output.wav (24000Hz, 1聲道)               │
└─────────────────────────────────────────────────────────┘
```

- Playback bar appears **only when** `last_wav_path.is_some()`.
- **Play/Pause** toggles `sink.play()` / `sink.pause()`.
- **Stop** calls `sink.stop()` and sets `audio_sink = None`.
- **Progress** updates every frame while playing.
- **Play button** near the "開始合成" button for manual replay.

---

## Lifecycle

```
Synthesis started
    │
    ▼
Worker thread → synthesise → write WAV → send WorkerMessage::SynthesisFinished
    │
    ▼
update() receives message
    │
    ├── store last_wav_path
    ├── create OutputStream + Sink
    ├── sink.append(file_reader)
    └── auto-play starts immediately
    │
    ▼
Every frame: update progress bar
    │
    ▼
sink.empty() → playback over → audio_sink = None
    │
    ▼
User clicks Play → new Sink from same file
User clicks Stop → sink.stop(); audio_sink = None
User starts new synthesis → sink.stop() (cancel current playback)
```

---

## Error handling

| Scenario | Behaviour |
|----------|-----------|
| WAV file deleted before replay | `sink.append()` returns error → log warning, disable play button |
| OutputStream creation fails | Log error, playback buttons stay hidden |
| Rodio panics on corrupted file | `std::panic::catch_unwind` → show error in status bar |

---

## Testing

- Manual: `cargo run --features ffi -p qwen-tts-app`
- Run `synth` with any backend, verify auto-play + progress + stop + replay.
- Unit tests for progress calculation edge cases (0-length, empty sink).

---

## Non-goals

- Waveform visualisation
- Volume slider (can be added later)
- Streaming / real-time playback during synthesis
- Platform-specific audio backends (rodio handles this)
