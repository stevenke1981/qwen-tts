# Stage Timing Infrastructure

**Part of:** pure-rust-ffi-parity-optimization (sub-task 1)
**Date:** 2026-06-15

## Goal

Add cold/warm stage timers and reproducible JSON/CSV benchmark output for the pure Rust synthesis pipeline, covering prompt construction, talker, predictor, codec, WAV, TTFA, and total time.

## Data Model

```rust
pub struct TimingEvent {
    pub stage: String,       // "talker_step", "predictor_frame", "codec_decode", "model_load", "total"
    pub duration_s: f64,
    pub category: String,    // "step", "frame", "decode", "load", "total"
    pub sequence: usize,     // index within category (step number / frame number)
}

pub struct TimingRecorder {
    pub events: Vec<TimingEvent>,
    start: Instant,
}
```

Methods:
- `record(stage, category, duration_s, sequence)` — directly append a pre-measured duration.
- `start(stage, category, sequence)` → `TimerGuard` — scoped stopwatch that records on drop.
- `to_json()`, `to_csv()`, `summary()` — export functions.

## Three-Layer Architecture

### Layer 1: Core (`src/timing.rs`, NEW)
- `TimingEvent`, `TimingRecorder` — standalone, no candle/tensor dependencies.
- JSON export via `serde_json` (optional, behind `timing-export` feature).
- CSV export as basic `String` builder.

### Layer 2: Pipeline hook (`src/pipeline.rs`, MODIFY)
- `Pipeline::synthesize()` accepts `timing_recorder: Option<&mut TimingRecorder>`.
- Records events at each phase: tokenize, prompt embed, talker prefill steps, cb0 sampling, predictor frames, codec decode, total.
- Behind `#[cfg(feature = "pipeline-timing")]` to keep release build overhead-free.

### Layer 3: Benchmark tests (`tests/q8_bench.rs`, `tests/cross_val.rs`, MODIFY)
- `bench_128_frames` wraps existing stages with `TimingRecorder`.
- Exports `.json` / `.csv` to `target/bench-results/`.
- New `bench_128_frames_end2end` in `cross_val.rs` exercising the full `Pipeline::synthesize()` path.

## Timing Points (Pipeline)

| Trigger | stage | category | sequence |
|---------|-------|----------|----------|
| Model loaded | `"model_load"` | `"load"` | 0 |
| Text tokenized | `"tokenize"` | `"load"` | 0 |
| Text embedded | `"prompt_embed"` | `"load"` | 0 |
| Each talker KV prefill step | `"talker_step"` | `"step"` | pos |
| Each cb0 sample | `"cb0_sample"` | `"step"` | frame_idx |
| Each predictor frame | `"predictor_frame"` | `"frame"` | frame_idx |
| Codec decode | `"codec_decode"` | `"decode"` | 0 |
| Total synthesis | `"total"` | `"total"` | 0 |

## Output Format

JSON (primary):
```json
[
  {"stage":"talker_step","duration_s":0.072,"category":"step","sequence":0},
  {"stage":"predictor_frame","duration_s":0.086,"category":"frame","sequence":0}
]
```

CSV (secondary, for spreadsheet):
```csv
stage,duration_s,category,sequence
talker_step,0.072,step,0
predictor_frame,0.086,frame,0
```

## Files Changed

| File | Change |
|------|--------|
| `src/timing.rs` | NEW — TimingEvent, TimingRecorder |
| `src/lib.rs` | Add `pub mod timing;` |
| `Cargo.toml` | Add `serde`, `serde_json` as optional deps behind `timing-export` feature |
| `src/pipeline.rs` | Add `timing_recorder` param to `synthesize()`, record events |
| `tests/q8_bench.rs` | Integrate TimingRecorder, export JSON |
| `tests/cross_val.rs` | Add end-to-end benchmark with Pipeline::synthesize |
