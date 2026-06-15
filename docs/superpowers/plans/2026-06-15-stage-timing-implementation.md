# Stage Timing Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add cold/warm stage timers and reproducible JSON/CSV benchmark output covering prompt, talker, predictor, codec, WAV, TTFA, and total time.

**Architecture:** Three-layer design — core `TimingRecorder` in `src/timing.rs`, optional hooks in `Pipeline::synthesize()`, and full integration in benchmark tests. No candle/tensor dependencies in the core layer.

**Tech Stack:** Rust, serde/serde_json (optional), standard library `std::time::Instant`.

**Spec:** `docs/superpowers/specs/2026-06-15-stage-timing-infrastructure-design.md`

**Parent plan:** `.opencode/plans/pure-rust-ffi-parity-optimization.md` (sub-task 1)

---

### Task 1: Core — TimingRecorder struct

**Files:**
- Create: `crates/backends/pure-rust/src/timing.rs`
- Modify: `crates/backends/pure-rust/src/lib.rs`

- [ ] **Step 1: Create `timing.rs` with TimingEvent and TimingRecorder**

```rust
use std::collections::HashMap;
use std::time::Instant;

/// A single timing measurement.
#[derive(Debug, Clone)]
pub struct TimingEvent {
    pub stage: String,
    pub duration_s: f64,
    pub category: String,
    pub sequence: usize,
}

/// Scoped guard that records elapsed time on drop.
pub struct TimerGuard<'a> {
    recorder: &'a mut TimingRecorder,
    stage: String,
    category: String,
    sequence: usize,
    start: Instant,
}

impl<'a> TimerGuard<'a> {
    fn new(
        recorder: &'a mut TimingRecorder,
        stage: String,
        category: String,
        sequence: usize,
    ) -> Self {
        Self { recorder, stage, category, sequence, start: Instant::now() }
    }
}

impl<'a> Drop for TimerGuard<'a> {
    fn drop(&mut self) {
        let dur = self.start.elapsed().as_secs_f64();
        self.recorder.record(
            std::mem::take(&mut self.stage),
            std::mem::take(&mut self.category),
            dur,
            self.sequence,
        );
    }
}

/// Accumulates timing events and can export to various formats.
#[derive(Debug, Clone)]
pub struct TimingRecorder {
    pub events: Vec<TimingEvent>,
}

impl TimingRecorder {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Directly append a pre-measured duration.
    pub fn record(&mut self, stage: String, category: String, duration_s: f64, sequence: usize) {
        self.events.push(TimingEvent { stage, duration_s, category, sequence });
    }

    /// Start a scoped timer; records on drop.
    pub fn start(&mut self, stage: String, category: String, sequence: usize) -> TimerGuard<'_> {
        TimerGuard::new(self, stage, category, sequence)
    }

    /// Total time for a given category.
    pub fn category_total(&self, category: &str) -> f64 {
        self.events
            .iter()
            .filter(|e| e.category == category)
            .map(|e| e.duration_s)
            .sum()
    }

    /// Summary: category → total seconds.
    pub fn summary(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        for e in &self.events {
            *map.entry(e.category.clone()).or_insert(0.0) += e.duration_s;
        }
        map
    }

    /// Export as JSON string.
    pub fn to_json(&self) -> String {
        let mut s = String::from("[\n");
        for (i, e) in self.events.iter().enumerate() {
            if i > 0 {
                s.push_str(",\n");
            }
            s.push_str(&format!(
                r#"  {{"stage":"{}","duration_s":{:.9},"category":"{}","sequence":{}}}"#,
                e.stage, e.duration_s, e.category, e.sequence
            ));
        }
        s.push_str("\n]\n");
        s
    }

    /// Export as CSV string.
    pub fn to_csv(&self) -> String {
        let mut s = String::from("stage,duration_s,category,sequence\n");
        for e in &self.events {
            s.push_str(&format!(
                "{},{:.9},{},{}\n",
                e.stage, e.duration_s, e.category, e.sequence
            ));
        }
        s
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }
}

impl Default for TimingRecorder {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 2: Add `pub mod timing;` to `lib.rs`**

```rust
// In crates/backends/pure-rust/src/lib.rs:
pub mod timing;
```

Find the other `pub mod` lines and insert after them, preserving alphabetical order.

- [ ] **Step 3: Build check**

Run: `cd crates\backends\pure-rust && cargo build --release --lib`
Expected: Success with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add crates/backends/pure-rust/src/timing.rs crates/backends/pure-rust/src/lib.rs
git commit -m "feat(timing): add TimingRecorder core struct with JSON/CSV export"
```

---

### Task 2: Pipeline timing hooks

**Files:**
- Modify: `crates/backends/pure-rust/src/pipeline.rs`
- Modify: `crates/backends/pure-rust/Cargo.toml`

- [ ] **Step 1: Add `pipeline-timing` feature to Cargo.toml**

```toml
# In crates/backends/pure-rust/Cargo.toml, under [features]:
pipeline-timing = []
```

Place it after any existing features. If `[features]` doesn't exist, create it.

- [ ] **Step 2: Modify `Pipeline::synthesize()` to accept `timing_recorder`**

```rust
// Modify the synthesize signature and add timing calls.

pub fn synthesize(
    &mut self,
    request: &SynthesisRequest,
    timing_recorder: Option<&mut TimingRecorder>,
) -> anyhow::Result<Vec<i16>> {
    // ... existing code ...
```

Wrap each major phase:

```rust
// After model load (top of function):
#[cfg(feature = "pipeline-timing")]
let _load_timer = timing_recorder.as_mut().map(|tr| tr.start(
    "model_load".into(), "load".into(), 0,
));

// After tokenization (around line 68):
#[cfg(feature = "pipeline-timing")]
{
    if let Some(ref mut tr) = timing_recorder {
        tr.record("tokenize".into(), "load".into(), tokenize_dur, 0);
    }
}

// After text_embeds (around line 79):
#[cfg(feature = "pipeline-timing")]
{
    if let Some(ref mut tr) = timing_recorder {
        tr.record("prompt_embed".into(), "load".into(), embed_dur, 0);
    }
}

// Wrap each talker forward_step in the prefill loop (lines 105-111):
#[cfg(feature = "pipeline-timing")]
let _step_timer = timing_recorder.as_mut().map(|tr| tr.start(
    "talker_step".into(), "step".into(), pos,
));
let hidden = self.talker.forward_step(&emb, &mut kv_cache, &cos_full, &sin_full)?;
#[cfg(feature = "pipeline-timing")]
drop(_step_timer);

// Wrap each cb0_sample:
#[cfg(feature = "pipeline-timing")]
let _cb0_timer = timing_recorder.as_mut().map(|tr| tr.start(
    "cb0_sample".into(), "step".into(), frame_idx,
));
let cb0_token = if do_sample { ... } else { ... };
#[cfg(feature = "pipeline-timing")]
drop(_cb0_timer);

// Wrap each predictor frame:
#[cfg(feature = "pipeline-timing")]
let _pred_timer = timing_recorder.as_mut().map(|tr| tr.start(
    "predictor_frame".into(), "frame".into(), frame_idx,
));
let frame = if do_sample { ... } else { ... };
#[cfg(feature = "pipeline-timing")]
drop(_pred_timer);

// Wrap codec decode:
#[cfg(feature = "pipeline-timing")]
let _codec_timer = timing_recorder.as_mut().map(|tr| tr.start(
    "codec_decode".into(), "decode".into(), 0,
));
let audio_f32 = self.codec_decoder.decode(&all_codes, total_frames);
#[cfg(feature = "pipeline-timing")]
drop(_codec_timer);
```

To keep the interface ergonomic, also add a timing-free overload:

```rust
pub fn synthesize_simple(&mut self, request: &SynthesisRequest) -> anyhow::Result<Vec<i16>> {
    self.synthesize(request, None)
}
```

- [ ] **Step 3: Import TimingRecorder**

Add to the import block at the top of `pipeline.rs`:

```rust
use crate::timing::TimingRecorder;
```

- [ ] **Step 4: Build check**

Run: `cd crates\backends\pure-rust && cargo build --release --lib`
Expected: Success.

- [ ] **Step 5: Build check with feature**

Run: `cd crates\backends\pure-rust && cargo build --release --lib --features pipeline-timing`
Expected: Success.

- [ ] **Step 6: Commit**

```bash
git add crates/backends/pure-rust/src/pipeline.rs crates/backends/pure-rust/Cargo.toml
git commit -m "feat(pipeline): add optional timing_recorder to synthesize()"
```

---

### Task 3: Benchmark integration — q8_bench.rs

**Files:**
- Modify: `crates/backends/pure-rust/tests/q8_bench.rs`

- [ ] **Step 1: Add TimingRecorder import + result output to bench_128_frames**

```rust
// Add import at top:
use qwen_tts_backend_pure_rust::timing::TimingRecorder;
use std::fs;
```

After loading both models, create the recorder:

```rust
let mut timing = TimingRecorder::new();
timing.record(
    "model_load".into(), "load".into(),
    load_talker_s + load_predictor_s, 0,
);
```

Wrap talker forward loop (line 250-255):

```rust
for _step in 0..128 {
    let start = Instant::now();
    let _h = talker
        .forward_step(&input, &mut talker_cache, &cos, &sin)
        .expect("talker forward_step");
    timing.record(
        "talker_step".into(), "step".into(),
        start.elapsed().as_secs_f64(), _step,
    );
}
```

Wrap predictor loop (line 287-306):

```rust
for _frame in 0..128 {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let start = Instant::now();
    let _codes = predictor
        .predict_one_frame_sampled(
            &talker_hidden, &c0_embed, 1.0, Some(40), Some(0.9), &mut rng,
        )
        .expect("predict frame");
    timing.record(
        "predictor_frame".into(), "frame".into(),
        start.elapsed().as_secs_f64(), _frame,
    );
}
```

Replace final summary block (after line 348) with JSON/CSV export:

```rust
// Export timing results
let bench_dir = project_root()
    .join("target")
    .join("bench-results");
let _ = fs::create_dir_all(&bench_dir);
let json_path = bench_dir.join("bench_128_frames.json");
let csv_path = bench_dir.join("bench_128_frames.csv");
fs::write(&json_path, &timing.to_json())
    .expect("write JSON results");
fs::write(&csv_path, &timing.to_csv())
    .expect("write CSV results");
println!();
println!("Timing results written to:");
println!("  {}", json_path.display());
println!("  {}", csv_path.display());

// Also print summary
let summary = timing.summary();
println!();
println!("=== Timing Summary ===");
for cat in ["load", "step", "frame", "decode"] {
    if let Some(total) = summary.get(cat) {
        println!("  {:<12} {:>8.3}s", cat, total);
    }
}
```

- [ ] **Step 2: Build and run benchmark**

Run: `cd crates\backends\pure-rust && cargo test --release --test q8_bench bench_128_frames -- --nocapture --include-ignored 2>&1`

Expected: Benchmark runs, prints timing summary, writes JSON + CSV to `target/bench-results/`.

- [ ] **Step 3: Verify JSON output**

Run: `Get-Content target\bench-results\bench_128_frames.json | Select-Object -First 5`
Expected: Valid JSON array with timing events.

- [ ] **Step 4: Commit**

```bash
git add crates/backends/pure-rust/tests/q8_bench.rs
git commit -m "feat(bench): integrate TimingRecorder into q8_bench, export JSON/CSV"
```

---

### Task 4: End-to-end benchmark — cross_val.rs

**Files:**
- Modify: `crates/backends/pure-rust/tests/cross_val.rs`

- [ ] **Step 1: Read existing cross_val.rs to understand structure**

Run: `type crates\backends\pure-rust\tests\cross_val.rs`

Analyze the imports, test functions, and patterns used.

- [ ] **Step 2: Add benchmark that exercises full Pipeline::synthesize**

Append after the last test function:

```rust
#[test]
#[ignore = "requires model files — run manually with --release"]
fn bench_128_frames_end2end() {
    use std::time::Instant;
    use qwen_tts_backend_pure_rust::pipeline::Pipeline;
    use qwen_tts_backend_pure_rust::timing::TimingRecorder;
    use qwen_tts_runtime::SynthesisRequest;

    let root = project_root();
    let talker_path = root.join("models").join("qwen-talker-1.7b-base-Q8_0.gguf");
    let codec_path = root.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf");
    let device = candle_core::Device::Cpu;

    let t0 = Instant::now();
    let mut pipeline = Pipeline::new(&talker_path, &codec_path, &device)
        .expect("load pipeline");
    let load_s = t0.elapsed().as_secs_f64();

    let mut timing = TimingRecorder::new();
    timing.record("model_load".into(), "load".into(), load_s, 0);

    // Cold run (first synthesis, KV caches cold)
    let request = SynthesisRequest {
        text: "純 Rust 語音合成測試。".to_string(),
        temperature: Some(1.0),
        top_k: Some(40),
        top_p: Some(0.9),
        do_sample: Some(true),
        max_new_tokens: Some(128),
        seed: Some(42),
        ..Default::default()
    };

    let t1 = Instant::now();
    let audio = pipeline.synthesize(&request, Some(&mut timing))
        .expect("cold synthesis");
    let cold_total = t1.elapsed().as_secs_f64();

    // Warm run (same text, second synthesis)
    let t2 = Instant::now();
    let audio2 = pipeline.synthesize(&request, Some(&mut timing))
        .expect("warm synthesis");
    let warm_total = t2.elapsed().as_secs_f64();

    println!();
    println!("=== End-to-End 128-frame benchmark (cross_val) ===");
    println!("  cold: {:.3}s", cold_total);
    println!("  warm: {:.3}s", warm_total);
    println!("  audio samples: {} (cold), {} (warm)", audio.len(), audio2.len());

    // Export timing
    let bench_dir = root.join("target").join("bench-results");
    let _ = std::fs::create_dir_all(&bench_dir);
    std::fs::write(bench_dir.join("bench_end2end.json"), &timing.to_json())
        .expect("write end2end JSON");
    std::fs::write(bench_dir.join("bench_end2end.csv"), &timing.to_csv())
        .expect("write end2end CSV");

    println!();
    println!("Timing summary (includes both cold + warm):");
    for (cat, total) in timing.summary() {
        println!("  {:<12} {:>8.3}s", cat, total);
    }

    // Basic sanity: audio should be non-empty
    assert!(!audio.is_empty(), "cold synthesis produced no audio");
    assert!(!audio2.is_empty(), "warm synthesis produced no audio");
    assert_eq!(audio.len(), audio2.len(),
        "cold and warm should produce same number of samples");
}
```

- [ ] **Step 3: Add `pipeline` module import if not already present**

Check the top of cross_val.rs for `use qwen_tts_backend_pure_rust::pipeline::*;` — if missing, add it.

- [ ] **Step 4: Build check**

Run: `cd crates\backends\pure-rust && cargo build --release --lib`
Expected: Success.

- [ ] **Step 5: Run short sanity**

Run: `cd crates\backends\pure-rust && cargo test --release --test cross_val bench_128_frames_end2end -- --nocapture --include-ignored 2>&1 | Select-String -First 30`

Expected: Tests compile and run. Audio length sanity check passes.

- [ ] **Step 6: Commit**

```bash
git add crates/backends/pure-rust/tests/cross_val.rs
git commit -m "feat(bench): add end-to-end 128-frame benchmark in cross_val.rs"
```

---

### Task 5: Update parent plan & lessons.md

**Files:**
- Modify: `.opencode/plans/pure-rust-ffi-parity-optimization.md`
- Modify: `lessons.md`

- [ ] **Step 1: Mark sub-task 1 as complete in parent plan**

Edit `.opencode/plans/pure-rust-ffi-parity-optimization.md`:

```markdown
1. [x] Add stage timers and reproducible FFI/Rust benchmark output.
```

- [ ] **Step 2: Append lesson to lessons.md**

```markdown
---
## Lesson #27 — 2026-06-15
**Trigger:** Stage timing infrastructure needed configurable granularity — per-step, per-frame, and total.
**Rule:** For benchmark timing infrastructure, use a three-layer design: (1) core data struct with serde-free JSON/CSV export, (2) optional pipeline hooks behind a feature flag, (3) test-level integration. This avoids production overhead while supporting both unit- and integration-level timing.
**Source:** stage timing implementation (sub-task 1)
```

- [ ] **Step 3: Commit**

```bash
git add .opencode/plans/pure-rust-ffi-parity-optimization.md lessons.md
git commit -m "docs: mark sub-task 1 complete, add lesson #27"
```
