//! Stage timing infrastructure: `TimingRecorder` for collecting per-stage
//! benchmark measurements and exporting as JSON or CSV.
//!
//! Three-layer design:
//!   1. Core (this file) — `TimingEvent`, `TimerGuard`, `TimingRecorder`
//!   2. Pipeline hooks — optional `&mut TimingRecorder` param in `synthesize()`
//!   3. Benchmark tests — full integration with JSON/CSV export to `target/bench-results/`
//!
//! No candle/tensor dependencies; only `std::time::Instant`.

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
///
/// ```ignore
/// let _guard = recorder.start("talker_step".into(), "step".into(), pos);
/// // ... work happens ...
/// drop(_guard); // records elapsed time automatically
/// ```
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

    /// Summary: category -> total seconds.
    pub fn summary(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        for e in &self.events {
            *map.entry(e.category.clone()).or_insert(0.0) += e.duration_s;
        }
        map
    }

    /// Export as JSON string (no serde dependency needed).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_summary() {
        let mut tr = TimingRecorder::new();
        tr.record("a".into(), "x".into(), 1.0, 0);
        tr.record("b".into(), "x".into(), 2.0, 1);
        tr.record("c".into(), "y".into(), 3.0, 0);

        let summary = tr.summary();
        assert!((summary["x"] - 3.0).abs() < 1e-9);
        assert!((summary["y"] - 3.0).abs() < 1e-9);
        assert!((tr.category_total("x") - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_timer_guard() {
        let mut tr = TimingRecorder::new();
        {
            let _g = tr.start("op".into(), "cat".into(), 42);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(tr.len(), 1);
        assert_eq!(tr.events[0].stage, "op");
        assert_eq!(tr.events[0].category, "cat");
        assert_eq!(tr.events[0].sequence, 42);
        assert!(tr.events[0].duration_s >= 0.001);
    }

    #[test]
    fn test_to_json() {
        let mut tr = TimingRecorder::new();
        tr.record("x".into(), "y".into(), 1.5, 0);
        let json = tr.to_json();
        assert!(json.contains("\"stage\":\"x\""));
        assert!(json.contains("\"duration_s\":1.5"));
        assert!(json.contains("\"category\":\"y\""));
    }

    #[test]
    fn test_to_csv() {
        let mut tr = TimingRecorder::new();
        tr.record("x".into(), "y".into(), 1.5, 0);
        let csv = tr.to_csv();
        assert!(csv.contains("x,1.5"));
        assert!(csv.contains("stage,duration_s,category,sequence"));
    }

    #[test]
    fn test_default() {
        let tr = TimingRecorder::default();
        assert!(tr.is_empty());
    }

    #[test]
    fn test_category_total_empty() {
        let tr = TimingRecorder::new();
        assert!((tr.category_total("nonexistent") - 0.0).abs() < 1e-9);
    }
}
