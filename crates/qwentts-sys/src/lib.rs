//! Raw FFI bindings and safe wrapper for qwentts.cpp shared library.
//!
//! # Overview
//!
//! This crate provides two layers:
//!
//! - [`raw`] — `extern "C"` function declarations and POD structs,
//!   directly generated from `qwen.h`.  The functions are `unsafe`.
//! - [`safe`] — Idiomatic Rust wrapper (`QwenTts` struct) that
//!   manages initialisation, synthesis and resource cleanup.
//!
//! Most end-users should use [`safe::QwenTts`] directly.
//!
//! # Feature flags
//!
//! (none at present)

// ─── Raw FFI declarations ────────────────────────────────────────────────
pub mod raw;
// ─── Safe wrapper ─────────────────────────────────────────────────────────
pub mod safe;
