//! Raw FFI function declarations from `qwen.h`.
//!
//! All functions are `unsafe` and map 1:1 to the C ABI exported by
//! `qwentts.cpp/src/qwen.cpp`.  The shared library (`qwen.dll` /
//! `libqwen.so` / `libqwen.dylib`) is linked by `build.rs`.
//!
//! See `vendor/qwentts.cpp/src/qwen.h` for the authoritative
//! declaration.

// Re-export all public symbols from the auto-generated (or hand-written) FFI module.
