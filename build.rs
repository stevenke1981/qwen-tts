fn main() {
    // Reserved for future native qwentts.cpp FFI build integration.
    // Current MVP uses an external qwen-tts executable.
    println!("cargo:rerun-if-changed=build.rs");
}
