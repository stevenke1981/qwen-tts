// build.rs — link qwen-core.lib and ggml DLLs for native FFI inference.
//
// The pre-built qwen-core static library and ggml DLLs:
//   - qwen-core.lib      in vendor/qwentts.cpp/build/Release/
//   - ggml*.dll          in vendor/qwentts.cpp/build/Release/
//   - ggml*.lib (import) in vendor/qwentts.cpp/build/ggml/src/Release/
//
// This build script:
//  1. Adds both directories to the linker search path.
//  2. Links qwen-core (static) and ggml/ggml-base/ggml-cpu (import libs for DLLs).
//  3. Copies ggml DLLs to the cargo output directory so they're findable at runtime.

use std::{env, fs, path::Path};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    // CPU backend is at crates/backends/cpu/ => go up 3 levels to workspace root.
    let workspace_root = Path::new(&manifest_dir).join("../../../");

    // --- Library directories ---
    let dll_dir = workspace_root.join("vendor/qwentts.cpp/build/Release/");
    let dll_dir = dll_dir.canonicalize().unwrap_or_else(|e| {
        panic!("Cannot find qwen-core DLL directory at {}: {e}", dll_dir.display())
    });

    let implib_dir = workspace_root.join("vendor/qwentts.cpp/build/ggml/src/Release/");
    let implib_dir = if implib_dir.exists() {
        implib_dir.canonicalize().unwrap()
    } else {
        // Fallback: import libs might be in the same directory as DLLs.
        dll_dir.clone()
    };

    // 1. Library search paths (import libs first, then DLL dir for qwen-core.lib)
    println!("cargo:rustc-link-search=native={}", implib_dir.display());
    println!("cargo:rustc-link-search=native={}", dll_dir.display());

    // 2. Link libraries
    //    qwen-core is a static library containing the full TTS pipeline.
    //    (found in dll_dir)
    println!("cargo:rustc-link-lib=static=qwen-core");

    //    ggml, ggml-base, ggml-cpu are shared libraries (DLLs).
    //    Their import libraries (.lib) are in implib_dir.
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");

    //    Windows system libraries needed by ggml.
    println!("cargo:rustc-link-lib=dylib=advapi32");
    println!("cargo:rustc-link-lib=dylib=ole32");

    // 3. Copy DLLs to cargo output directory for runtime resolution.
    //    OUT_DIR is typically target/debug/build/<crate>-<hash>/out/.
    //    We go up 3 levels to get target/debug/ (or target/release/).
    let dlls = ["ggml.dll", "ggml-base.dll", "ggml-cpu.dll"];
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    // target/debug/ or target/release/
    let profile_dir = out_path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("OUT_DIR should be target/<profile>/build/<crate>/out/");

    // target/debug/deps/ or target/release/deps/
    let deps_dir = profile_dir.join("deps");
    let output_dirs = [profile_dir.as_os_str(), deps_dir.as_os_str()];

    for dll in &dlls {
        let src = dll_dir.join(dll);
        for dest_os in &output_dirs {
            let dst_dir = Path::new(dest_os);
            let dst = dst_dir.join(dll);
            if src.exists() {
                let needs_copy = !dst.exists()
                    || dst.metadata().ok().and_then(|m| m.modified().ok())
                        < src.metadata().ok().and_then(|m| m.modified().ok());
                if needs_copy {
                    fs::copy(&src, &dst).unwrap_or_else(|e| {
                        panic!("Failed to copy {} to {}: {e}", src.display(), dst.display())
                    });
                }
            } else {
                panic!("Required DLL not found at {}", src.display());
            }
        }
        println!("cargo:rerun-if-changed={}", src.display());
    }

    // 4. Rerun build if the static library changes
    let lib_path = dll_dir.join("qwen-core.lib");
    println!("cargo:rerun-if-changed={}", lib_path.display());
}
