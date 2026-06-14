use std::{env, path::PathBuf, process::Command};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // workspace root is ../../ (crates/qwentts-sys → crates → workspace)
    let project_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two levels below workspace root")
        .to_path_buf();

    let vendor_dir = project_root.join("vendor").join("qwentts.cpp");
    let build_dir = vendor_dir.join("build");
    let dist_dir = project_root.join("dist");

    // Search order: dist/ → build/Release/ → cmake build
    let lib_name = "qwen";
    let found = if cfg!(target_os = "windows") {
        search_dll(&dist_dir, "qwen.dll")
            .or_else(|| search_dll(&build_dir.join("Release"), "qwen.dll"))
            .or_else(|| {
                eprintln!(
                    "cargo:warning=qwen.dll not found; attempting cmake build with QWEN_SHARED=ON"
                );
                cmake_build_shared(&vendor_dir, &build_dir);
                search_dll(&build_dir.join("Release"), "qwen.dll")
            })
    } else if cfg!(target_os = "linux") {
        search_dll(&dist_dir, "libqwen.so")
            .or_else(|| search_dll(&build_dir, "libqwen.so"))
            .or_else(|| {
                cmake_build_shared(&vendor_dir, &build_dir);
                search_dll(&build_dir, "libqwen.so")
            })
    } else {
        // macOS
        search_dll(&dist_dir, "libqwen.dylib")
            .or_else(|| search_dll(&build_dir, "libqwen.dylib"))
            .or_else(|| {
                cmake_build_shared(&vendor_dir, &build_dir);
                search_dll(&build_dir, "libqwen.dylib")
            })
    };

    match found {
        Some(path) => {
            let dir = path.parent().unwrap();
            println!("cargo:rustc-link-lib={lib_name}");
            println!("cargo:rustc-link-search={}", dir.display());
            println!("cargo:warning=linked qwen shared library: {}", path.display());
        }
        None => {
            eprintln!("cargo:warning=qwen shared library not found and cmake build failed");
            eprintln!("cargo:warning=To build manually:");
            eprintln!("cargo:warning=  cmake -S vendor/qwentts.cpp -B vendor/qwentts.cpp/build -DQWEN_SHARED=ON");
            eprintln!("cargo:warning=  cmake --build vendor/qwentts.cpp/build --config Release --target qwen");
        }
    }

    println!("cargo:rerun-if-changed=vendor/qwentts.cpp/src/qwen.h");
    println!("cargo:rerun-if-changed=build.rs");
}

fn search_dll(dir: &PathBuf, name: &str) -> Option<PathBuf> {
    let path = dir.join(name);
    if path.exists() { Some(path) } else { None }
}

fn cmake_build_shared(vendor_dir: &PathBuf, build_dir: &PathBuf) {
    // Run cmake configure
    let cfg_status = Command::new("cmake")
        .args([
            "-S",
            vendor_dir.to_str().unwrap(),
            "-B",
            build_dir.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DQWEN_SHARED=ON",
        ])
        .status();

    match cfg_status {
        Ok(s) if s.success() => {
            eprintln!("cargo:warning=cmake configure succeeded");
        }
        Ok(s) => {
            eprintln!("cargo:warning=cmake configure failed (exit={s:?})");
            return;
        }
        Err(e) => {
            eprintln!("cargo:warning=cmake not found: {e}");
            return;
        }
    }

    // Build the shared library target
    let build_status = Command::new("cmake")
        .args([
            "--build",
            build_dir.to_str().unwrap(),
            "--config",
            "Release",
            "--target",
            "qwen",
        ])
        .status();

    match build_status {
        Ok(s) if s.success() => {
            eprintln!("cargo:warning=cmake build --target qwen succeeded");
        }
        Ok(s) => {
            eprintln!("cargo:warning=cmake build --target qwen failed (exit={s:?})");
        }
        Err(e) => {
            eprintln!("cargo:warning=cmake build command failed: {e}");
        }
    }
}
