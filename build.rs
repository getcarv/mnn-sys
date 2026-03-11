use std::{env,
          path::{Path,
                 PathBuf}};

#[cfg(feature = "build-mnn")]
fn target_os() -> String {
    env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string())
}

#[cfg(feature = "build-mnn")]
fn target_arch() -> String {
    env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string())
}

#[cfg(feature = "build-mnn")]
fn is_ios() -> bool {
    target_os() == "ios"
}

#[cfg(feature = "build-mnn")]
fn is_macos() -> bool {
    target_os() == "macos"
}

#[cfg(feature = "build-mnn")]
fn is_apple() -> bool {
    is_ios() || is_macos()
}

#[cfg(feature = "build-mnn")]
fn is_linux() -> bool {
    target_os() == "linux"
}

#[cfg(feature = "build-mnn")]
fn is_windows() -> bool {
    target_os() == "windows"
}

#[cfg(feature = "build-mnn")]
fn is_android() -> bool {
    target_os() == "android"
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Use the MNN submodule
    let mnn_dir = manifest_dir.join("MNN");

    if !mnn_dir.exists() {
        panic!("MNN submodule not found at {:?}. Please run: git submodule update --init --recursive", mnn_dir);
    }

    // Build MNN library
    #[cfg(feature = "build-mnn")]
    let mnn_include_dir = mnn_dir.join("include");
    #[cfg(feature = "build-mnn")]
    let mnn_lib_dir = build_mnn(&mnn_dir, &out_dir);

    #[cfg(feature = "system-mnn")]
    {
        println!("cargo:rustc-link-lib=MNN");
    }

    // Build the C wrapper
    #[cfg(feature = "build-mnn")]
    build_wrapper(&manifest_dir, &mnn_include_dir, &mnn_lib_dir, &out_dir);

    // Generate Rust bindings from the C wrapper header
    generate_bindings(&manifest_dir, &out_dir);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");
    println!("cargo:rerun-if-changed=MNN");
}

#[cfg(feature = "build-mnn")]
fn build_mnn(mnn_dir: &Path, _out_dir: &Path) -> PathBuf {
    use cmake::Config;

    let mut config = Config::new(mnn_dir);

    config
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_SEP_BUILD", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("MNN_BUILD_DEMO", "OFF")
        .define("MNN_BUILD_QUANTOOLS", "OFF")
        .define("MNN_EVALUATION", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_BUILD_TEST", "OFF")
        .define("MNN_BUILD_BENCHMARK", "OFF")
        .define("MNN_LOW_MEMORY", "ON") // Enable low memory mode for dynamic quantization
        .define("MNN_BUILD_MINI", "OFF") // Ensure Express module is included (for Module API)
        .define("MNN_SKIPBUILD_GEOMETRY", "OFF") // Required for Express/Module API
        .define("CMAKE_BUILD_TYPE", "Release");

    // iOS-specific configurations - use MNN's official toolchain file
    if is_ios() {
        let arch = target_arch();
        let toolchain_file = mnn_dir.join("cmake/ios.toolchain.cmake");

        // Use MNN's iOS toolchain file (this is how MNN officially builds for iOS)
        config.define("CMAKE_TOOLCHAIN_FILE", toolchain_file.to_str().unwrap());

        // Set platform and architecture
        let (platform, cmake_arch) = match arch.as_str() {
            "aarch64" | "arm64" => ("OS64", "arm64"),
            "x86_64" => ("SIMULATOR64", "x86_64"),
            _ => ("OS64", "arm64"),
        };
        config.define("PLATFORM", platform);
        config.define("ARCHS", cmake_arch);
        config.define("ENABLE_BITCODE", "0");
        config.define("DEPLOYMENT_TARGET", "12.0");

        // MNN iOS-specific options (from official buildiOS.sh)
        config.define("MNN_AAPL_FMWK", "0"); // We want static lib, not framework
        config.define("MNN_ARM82", "ON");
        config.define("MNN_USE_THREAD_POOL", "OFF");

        // Enable Metal on iOS (it's the primary GPU backend)
        #[cfg(feature = "metal")]
        config.define("MNN_METAL", "ON");

        // CoreML support on iOS
        #[cfg(feature = "coreml")]
        config.define("MNN_COREML", "ON");
    }

    // macOS-specific configurations
    if is_macos() {
        #[cfg(feature = "metal")]
        config.define("MNN_METAL", "ON");

        #[cfg(feature = "coreml")]
        config.define("MNN_COREML", "ON");
    }

    // Android-specific configurations
    if is_android() {
        // Enable OpenCL for GPU acceleration on Android (always enabled for Android)
        config.define("MNN_OPENCL", "ON");

        // Enable ARM optimizations
        config.define("MNN_ARM82", "ON");

        // Use thread pool instead of OpenMP for Android
        config.define("MNN_USE_THREAD_POOL", "ON");
        config.define("MNN_OPENMP", "OFF");

        // Use standard cmake install targets (without this, MNN uses custom post-build
        // commands that don't define an install target, causing cmake-rs to fail)
        config.define("MNN_BUILD_FOR_ANDROID_COMMAND", "ON");
    }

    let dst = config.build();

    // Return the library directory
    let lib_dir = dst.join("lib");
    if lib_dir.exists() {
        lib_dir
    } else {
        dst.join("lib64")
    }
}

#[cfg(feature = "build-mnn")]
fn build_wrapper(manifest_dir: &Path, mnn_include_dir: &Path, mnn_lib_dir: &Path, out_dir: &Path) {
    let wrapper_cpp = manifest_dir.join("wrapper.cpp");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(&wrapper_cpp)
        .include(mnn_include_dir)
        .include(manifest_dir)
        .flag_if_supported("-std=c++14")
        .flag_if_supported("-fPIC")
        .warnings(false);

    // iOS-specific compiler configuration
    if is_ios() {
        let arch = target_arch();
        let sdk = if arch == "aarch64" || arch == "arm64" { "iphoneos" } else { "iphonesimulator" };

        let sdk_path = std::process::Command::new("xcrun")
            .args(["--sdk", sdk, "--show-sdk-path"])
            .output()
            .expect("Failed to get iOS SDK path")
            .stdout;
        let sdk_path = String::from_utf8_lossy(&sdk_path).trim().to_string();

        build.flag("-isysroot").flag(&sdk_path).flag("-miphoneos-version-min=12.0");
    }

    build.compile("mnn_wrapper");

    // Link the MNN static library
    println!("cargo:rustc-link-search=native={}", mnn_lib_dir.display());
    println!("cargo:rustc-link-lib=static=MNN");

    // Link against the wrapper we just built
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=mnn_wrapper");

    // Link C++ standard library and frameworks based on target OS
    if is_apple() {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");

        #[cfg(feature = "metal")]
        {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        }

        #[cfg(feature = "coreml")]
        {
            println!("cargo:rustc-link-lib=framework=CoreML");
            println!("cargo:rustc-link-lib=framework=CoreVideo");
        }

        // iOS-specific frameworks
        if is_ios() {
            println!("cargo:rustc-link-lib=framework=UIKit");
        }
    }

    if is_linux() {
        println!("cargo:rustc-link-lib=stdc++");
    }

    if is_windows() {
        println!("cargo:rustc-link-lib=msvcrt");
    }

    if is_android() {
        // Android NDK uses libc++ as the C++ standard library
        println!("cargo:rustc-link-lib=c++_static");
        println!("cargo:rustc-link-lib=c++abi");
    }
}

fn generate_bindings(manifest_dir: &Path, out_dir: &Path) {
    let wrapper_h = manifest_dir.join("wrapper.h");

    let mut builder = bindgen::Builder::default()
        .header(wrapper_h.to_str().unwrap())
        // Allow all MNNC wrapper types and functions
        .allowlist_type("MNNC_.*")
        .allowlist_function("mnnc_.*")
        .allowlist_var("MNNC_.*")
        // Derive common traits
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        .derive_eq(true)
        .derive_hash(true)
        // Use core types
        .use_core();

    // Fix target triple for iOS simulator - Clang expects "simulator" not "sim"
    let target = env::var("TARGET").unwrap_or_default();
    if target.ends_with("-ios-sim") {
        let clang_target = target.replace("-ios-sim", "-ios-simulator");
        builder = builder.clang_arg(format!("--target={clang_target}"));
    }

    // Handle cross-compilation for Linux targets (e.g., aarch64-unknown-linux-musl).
    // bindgen uses libclang which doesn't know about the cross-compiler's sysroot.
    if target.contains("linux") {
        builder = builder.clang_arg(format!("--target={target}"));

        let cc_env_var = format!("CC_{}", target.replace('-', "_"));
        let compiler = env::var(&cc_env_var).unwrap_or_else(|_| format!("{target}-gcc"));

        if let Ok(output) = std::process::Command::new(&compiler).arg("-print-sysroot").output() {
            let sysroot = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !sysroot.is_empty() {
                builder = builder.clang_arg(format!("--sysroot={sysroot}"));
            }
        }

        if let Ok(output) = std::process::Command::new(&compiler).arg("-print-file-name=include").output() {
            let include_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !include_dir.is_empty() && include_dir != "include" {
                builder = builder.clang_arg(format!("-isystem{include_dir}"));
            }
        }
    }

    let bindings = builder.generate().expect("Failed to generate bindings");

    bindings.write_to_file(out_dir.join("bindings.rs")).expect("Failed to write bindings");
}
