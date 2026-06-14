use clap::{error::ErrorKind, Parser, Subcommand, ValueEnum};
use qwen_tts_backend_cpu::CpuBackend;
use qwen_tts_core::{graph::TtsGraph, GgufProbe, TtsModelSet};
#[cfg(feature = "ffi")]
use qwen_tts_runtime::FfiBackend;
use qwen_tts_runtime::{
    backend_status, default_backend_executable, default_model_status, default_voice_output_path,
    ensure_default_models_with_progress, find_qwentts_executable, setup_qwentts_backend,
    BackendStatus, DeviceKind, ExternalQwenTtsBackend, ModelDownloadProgress, Scheduler,
    SynthesisRequest, DEFAULT_MODELS_DIR, DEFAULT_MODEL_FILES,
};
use std::{
    env,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

const DEFAULT_TALKER_MODEL: &str = "./models/qwen-talker-1.7b-base-Q8_0.gguf";
const DEFAULT_CODEC_MODEL: &str = "./models/qwen-tokenizer-12hz-Q8_0.gguf";

#[derive(Debug, Parser)]
#[command(name = "qwen-tts", about = "Qwen TTS", arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Backend(BackendArgs),
    Inspect(InspectArgs),
    Graph,
    Models(ModelsArgs),
    SetupScript(SetupScriptArgs),
    Synth(SynthArgs),
}

#[derive(Debug, Parser)]
struct BackendArgs {
    #[command(subcommand)]
    command: BackendCommand,
}

#[derive(Debug, Subcommand)]
enum BackendCommand {
    Status,
    Setup,
}

#[derive(Debug, Parser)]
struct InspectArgs {
    #[arg(long)]
    talker: PathBuf,
    #[arg(long)]
    codec: PathBuf,
}

#[derive(Debug, Parser)]
struct SetupScriptArgs {
    #[arg(long, default_value = "cpu")]
    target: SetupTarget,
}

#[derive(Debug, Parser)]
struct ModelsArgs {
    #[command(subcommand)]
    command: ModelsCommand,
}

#[derive(Debug, Subcommand)]
enum ModelsCommand {
    Status(ModelPathArgs),
    Download(ModelDownloadArgs),
}

#[derive(Debug, Parser)]
struct ModelPathArgs {
    #[arg(long, default_value = DEFAULT_MODELS_DIR)]
    models_dir: PathBuf,
}

#[derive(Debug, Parser)]
struct ModelDownloadArgs {
    #[arg(long, default_value = DEFAULT_MODELS_DIR)]
    models_dir: PathBuf,
    #[arg(long)]
    dry_run: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SetupTarget {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

impl SetupTarget {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendMode {
    NativeCpu,
    Qwentts,
    #[cfg(feature = "ffi")]
    Ffi,
}

#[derive(Debug, Parser)]
struct SynthArgs {
    #[arg(long)]
    text: String,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = "Chinese")]
    lang: String,
    #[arg(long)]
    speaker: Option<String>,
    #[arg(long)]
    instruct: Option<String>,
    #[arg(long)]
    flash_attention: bool,
    #[arg(long)]
    clamp_fp16: bool,
    #[arg(long, default_value = "auto")]
    device: DeviceKind,
    #[arg(long, default_value = "native-cpu")]
    backend: BackendMode,
    #[arg(long = "bin")]
    qwen_tts_bin: Option<PathBuf>,
    #[arg(long)]
    talker: Option<PathBuf>,
    #[arg(long)]
    codec: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(err) => {
            let exit_code = match err.kind() {
                ErrorKind::DisplayHelp
                | ErrorKind::DisplayVersion
                | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => ExitCode::SUCCESS,
                _ => ExitCode::FAILURE,
            };
            let _ = err.print();
            return exit_code;
        }
    };

    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Backend(args) => backend(&args),
        Command::Inspect(args) => inspect(&args),
        Command::Synth(args) => synth(&args),
        Command::Models(args) => models(&args),
        Command::Graph => {
            graph();
            Ok(())
        }
        Command::SetupScript(args) => {
            setup_script(&args);
            Ok(())
        }
    }
}

fn inspect(args: &InspectArgs) -> Result<(), String> {
    for (label, path) in [("talker", &args.talker), ("codec", &args.codec)] {
        let probe = GgufProbe::open(path).map_err(|err| format!("{label}: {err}"))?;
        println!("{label}: {}", path.display());
        println!("  version: {}", probe.version);
        println!("  tensors: {}", probe.tensor_count);
        println!("  metadata kv: {}", probe.metadata_kv_count);
    }
    Ok(())
}

fn synth(args: &SynthArgs) -> Result<(), String> {
    let project_root = project_root_dir();
    let should_download_defaults = args.talker.is_none()
        && args.codec.is_none()
        && env::var_os("QWEN_TTS_TALKER").is_none()
        && env::var_os("QWEN_TTS_CODEC").is_none();
    if should_download_defaults {
        let status = default_model_status(DEFAULT_MODELS_DIR);
        if !status.is_complete() {
            println!("default GGUF models missing; downloading to {DEFAULT_MODELS_DIR} ...");
            ensure_models_with_cli_progress(DEFAULT_MODELS_DIR).map_err(|err| err.to_string())?;
        }
    }

    let talker = path_from_arg_env_or_default(
        args.talker.as_ref(),
        "QWEN_TTS_TALKER",
        DEFAULT_TALKER_MODEL,
    );
    let codec =
        path_from_arg_env_or_default(args.codec.as_ref(), "QWEN_TTS_CODEC", DEFAULT_CODEC_MODEL);

    let mut scheduler = Scheduler::new();
    match args.backend {
        BackendMode::NativeCpu => scheduler.register(CpuBackend::new()),
        BackendMode::Qwentts => {
            let qwen_tts_bin =
                resolve_backend_executable(&project_root, args.qwen_tts_bin.as_ref())?;
            scheduler.register(ExternalQwenTtsBackend::new(qwen_tts_bin, args.device));
        }
        #[cfg(feature = "ffi")]
        BackendMode::Ffi => {
            let mut ffi_bk = FfiBackend::new(talker.clone(), codec.clone(), args.device);
            ffi_bk.use_flash_attn = args.flash_attention;
            ffi_bk.clamp_fp16 = args.clamp_fp16;
            scheduler.register(ffi_bk);
        }
    }

    let request = SynthesisRequest {
        text: args.text.clone(),
        language: args.lang.clone(),
        speaker: args.speaker.clone(),
        instruct: args.instruct.clone(),
        out_path: args.out.clone().unwrap_or_else(default_voice_output_path),
        device: args.device,
        models: TtsModelSet::new(talker, codec),
    };

    let response = scheduler
        .synthesize(&request)
        .map_err(|err| err.to_string())?;
    println!("generated: {}", response.wav_path.display());
    println!(
        "format: {} Hz / {} channel(s)",
        response.sample_rate_hz, response.channels
    );
    println!("backend: {}", response.backend_name);
    if response.backend_name == "native-cpu-rust" {
        println!("note: native CPU Rust backend is an experimental rewrite milestone");
    }
    Ok(())
}

fn backend(args: &BackendArgs) -> Result<(), String> {
    let project_root = project_root_dir();
    match args.command {
        BackendCommand::Status => {
            print_backend_status(&backend_status(&project_root, None));
            Ok(())
        }
        BackendCommand::Setup => {
            println!(
                "setting up qwentts.cpp backend under {}",
                project_root.display()
            );
            let status = setup_qwentts_backend(&project_root).map_err(|err| err.to_string())?;
            print_backend_status(&status);
            Ok(())
        }
    }
}

fn resolve_backend_executable(
    project_root: &std::path::Path,
    explicit: Option<&PathBuf>,
) -> Result<PathBuf, String> {
    find_qwentts_executable(project_root, explicit.map(PathBuf::as_path)).ok_or_else(|| {
        let expected = default_backend_executable(project_root);
        format!(
            "qwen-tts backend executable not found. Expected {}. Run `qwen-tts backend setup` or set QWEN_TTS_BIN.",
            expected.display()
        )
    })
}

fn print_backend_status(status: &BackendStatus) {
    println!("backend source: {}", status.source_dir.display());
    println!(
        "expected executable: {}",
        status.expected_executable.display()
    );
    match &status.resolved_executable {
        Some(path) => println!("resolved executable: {}", path.display()),
        None => println!("resolved executable: missing"),
    }
}

fn models(args: &ModelsArgs) -> Result<(), String> {
    match &args.command {
        ModelsCommand::Status(args) => {
            print_model_status(&args.models_dir);
            Ok(())
        }
        ModelsCommand::Download(args) => {
            if args.dry_run {
                println!(
                    "would download default GGUF models to {}",
                    args.models_dir.display()
                );
                for file in DEFAULT_MODEL_FILES {
                    println!(
                        "{}: {} -> {}",
                        file.role,
                        file.url,
                        file.path_in(&args.models_dir).display()
                    );
                }
                return Ok(());
            }

            let existing_status = default_model_status(&args.models_dir);
            if existing_status.is_complete() {
                println!(
                    "default GGUF models already exist in {}; no download needed",
                    args.models_dir.display()
                );
                print_model_status_from(&existing_status);
                return Ok(());
            }

            println!(
                "{} default GGUF model(s) missing; downloading to {}",
                existing_status.missing_files().len(),
                args.models_dir.display()
            );
            let status =
                ensure_models_with_cli_progress(&args.models_dir).map_err(|err| err.to_string())?;
            print_model_status_from(&status);
            Ok(())
        }
    }
}

fn print_model_status(models_dir: &PathBuf) {
    let status = default_model_status(models_dir);
    print_model_status_from(&status);
}

fn print_model_status_from(status: &qwen_tts_runtime::DefaultModelStatus) {
    println!("models dir: {}", status.models_dir.display());
    for file in &status.files {
        let state = if file.exists { "present" } else { "missing" };
        let size = file
            .size_bytes
            .map_or_else(|| "-".to_owned(), |value| format!("{value} bytes"));
        println!(
            "{}: {} ({}) [{}]",
            file.file.role,
            file.path.display(),
            size,
            state
        );
    }
}

fn ensure_models_with_cli_progress(
    models_dir: impl AsRef<std::path::Path>,
) -> qwen_tts_runtime::ModelDownloadResult<qwen_tts_runtime::DefaultModelStatus> {
    ensure_default_models_with_progress(models_dir, |progress| {
        print_download_progress(&progress);
    })
}

fn print_download_progress(progress: &ModelDownloadProgress) {
    let marker = if progress.finished {
        "done"
    } else {
        "downloading"
    };
    match progress.total_bytes {
        Some(total_bytes) if total_bytes > 0 => {
            let percent = progress.downloaded_bytes.saturating_mul(100) / total_bytes;
            eprint!(
                "\r{marker}: {} {}/{} bytes ({}%)",
                progress.file_name, progress.downloaded_bytes, total_bytes, percent
            );
        }
        _ => {
            eprint!(
                "\r{marker}: {} {} bytes",
                progress.file_name, progress.downloaded_bytes
            );
        }
    }
    let _ = io::stderr().flush();
    if progress.finished {
        eprintln!();
    }
}

fn graph() {
    let graph = TtsGraph::qwen_tts_default();
    for (index, node) in graph.nodes.iter().enumerate() {
        println!("{:02}. {:?} - {}", index + 1, node.kind, node.name);
    }
}

fn setup_script(args: &SetupScriptArgs) {
    println!("#!/usr/bin/env bash");
    println!("set -euo pipefail");
    println!("# target: {}", args.target.as_str());
    println!("mkdir -p vendor models");
    println!("if [ ! -d vendor/qwentts.cpp ]; then git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp vendor/qwentts.cpp; fi");
    println!("git -C vendor/qwentts.cpp submodule update --init --recursive");
    println!("huggingface-cli download Serveurperso/Qwen3-TTS-GGUF qwen-talker-1.7b-base-Q8_0.gguf qwen-tokenizer-12hz-Q8_0.gguf --local-dir models");
    println!(
        "cmake -S vendor/qwentts.cpp -B vendor/qwentts.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON"
    );
    println!("cmake --build vendor/qwentts.cpp/build --config Release --target qwen-tts -j");
    println!("echo 'Run cargo run -p qwen-tts-cli -- synth --text ...'");
}

fn project_root_dir() -> PathBuf {
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn path_from_arg_env_or_default(
    arg: Option<&PathBuf>,
    env_name: &str,
    default_value: &str,
) -> PathBuf {
    arg.cloned()
        .or_else(|| env::var_os(env_name).map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(default_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse<const N: usize>(args: [&str; N]) -> Cli {
        Cli::try_parse_from(args).expect("CLI args should parse")
    }

    #[test]
    fn parses_inspect_args() {
        let cli = parse([
            "qwen-tts",
            "inspect",
            "--talker",
            "talker.gguf",
            "--codec",
            "codec.gguf",
        ]);

        let Command::Inspect(args) = cli.command else {
            panic!("expected inspect command");
        };
        assert_eq!(args.talker, PathBuf::from("talker.gguf"));
        assert_eq!(args.codec, PathBuf::from("codec.gguf"));
    }

    #[test]
    fn parses_backend_status() {
        let cli = parse(["qwen-tts", "backend", "status"]);

        let Command::Backend(args) = cli.command else {
            panic!("expected backend command");
        };
        assert!(matches!(args.command, BackendCommand::Status));
    }

    #[test]
    fn parses_synth_defaults_without_resolving_env() {
        let cli = parse(["qwen-tts", "synth", "--text", "hello"]);

        let Command::Synth(args) = cli.command else {
            panic!("expected synth command");
        };
        assert_eq!(args.text, "hello");
        assert_eq!(args.out, None);
        assert_eq!(args.lang, "Chinese");
        assert_eq!(args.speaker, None);
        assert_eq!(args.device, DeviceKind::Auto);
        assert_eq!(args.backend, BackendMode::NativeCpu);
        assert_eq!(args.qwen_tts_bin, None);
        assert_eq!(args.talker, None);
        assert_eq!(args.codec, None);
    }

    #[test]
    fn parses_synth_overrides() {
        let cli = parse([
            "qwen-tts",
            "synth",
            "--text",
            "hello",
            "--lang",
            "English",
            "--speaker",
            "alice",
            "--device",
            "cuda",
            "--backend",
            "qwentts",
            "--out",
            "speech.wav",
            "--bin",
            "qwen-tts-bin",
            "--talker",
            "talker.gguf",
            "--codec",
            "codec.gguf",
        ]);

        let Command::Synth(args) = cli.command else {
            panic!("expected synth command");
        };
        assert_eq!(args.lang, "English");
        assert_eq!(args.speaker, Some(String::from("alice")));
        assert_eq!(args.device, DeviceKind::Cuda);
        assert_eq!(args.backend, BackendMode::Qwentts);
        assert_eq!(args.out, Some(PathBuf::from("speech.wav")));
        assert_eq!(args.qwen_tts_bin, Some(PathBuf::from("qwen-tts-bin")));
        assert_eq!(args.talker, Some(PathBuf::from("talker.gguf")));
        assert_eq!(args.codec, Some(PathBuf::from("codec.gguf")));
    }

    #[test]
    fn missing_command_displays_help() {
        let err = Cli::try_parse_from(["qwen-tts"]).expect_err("missing command should show help");
        assert_eq!(
            err.kind(),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
    }

    #[test]
    fn parses_setup_script_default_target() {
        let cli = parse(["qwen-tts", "setup-script"]);

        let Command::SetupScript(args) = cli.command else {
            panic!("expected setup-script command");
        };
        assert_eq!(args.target, SetupTarget::Cpu);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn parses_synth_backend_ffi() {
        let cli = parse(["qwen-tts", "synth", "--text", "hi", "--backend", "ffi"]);

        let Command::Synth(args) = cli.command else {
            panic!("expected synth command");
        };
        assert_eq!(args.backend, BackendMode::Ffi);
    }

    #[test]
    fn parses_model_download_dry_run() {
        let cli = parse(["qwen-tts", "models", "download", "--dry-run"]);

        let Command::Models(args) = cli.command else {
            panic!("expected models command");
        };
        let ModelsCommand::Download(download) = args.command else {
            panic!("expected download command");
        };
        assert!(download.dry_run);
        assert_eq!(download.models_dir, PathBuf::from(DEFAULT_MODELS_DIR));
    }
}
