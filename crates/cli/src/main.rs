use clap::{error::ErrorKind, Parser, Subcommand, ValueEnum};
use qwen_tts_core::{graph::TtsGraph, GgufProbe, TtsModelSet};
use qwen_tts_runtime::{
    default_model_status, ensure_default_models, DeviceKind, ExternalQwenTtsBackend, Scheduler,
    SynthesisRequest, DEFAULT_MODELS_DIR, DEFAULT_MODEL_FILES,
};
use std::{env, path::PathBuf, process::ExitCode};

const DEFAULT_QWEN_TTS_BIN: &str = "./vendor/qwentts.cpp/build/bin/qwen-tts";
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
    Inspect(InspectArgs),
    Graph,
    Models(ModelsArgs),
    SetupScript(SetupScriptArgs),
    Synth(SynthArgs),
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

#[derive(Debug, Parser)]
struct SynthArgs {
    #[arg(long)]
    text: String,
    #[arg(long, default_value = "output.wav")]
    out: PathBuf,
    #[arg(long, default_value = "Chinese")]
    lang: String,
    #[arg(long)]
    speaker: Option<String>,
    #[arg(long, default_value = "auto")]
    device: DeviceKind,
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
    let should_download_defaults = args.talker.is_none()
        && args.codec.is_none()
        && env::var_os("QWEN_TTS_TALKER").is_none()
        && env::var_os("QWEN_TTS_CODEC").is_none();
    if should_download_defaults {
        let status = default_model_status(DEFAULT_MODELS_DIR);
        if !status.is_complete() {
            println!("default GGUF models missing; downloading to {DEFAULT_MODELS_DIR} ...");
            ensure_default_models(DEFAULT_MODELS_DIR).map_err(|err| err.to_string())?;
        }
    }

    let qwen_tts_bin = path_from_arg_env_or_default(
        args.qwen_tts_bin.as_ref(),
        "QWEN_TTS_BIN",
        DEFAULT_QWEN_TTS_BIN,
    );
    let talker = path_from_arg_env_or_default(
        args.talker.as_ref(),
        "QWEN_TTS_TALKER",
        DEFAULT_TALKER_MODEL,
    );
    let codec =
        path_from_arg_env_or_default(args.codec.as_ref(), "QWEN_TTS_CODEC", DEFAULT_CODEC_MODEL);

    let mut scheduler = Scheduler::new();
    scheduler.register(ExternalQwenTtsBackend::new(qwen_tts_bin, args.device));

    let request = SynthesisRequest {
        text: args.text.clone(),
        language: args.lang.clone(),
        speaker: args.speaker.clone(),
        out_path: args.out.clone(),
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
    Ok(())
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

            let status = ensure_default_models(&args.models_dir).map_err(|err| err.to_string())?;
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
    println!("if [ ! -d vendor/qwentts.cpp ]; then git clone https://github.com/ServeurpersoCom/qwentts.cpp vendor/qwentts.cpp; fi");
    println!("huggingface-cli download Serveurperso/Qwen3-TTS-GGUF qwen-talker-1.7b-base-Q8_0.gguf qwen-tokenizer-12hz-Q8_0.gguf --local-dir models");
    println!("cmake -S vendor/qwentts.cpp -B vendor/qwentts.cpp/build -DCMAKE_BUILD_TYPE=Release");
    println!("cmake --build vendor/qwentts.cpp/build --config Release -j --target qwen-tts");
    println!("echo 'Run cargo run -p qwen-tts-cli -- synth --text ...'");
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
    fn parses_synth_defaults_without_resolving_env() {
        let cli = parse(["qwen-tts", "synth", "--text", "hello"]);

        let Command::Synth(args) = cli.command else {
            panic!("expected synth command");
        };
        assert_eq!(args.text, "hello");
        assert_eq!(args.out, PathBuf::from("output.wav"));
        assert_eq!(args.lang, "Chinese");
        assert_eq!(args.speaker, None);
        assert_eq!(args.device, DeviceKind::Auto);
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
        assert_eq!(args.out, PathBuf::from("speech.wav"));
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
