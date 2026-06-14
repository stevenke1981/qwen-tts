use eframe::egui;
use qwen_tts_core::TtsModelSet;
use qwen_tts_runtime::{
    backend_status, default_backend_executable, default_model_status,
    ensure_default_models_with_progress, find_qwentts_executable, setup_qwentts_backend,
    BackendStatus, DeviceKind, ExternalQwenTtsBackend, ModelDownloadProgress, Scheduler,
    SynthesisRequest, DEFAULT_CODEC_FILE, DEFAULT_MODELS_DIR, DEFAULT_TALKER_FILE,
};
use std::{
    fs,
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([920.0, 680.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Qwen TTS 語音合成",
        options,
        Box::new(|cc| {
            configure_fonts(&cc.egui_ctx);
            Box::new(QwenTtsApp::default())
        }),
    )
}

fn configure_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    if let Some((name, data)) = load_cjk_font() {
        fonts.font_data.insert(name.clone(), data);
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(family).or_default().push(name.clone());
        }
    }

    ctx.set_fonts(fonts);
}

#[cfg(target_os = "windows")]
fn load_cjk_font() -> Option<(String, egui::FontData)> {
    let candidates = [
        ("Microsoft-JhengHei", "C:\\Windows\\Fonts\\msjh.ttc"),
        ("Microsoft-JhengHei-Bold", "C:\\Windows\\Fonts\\msjhbd.ttc"),
        ("MingLiU", "C:\\Windows\\Fonts\\mingliu.ttc"),
    ];

    candidates.iter().find_map(|(name, path)| {
        fs::read(path)
            .ok()
            .map(|bytes| ((*name).to_owned(), egui::FontData::from_owned(bytes)))
    })
}

#[cfg(not(target_os = "windows"))]
fn load_cjk_font() -> Option<(String, egui::FontData)> {
    None
}

#[derive(Debug)]
enum WorkerMessage {
    BackendSetupFinished(Result<String, String>),
    DownloadProgress(String),
    DownloadFinished(Result<String, String>),
    SynthesisFinished(Result<String, String>),
}

#[allow(clippy::struct_excessive_bools)]
struct QwenTtsApp {
    text: String,
    language: String,
    speaker: String,
    qwen_tts_bin: String,
    models_dir: String,
    output_path: String,
    device: DeviceKind,
    status: String,
    busy: bool,
    prompted_for_missing_models: bool,
    prompted_for_missing_backend: bool,
    show_download_prompt: bool,
    show_backend_prompt: bool,
    receiver: Option<Receiver<WorkerMessage>>,
}

impl Default for QwenTtsApp {
    fn default() -> Self {
        let project_root = project_root_dir();
        let qwen_tts_bin = find_qwentts_executable(&project_root, None)
            .unwrap_or_else(|| default_backend_executable(&project_root));
        Self {
            text: "你好，這是 Qwen TTS GUI 測試。".to_owned(),
            language: "Chinese".to_owned(),
            speaker: String::new(),
            qwen_tts_bin: qwen_tts_bin.display().to_string(),
            models_dir: project_models_dir().display().to_string(),
            output_path: "output.wav".to_owned(),
            device: DeviceKind::Auto,
            status: "就緒".to_owned(),
            busy: false,
            prompted_for_missing_models: false,
            prompted_for_missing_backend: false,
            show_download_prompt: false,
            show_backend_prompt: false,
            receiver: None,
        }
    }
}

impl eframe::App for QwenTtsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive_worker_messages();
        self.prompt_for_missing_backend_once();
        self.prompt_for_missing_models_once();

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("Qwen TTS 語音合成");
                ui.separator();
                ui.label(&self.status);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            self.backend_section(ui);
            ui.separator();
            self.model_section(ui);
            ui.separator();
            self.synthesis_section(ui);
            ui.separator();
            self.run_section(ui);
        });

        self.download_prompt_window(ctx);
        self.backend_prompt_window(ctx);

        if self.busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

impl QwenTtsApp {
    fn receive_worker_messages(&mut self) {
        let Some(receiver) = self.receiver.take() else {
            return;
        };

        match receiver.try_recv() {
            Ok(WorkerMessage::BackendSetupFinished(result)) => {
                self.busy = false;
                self.status = match result {
                    Ok(path) => {
                        self.qwen_tts_bin.clone_from(&path);
                        format!("backend 已就緒：{path}")
                    }
                    Err(message) => format!("Backend setup failed: {message}"),
                };
            }
            Ok(WorkerMessage::DownloadProgress(message)) => {
                self.status = message;
                self.receiver = Some(receiver);
            }
            Ok(WorkerMessage::DownloadFinished(result)) => {
                self.busy = false;
                self.status = match result {
                    Ok(message) => message,
                    Err(message) => format!("Download failed: {message}"),
                };
            }
            Ok(WorkerMessage::SynthesisFinished(result)) => {
                self.busy = false;
                self.status = match result {
                    Ok(message) => message,
                    Err(message) => format!("Synthesis failed: {message}"),
                };
            }
            Err(mpsc::TryRecvError::Empty) => {
                self.receiver = Some(receiver);
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.busy = false;
                self.set_status("Worker disconnected");
            }
        }
    }

    fn set_status(&mut self, value: &str) {
        self.status.clear();
        self.status.push_str(value);
    }

    fn prompt_for_missing_models_once(&mut self) {
        if self.prompted_for_missing_models || self.busy {
            return;
        }

        self.prompted_for_missing_models = true;
        let status = default_model_status(&self.models_dir);
        if !status.is_complete() {
            self.show_download_prompt = true;
            self.status = format!(
                "專案 models 資料夾缺少 {} 個預設 GGUF 模型",
                status.missing_files().len()
            );
        }
    }

    fn prompt_for_missing_backend_once(&mut self) {
        if self.prompted_for_missing_backend || self.busy {
            return;
        }

        self.prompted_for_missing_backend = true;
        let status = current_backend_status(Some(&self.qwen_tts_bin));
        if !status.is_available() {
            self.show_backend_prompt = true;
            "找不到 qwentts.cpp backend，請先建置 backend".clone_into(&mut self.status);
        }
    }

    fn backend_prompt_window(&mut self, ctx: &egui::Context) {
        if !self.show_backend_prompt {
            return;
        }

        egui::Window::new("建置 qwentts.cpp backend")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("找不到 qwen-tts backend 執行檔。");
                ui.label("需要先建置 qwentts.cpp CPU backend 才能生成語音。");
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!self.busy, egui::Button::new("建置 backend"))
                        .clicked()
                    {
                        self.show_backend_prompt = false;
                        self.start_backend_setup();
                    }
                    if ui.button("稍後").clicked() {
                        self.show_backend_prompt = false;
                        self.set_status("已略過 backend 建置，可稍後按「建置 backend」");
                    }
                });
            });
    }

    fn backend_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Backend");
        let status = current_backend_status(Some(&self.qwen_tts_bin));
        egui::Grid::new("backend_status_grid")
            .num_columns(2)
            .spacing([18.0, 6.0])
            .show(ui, |ui| {
                ui.label("狀態");
                ui.label(if status.is_available() {
                    "已就緒"
                } else {
                    "缺少 qwen-tts backend"
                });
                ui.end_row();

                ui.label("執行檔");
                ui.monospace(
                    status
                        .resolved_executable
                        .as_ref()
                        .unwrap_or(&status.expected_executable)
                        .display()
                        .to_string(),
                );
                ui.end_row();
            });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy, egui::Button::new("重新偵測 backend"))
                .clicked()
            {
                self.refresh_backend_path();
            }
            if ui
                .add_enabled(!self.busy, egui::Button::new("建置 backend"))
                .clicked()
            {
                self.start_backend_setup();
            }
        });
    }

    fn download_prompt_window(&mut self, ctx: &egui::Context) {
        if !self.show_download_prompt {
            return;
        }

        egui::Window::new("下載預設 GGUF 模型")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("專案 models 資料夾缺少 Qwen TTS 預設模型。");
                ui.label(format!("下載位置：{}", self.models_dir));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!self.busy, egui::Button::new("下載到 models"))
                        .clicked()
                    {
                        self.show_download_prompt = false;
                        self.start_download();
                    }
                    if ui.button("稍後").clicked() {
                        self.show_download_prompt = false;
                        self.set_status("已略過自動下載，可稍後按「下載 GGUF」");
                    }
                });
            });
    }

    fn model_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("模型");
        ui.horizontal(|ui| {
            ui.label("資料夾");
            ui.text_edit_singleline(&mut self.models_dir);
            if ui.button("Refresh").clicked() {
                self.refresh_model_status();
            }
            if ui
                .add_enabled(!self.busy, egui::Button::new("下載 GGUF"))
                .clicked()
            {
                self.start_download();
            }
        });

        let status = default_model_status(&self.models_dir);
        egui::Grid::new("model_status_grid")
            .num_columns(4)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.strong("角色");
                ui.strong("檔案");
                ui.strong("狀態");
                ui.strong("大小");
                ui.end_row();

                for file in status.files {
                    ui.label(file.file.role);
                    ui.monospace(file.path.display().to_string());
                    ui.label(if file.exists { "已存在" } else { "缺少" });
                    ui.label(file.size_bytes.map_or_else(|| "-".to_owned(), format_bytes));
                    ui.end_row();
                }
            });
    }

    fn synthesis_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("語音合成");
        ui.label("文字");
        ui.add(
            egui::TextEdit::multiline(&mut self.text)
                .desired_rows(7)
                .lock_focus(true),
        );

        egui::Grid::new("synthesis_form")
            .num_columns(2)
            .spacing([18.0, 8.0])
            .show(ui, |ui| {
                ui.label("語言");
                ui.text_edit_singleline(&mut self.language);
                ui.end_row();

                ui.label("說話者");
                ui.text_edit_singleline(&mut self.speaker);
                ui.end_row();

                ui.label("qwen-tts 執行檔");
                ui.text_edit_singleline(&mut self.qwen_tts_bin);
                ui.end_row();

                ui.label("輸出 WAV");
                ui.text_edit_singleline(&mut self.output_path);
                ui.end_row();
            });

        ui.horizontal_wrapped(|ui| {
            ui.label("裝置");
            for device in [
                DeviceKind::Auto,
                DeviceKind::Cpu,
                DeviceKind::Cuda,
                DeviceKind::Rocm,
                DeviceKind::Metal,
                DeviceKind::Wgpu,
                DeviceKind::Sycl,
            ] {
                ui.radio_value(&mut self.device, device, device.to_string());
            }
        });
    }

    fn run_section(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy, egui::Button::new("開始合成"))
                .clicked()
            {
                self.start_synthesis();
            }
            if ui.button("重設預設值").clicked() {
                *self = Self::default();
            }
        });
    }

    fn refresh_model_status(&mut self) {
        let status = default_model_status(&self.models_dir);
        self.status = if status.is_complete() {
            "預設 GGUF 模型已存在".to_owned()
        } else {
            format!("缺少 {} 個預設 GGUF 模型", status.missing_files().len())
        };
    }

    fn start_download(&mut self) {
        let models_dir = PathBuf::from(self.models_dir.clone());
        let status = default_model_status(&models_dir);
        if status.is_complete() {
            self.set_status("預設 GGUF 模型已存在，不需要下載");
            return;
        }

        let (sender, receiver) = mpsc::channel();
        self.receiver = Some(receiver);
        self.busy = true;
        self.status = format!(
            "缺少 {} 個預設 GGUF 模型，準備下載...",
            status.missing_files().len()
        );

        thread::spawn(move || {
            let progress_sender = sender.clone();
            let result = ensure_default_models_with_progress(&models_dir, |progress| {
                let _ = progress_sender.send(WorkerMessage::DownloadProgress(
                    format_download_progress(&progress),
                ));
            })
            .map(|status| {
                format!(
                    "已下載/確認 {} 個模型檔案：{}",
                    status.files.len(),
                    status.models_dir.display()
                )
            })
            .map_err(|err| err.to_string());
            let _ = sender.send(WorkerMessage::DownloadFinished(result));
        });
    }

    fn refresh_backend_path(&mut self) {
        let status = current_backend_status(Some(&self.qwen_tts_bin));
        if let Some(path) = status.resolved_executable {
            self.qwen_tts_bin = path.display().to_string();
            "backend 已就緒".clone_into(&mut self.status);
        } else {
            self.qwen_tts_bin = status.expected_executable.display().to_string();
            "找不到 qwentts.cpp backend".clone_into(&mut self.status);
        }
    }

    fn start_backend_setup(&mut self) {
        let project_root = project_root_dir();
        let (sender, receiver) = mpsc::channel();
        self.receiver = Some(receiver);
        self.busy = true;
        self.set_status("正在建置 qwentts.cpp backend，這可能需要幾分鐘...");

        thread::spawn(move || {
            let result = setup_qwentts_backend(&project_root)
                .map(|status| {
                    status
                        .resolved_executable
                        .unwrap_or(status.expected_executable)
                        .display()
                        .to_string()
                })
                .map_err(|err| err.to_string());
            let _ = sender.send(WorkerMessage::BackendSetupFinished(result));
        });
    }

    fn start_synthesis(&mut self) {
        let text = self.text.trim().to_owned();
        if text.is_empty() {
            self.set_status("文字不能空白");
            return;
        }

        let models_dir = PathBuf::from(self.models_dir.clone());
        let request = SynthesisRequest {
            text,
            language: self.language.clone(),
            speaker: non_empty_string(&self.speaker),
            out_path: PathBuf::from(self.output_path.clone()),
            device: self.device,
            models: TtsModelSet::new(
                models_dir.join(DEFAULT_TALKER_FILE),
                models_dir.join(DEFAULT_CODEC_FILE),
            ),
        };
        let qwen_tts_bin = PathBuf::from(self.qwen_tts_bin.clone());
        let (sender, receiver) = mpsc::channel();
        self.receiver = Some(receiver);
        self.busy = true;
        self.set_status("正在合成...");

        thread::spawn(move || {
            let result = run_synthesis(qwen_tts_bin, &request, &sender);
            let _ = sender.send(WorkerMessage::SynthesisFinished(result));
        });
    }
}

fn run_synthesis(
    qwen_tts_bin: PathBuf,
    request: &SynthesisRequest,
    sender: &Sender<WorkerMessage>,
) -> Result<String, String> {
    let models_dir = request
        .models
        .talker
        .path
        .parent()
        .map_or_else(|| PathBuf::from(DEFAULT_MODELS_DIR), PathBuf::from);
    let status = default_model_status(&models_dir);
    if !status.is_complete() {
        ensure_default_models_with_progress(models_dir, |progress| {
            let _ = sender.send(WorkerMessage::DownloadProgress(format_download_progress(
                &progress,
            )));
        })
        .map_err(|err| err.to_string())?;
    }

    let mut scheduler = Scheduler::new();
    scheduler.register(ExternalQwenTtsBackend::new(qwen_tts_bin, request.device));
    let response = scheduler
        .synthesize(request)
        .map_err(|err| err.to_string())?;

    Ok(format!(
        "已產生 {}（{} Hz，{} 聲道）",
        response.wav_path.display(),
        response.sample_rate_hz,
        response.channels
    ))
}

fn current_backend_status(configured: Option<&str>) -> BackendStatus {
    let configured_path = configured
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from);
    backend_status(project_root_dir(), configured_path.as_deref())
}

fn non_empty_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn format_bytes(value: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    if value >= GIB {
        let whole = value / GIB;
        let fraction = (value % GIB) * 100 / GIB;
        format!("{whole}.{fraction:02} GiB")
    } else if value >= MIB {
        let whole = value / MIB;
        let fraction = (value % MIB) * 10 / MIB;
        format!("{whole}.{fraction} MiB")
    } else {
        format!("{value} bytes")
    }
}

fn format_download_progress(progress: &ModelDownloadProgress) -> String {
    let action = if progress.finished {
        "下載完成"
    } else {
        "正在下載"
    };
    match progress.total_bytes {
        Some(total_bytes) if total_bytes > 0 => {
            let percent = progress.downloaded_bytes.saturating_mul(100) / total_bytes;
            format!(
                "{action} {}：{} / {} bytes（{}%）",
                progress.file_name, progress.downloaded_bytes, total_bytes, percent
            )
        }
        _ => format!(
            "{action} {}：{} bytes",
            progress.file_name, progress.downloaded_bytes
        ),
    }
}

fn project_models_dir() -> PathBuf {
    project_root_dir().join("models")
}

fn project_root_dir() -> PathBuf {
    if let Ok(current_dir) = std::env::current_dir() {
        if current_dir.join("Cargo.toml").is_file() {
            return current_dir;
        }
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(root) = find_ancestor_with_manifest(&exe_path) {
            return root;
        }
    }

    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn find_ancestor_with_manifest(path: &std::path::Path) -> Option<PathBuf> {
    let start = if path.is_file() {
        path.parent().unwrap_or(path)
    } else {
        path
    };

    start
        .ancestors()
        .find(|candidate| candidate.join("Cargo.toml").is_file())
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_speaker_is_none() {
        assert_eq!(non_empty_string("   "), None);
        assert_eq!(non_empty_string("alice"), Some("alice".to_owned()));
    }

    #[test]
    fn device_display_round_trips() {
        for device in [DeviceKind::Auto, DeviceKind::Cpu, DeviceKind::Cuda] {
            assert_eq!(device.to_string().parse::<DeviceKind>().unwrap(), device);
        }
    }

    #[test]
    fn formats_download_progress_with_percent() {
        let message = format_download_progress(&ModelDownloadProgress {
            role: "talker",
            file_name: "talker.gguf",
            downloaded_bytes: 50,
            total_bytes: Some(100),
            finished: false,
        });

        assert!(message.contains("50%"));
    }

    #[test]
    fn finds_project_root_from_nested_path() {
        let app_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace = app_dir
            .parent()
            .and_then(std::path::Path::parent)
            .expect("app crate should live under workspace/crates/app");
        let nested = workspace.join("dist").join("qwen-tts-gui.exe");

        assert_eq!(find_ancestor_with_manifest(&nested).unwrap(), workspace);
    }
}
