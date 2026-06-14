use eframe::egui;
use qwen_tts_core::TtsModelSet;
use qwen_tts_runtime::{
    default_model_status, ensure_default_models, DeviceKind, ExternalQwenTtsBackend, Scheduler,
    SynthesisRequest, DEFAULT_CODEC_FILE, DEFAULT_MODELS_DIR, DEFAULT_TALKER_FILE,
};
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver},
    thread,
};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([920.0, 680.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Qwen TTS",
        options,
        Box::new(|_cc| Box::new(QwenTtsApp::default())),
    )
}

#[derive(Debug)]
enum WorkerMessage {
    DownloadFinished(Result<String, String>),
    SynthesisFinished(Result<String, String>),
}

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
    receiver: Option<Receiver<WorkerMessage>>,
}

impl Default for QwenTtsApp {
    fn default() -> Self {
        Self {
            text: "你好，這是 Qwen TTS GUI 測試。".to_owned(),
            language: "Chinese".to_owned(),
            speaker: String::new(),
            qwen_tts_bin: "./vendor/qwentts.cpp/build/bin/qwen-tts".to_owned(),
            models_dir: DEFAULT_MODELS_DIR.to_owned(),
            output_path: "output.wav".to_owned(),
            device: DeviceKind::Auto,
            status: "Ready".to_owned(),
            busy: false,
            receiver: None,
        }
    }
}

impl eframe::App for QwenTtsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive_worker_messages();

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("Qwen TTS");
                ui.separator();
                ui.label(&self.status);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            self.model_section(ui);
            ui.separator();
            self.synthesis_section(ui);
            ui.separator();
            self.run_section(ui);
        });

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

    fn model_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Models");
        ui.horizontal(|ui| {
            ui.label("Folder");
            ui.text_edit_singleline(&mut self.models_dir);
            if ui.button("Refresh").clicked() {
                self.refresh_model_status();
            }
            if ui
                .add_enabled(!self.busy, egui::Button::new("Download GGUF"))
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
                ui.strong("Role");
                ui.strong("File");
                ui.strong("Status");
                ui.strong("Size");
                ui.end_row();

                for file in status.files {
                    ui.label(file.file.role);
                    ui.monospace(file.path.display().to_string());
                    ui.label(if file.exists { "Present" } else { "Missing" });
                    ui.label(file.size_bytes.map_or_else(|| "-".to_owned(), format_bytes));
                    ui.end_row();
                }
            });
    }

    fn synthesis_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Synthesis");
        ui.label("Text");
        ui.add(
            egui::TextEdit::multiline(&mut self.text)
                .desired_rows(7)
                .lock_focus(true),
        );

        egui::Grid::new("synthesis_form")
            .num_columns(2)
            .spacing([18.0, 8.0])
            .show(ui, |ui| {
                ui.label("Language");
                ui.text_edit_singleline(&mut self.language);
                ui.end_row();

                ui.label("Speaker");
                ui.text_edit_singleline(&mut self.speaker);
                ui.end_row();

                ui.label("qwen-tts binary");
                ui.text_edit_singleline(&mut self.qwen_tts_bin);
                ui.end_row();

                ui.label("Output WAV");
                ui.text_edit_singleline(&mut self.output_path);
                ui.end_row();
            });

        ui.horizontal_wrapped(|ui| {
            ui.label("Device");
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
                .add_enabled(!self.busy, egui::Button::new("Synthesize"))
                .clicked()
            {
                self.start_synthesis();
            }
            if ui.button("Reset defaults").clicked() {
                *self = Self::default();
            }
        });
    }

    fn refresh_model_status(&mut self) {
        let status = default_model_status(&self.models_dir);
        self.status = if status.is_complete() {
            "Default GGUF models are present".to_owned()
        } else {
            format!(
                "{} default GGUF model(s) missing",
                status.missing_files().len()
            )
        };
    }

    fn start_download(&mut self) {
        let models_dir = PathBuf::from(self.models_dir.clone());
        let (sender, receiver) = mpsc::channel();
        self.receiver = Some(receiver);
        self.busy = true;
        self.set_status("Downloading default GGUF models...");

        thread::spawn(move || {
            let result = ensure_default_models(&models_dir)
                .map(|status| {
                    format!(
                        "Downloaded/verified {} model files in {}",
                        status.files.len(),
                        status.models_dir.display()
                    )
                })
                .map_err(|err| err.to_string());
            let _ = sender.send(WorkerMessage::DownloadFinished(result));
        });
    }

    fn start_synthesis(&mut self) {
        let text = self.text.trim().to_owned();
        if text.is_empty() {
            self.set_status("Text cannot be empty");
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
        self.set_status("Synthesizing...");

        thread::spawn(move || {
            let result = run_synthesis(qwen_tts_bin, &request);
            let _ = sender.send(WorkerMessage::SynthesisFinished(result));
        });
    }
}

fn run_synthesis(qwen_tts_bin: PathBuf, request: &SynthesisRequest) -> Result<String, String> {
    let models_dir = request
        .models
        .talker
        .path
        .parent()
        .map_or_else(|| PathBuf::from(DEFAULT_MODELS_DIR), PathBuf::from);
    ensure_default_models(models_dir).map_err(|err| err.to_string())?;

    let mut scheduler = Scheduler::new();
    scheduler.register(ExternalQwenTtsBackend::new(qwen_tts_bin, request.device));
    let response = scheduler
        .synthesize(request)
        .map_err(|err| err.to_string())?;

    Ok(format!(
        "Generated {} ({} Hz, {} channel(s))",
        response.wav_path.display(),
        response.sample_rate_hz,
        response.channels
    ))
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
}
