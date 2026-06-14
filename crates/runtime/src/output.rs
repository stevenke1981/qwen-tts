use chrono::{DateTime, Local, TimeZone};
use std::fmt;
use std::path::PathBuf;

pub const DEFAULT_OUTPUT_DIR: &str = "output";
pub const DEFAULT_OUTPUT_PREFIX: &str = "voice";

#[must_use]
pub fn default_voice_output_path() -> PathBuf {
    default_voice_output_path_from(&Local::now())
}

#[must_use]
pub fn default_voice_output_path_from<Tz>(time: &DateTime<Tz>) -> PathBuf
where
    Tz: TimeZone,
    Tz::Offset: fmt::Display,
{
    PathBuf::from(DEFAULT_OUTPUT_DIR).join(format!(
        "{DEFAULT_OUTPUT_PREFIX}-{}-{:03}.wav",
        time.format("%Y%m%d-%H%M%S"),
        time.timestamp_subsec_millis()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    #[test]
    fn default_voice_output_path_uses_output_dir_and_timestamp() {
        let time = FixedOffset::east_opt(8 * 3600)
            .unwrap()
            .with_ymd_and_hms(2026, 6, 14, 16, 31, 16)
            .unwrap()
            + chrono::Duration::milliseconds(336);
        let path = default_voice_output_path_from(&time);

        assert_eq!(
            path,
            PathBuf::from("output").join("voice-20260614-163116-336.wav")
        );
    }
}
