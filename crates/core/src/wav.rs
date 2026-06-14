use crate::AudioSpec;
use std::{
    fmt,
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavMetadata {
    pub audio_format: u16,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub data_size_bytes: u32,
}

#[derive(Debug)]
pub enum WavValidationError {
    Io(io::Error),
    InvalidHeader(&'static str),
    MissingFmtChunk,
    MissingDataChunk,
    UnsupportedFormat(u16),
    EmptyDataChunk,
    MetadataMismatch {
        field: &'static str,
        expected: u64,
        actual: u64,
    },
}

impl fmt::Display for WavValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "WAV I/O error: {err}"),
            Self::InvalidHeader(message) => write!(f, "invalid WAV header: {message}"),
            Self::MissingFmtChunk => write!(f, "invalid WAV header: missing fmt chunk"),
            Self::MissingDataChunk => write!(f, "invalid WAV header: missing data chunk"),
            Self::UnsupportedFormat(format) => {
                write!(f, "unsupported WAV audio format: {format}")
            }
            Self::EmptyDataChunk => write!(f, "invalid WAV header: data chunk is empty"),
            Self::MetadataMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "WAV metadata mismatch for {field}: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for WavValidationError {}

impl From<io::Error> for WavValidationError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub type WavValidationResult<T> = Result<T, WavValidationError>;

/// Reads RIFF/WAVE header metadata without decoding audio samples.
///
/// # Errors
///
/// Returns an error when the file cannot be read, the RIFF/WAVE markers are
/// invalid, or required `fmt ` / `data` chunks are missing.
pub fn read_wav_metadata(path: impl AsRef<Path>) -> WavValidationResult<WavMetadata> {
    let mut file = File::open(path)?;
    let mut header = [0_u8; 12];
    file.read_exact(&mut header)?;

    if &header[0..4] != b"RIFF" {
        return Err(WavValidationError::InvalidHeader("missing RIFF marker"));
    }
    if &header[8..12] != b"WAVE" {
        return Err(WavValidationError::InvalidHeader("missing WAVE marker"));
    }

    let mut fmt = None;
    let mut data_size_bytes = None;

    loop {
        let mut chunk_header = [0_u8; 8];
        match file.read_exact(&mut chunk_header) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err.into()),
        }

        let chunk_id = &chunk_header[0..4];
        let chunk_size = u32::from_le_bytes([
            chunk_header[4],
            chunk_header[5],
            chunk_header[6],
            chunk_header[7],
        ]);

        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err(WavValidationError::InvalidHeader("fmt chunk is too small"));
                }
                let mut fmt_header = [0_u8; 16];
                file.read_exact(&mut fmt_header)?;
                fmt = Some((
                    u16::from_le_bytes([fmt_header[0], fmt_header[1]]),
                    u16::from_le_bytes([fmt_header[2], fmt_header[3]]),
                    u32::from_le_bytes([
                        fmt_header[4],
                        fmt_header[5],
                        fmt_header[6],
                        fmt_header[7],
                    ]),
                    u16::from_le_bytes([fmt_header[14], fmt_header[15]]),
                ));
                skip_remaining_chunk_bytes(&mut file, chunk_size - 16)?;
            }
            b"data" => {
                data_size_bytes = Some(chunk_size);
                skip_remaining_chunk_bytes(&mut file, chunk_size)?;
            }
            _ => skip_remaining_chunk_bytes(&mut file, chunk_size)?,
        }

        if fmt.is_some() && data_size_bytes.is_some() {
            break;
        }
    }

    let (audio_format, channels, sample_rate_hz, bits_per_sample) =
        fmt.ok_or(WavValidationError::MissingFmtChunk)?;
    let data_size_bytes = data_size_bytes.ok_or(WavValidationError::MissingDataChunk)?;

    Ok(WavMetadata {
        audio_format,
        sample_rate_hz,
        channels,
        bits_per_sample,
        data_size_bytes,
    })
}

/// Reads and validates WAV metadata against an expected audio spec.
///
/// # Errors
///
/// Returns an error for unreadable files, invalid WAV headers, unsupported
/// audio format, empty data chunks, or metadata mismatches.
pub fn validate_wav_file(
    path: impl AsRef<Path>,
    expected: AudioSpec,
) -> WavValidationResult<WavMetadata> {
    let metadata = read_wav_metadata(path)?;
    validate_wav_metadata(metadata, expected)?;
    Ok(metadata)
}

/// Validates parsed WAV metadata against an expected audio spec.
///
/// # Errors
///
/// Returns an error for unsupported audio format, empty data chunks, or
/// mismatched sample rate, channel count, or bits per sample.
pub fn validate_wav_metadata(
    metadata: WavMetadata,
    expected: AudioSpec,
) -> WavValidationResult<()> {
    if metadata.audio_format != 1 {
        return Err(WavValidationError::UnsupportedFormat(metadata.audio_format));
    }
    if metadata.sample_rate_hz != expected.sample_rate_hz {
        return Err(WavValidationError::MetadataMismatch {
            field: "sample_rate_hz",
            expected: u64::from(expected.sample_rate_hz),
            actual: u64::from(metadata.sample_rate_hz),
        });
    }
    if metadata.channels != expected.channels {
        return Err(WavValidationError::MetadataMismatch {
            field: "channels",
            expected: u64::from(expected.channels),
            actual: u64::from(metadata.channels),
        });
    }
    if metadata.bits_per_sample != expected.bits_per_sample {
        return Err(WavValidationError::MetadataMismatch {
            field: "bits_per_sample",
            expected: u64::from(expected.bits_per_sample),
            actual: u64::from(metadata.bits_per_sample),
        });
    }
    if metadata.data_size_bytes == 0 {
        return Err(WavValidationError::EmptyDataChunk);
    }
    Ok(())
}

fn skip_remaining_chunk_bytes(file: &mut File, byte_count: u32) -> io::Result<()> {
    let padded_byte_count = byte_count + (byte_count % 2);
    file.seek(SeekFrom::Current(i64::from(padded_byte_count)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_wav_file, WavValidationError};
    use crate::AudioSpec;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn validates_minimal_wav_metadata() {
        let path = temp_wav_path("validates_minimal_wav_metadata");
        write_minimal_wav(&path, 24_000, 1, 16, 4);

        let metadata = validate_wav_file(&path, AudioSpec::default()).unwrap();

        assert_eq!(metadata.sample_rate_hz, 24_000);
        assert_eq!(metadata.channels, 1);
        assert_eq!(metadata.bits_per_sample, 16);
        assert_eq!(metadata.data_size_bytes, 4);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_sample_rate_mismatch() {
        let path = temp_wav_path("rejects_sample_rate_mismatch");
        write_minimal_wav(&path, 48_000, 1, 16, 4);

        let err = validate_wav_file(&path, AudioSpec::default()).unwrap_err();

        match err {
            WavValidationError::MetadataMismatch { field, .. } => {
                assert_eq!(field, "sample_rate_hz");
            }
            other => panic!("unexpected error: {other}"),
        }

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_empty_data_chunk() {
        let path = temp_wav_path("rejects_empty_data_chunk");
        write_minimal_wav(&path, 24_000, 1, 16, 0);

        let err = validate_wav_file(&path, AudioSpec::default()).unwrap_err();

        assert!(matches!(err, WavValidationError::EmptyDataChunk));

        fs::remove_file(path).unwrap();
    }

    fn write_minimal_wav(
        path: &PathBuf,
        sample_rate_hz: u32,
        channels: u16,
        bits_per_sample: u16,
        data_size_bytes: u32,
    ) {
        let byte_rate = sample_rate_hz * u32::from(channels) * u32::from(bits_per_sample) / 8;
        let block_align = channels * bits_per_sample / 8;
        let riff_size = 36 + data_size_bytes;
        let mut bytes = Vec::new();

        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&riff_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate_hz.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size_bytes.to_le_bytes());
        bytes.extend(std::iter::repeat(0).take(data_size_bytes as usize));

        fs::write(path, bytes).unwrap();
    }

    fn temp_wav_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "qwen_tts_core_{name}_{}_{}.wav",
            std::process::id(),
            nonce
        ))
    }
}
