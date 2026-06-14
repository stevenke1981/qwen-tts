#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpec {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
}

impl Default for AudioSpec {
    fn default() -> Self {
        Self {
            sample_rate_hz: 24_000,
            channels: 1,
            bits_per_sample: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub spec: AudioSpec,
    pub samples_f32: Vec<f32>,
}

impl AudioBuffer {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            spec: AudioSpec::default(),
            samples_f32: Vec::new(),
        }
    }
}
