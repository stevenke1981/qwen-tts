#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    TokenizeText,
    TalkerForward,
    PredictAcousticCodes,
    CodecDecode,
    WriteWav,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub name: String,
    pub kind: NodeKind,
}

#[derive(Debug, Default, Clone)]
pub struct TtsGraph {
    pub nodes: Vec<GraphNode>,
}

impl TtsGraph {
    #[must_use]
    pub fn qwen_tts_default() -> Self {
        Self {
            nodes: vec![
                GraphNode {
                    name: "tokenize_text".into(),
                    kind: NodeKind::TokenizeText,
                },
                GraphNode {
                    name: "talker_forward".into(),
                    kind: NodeKind::TalkerForward,
                },
                GraphNode {
                    name: "predict_acoustic_codes".into(),
                    kind: NodeKind::PredictAcousticCodes,
                },
                GraphNode {
                    name: "codec_decode".into(),
                    kind: NodeKind::CodecDecode,
                },
                GraphNode {
                    name: "write_wav".into(),
                    kind: NodeKind::WriteWav,
                },
            ],
        }
    }
}
