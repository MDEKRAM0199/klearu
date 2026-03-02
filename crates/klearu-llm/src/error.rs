use std::io;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensors(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("Weight loading error: {0}")]
    WeightLoad(String),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Missing weight: {0}")]
    MissingWeight(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),
}

pub type Result<T> = std::result::Result<T, LlmError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LlmError::InvalidConfig("bad param".into());
        assert!(err.to_string().contains("bad param"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: LlmError = io_err.into();
        assert!(err.to_string().contains("file not found"));
    }
}
