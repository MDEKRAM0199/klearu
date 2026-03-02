use std::path::Path;

use crate::error::{LlmError, Result};

/// Thin wrapper around the HuggingFace `tokenizers` crate.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| LlmError::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| LlmError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| LlmError::Tokenizer(e.to_string()))
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
