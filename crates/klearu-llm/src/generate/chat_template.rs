use std::path::Path;

use serde::Deserialize;

use crate::error::{LlmError, Result};

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// Known chat template formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplate {
    /// TinyLlama / Zephyr style:
    /// `<|system|>\n{content}</s>\n<|user|>\n{content}</s>\n<|assistant|>\n`
    Zephyr,
    /// ChatML (used by many models):
    /// `<|im_start|>system\n{content}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n`
    ChatML,
    /// Llama-2 chat format:
    /// `[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] `
    Llama2,
    /// Llama-3 / Llama-3.1 chat format:
    /// `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>...`
    Llama3,
    /// Mistral Instruct v0.1 format:
    /// `[INST] {user} [/INST]`
    MistralInstruct,
    /// Raw: no template, just concatenate content.
    Raw,
}

/// Partial tokenizer_config.json for auto-detection.
#[derive(Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    chat_template: Option<String>,
}

impl ChatTemplate {
    /// Try to auto-detect the chat template from a model directory.
    ///
    /// Reads `tokenizer_config.json` and matches known patterns.
    pub fn detect(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("tokenizer_config.json");
        if !config_path.exists() {
            return Ok(ChatTemplate::Raw);
        }

        let json = std::fs::read_to_string(&config_path)?;
        let config: TokenizerConfig =
            serde_json::from_str(&json).map_err(LlmError::Json)?;

        match config.chat_template.as_deref() {
            Some(tmpl) => Ok(Self::from_template_string(tmpl)),
            None => Ok(ChatTemplate::Raw),
        }
    }

    /// Detect template format from a Jinja-style template string.
    fn from_template_string(tmpl: &str) -> Self {
        if tmpl.contains("<|im_start|>") {
            ChatTemplate::ChatML
        } else if tmpl.contains("<|start_header_id|>") {
            ChatTemplate::Llama3
        } else if tmpl.contains("<|system|>") || tmpl.contains("<|user|>") {
            ChatTemplate::Zephyr
        } else if tmpl.contains("<<SYS>>") {
            ChatTemplate::Llama2
        } else if tmpl.contains("[INST]") {
            ChatTemplate::MistralInstruct
        } else {
            ChatTemplate::Raw
        }
    }

    /// Format a conversation into a prompt string.
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::Zephyr => format_zephyr(messages),
            ChatTemplate::ChatML => format_chatml(messages),
            ChatTemplate::Llama2 => format_llama2(messages),
            ChatTemplate::Llama3 => format_llama3(messages),
            ChatTemplate::MistralInstruct => format_mistral_instruct(messages),
            ChatTemplate::Raw => format_raw(messages),
        }
    }
}

fn format_zephyr(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!("<|{role}|>\n{}</s>\n", msg.content));
    }
    out.push_str("<|assistant|>\n");
    out
}

fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", msg.content));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn format_llama2(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    let system_msg = messages
        .iter()
        .find(|m| m.role == Role::System)
        .map(|m| m.content.as_str());

    let non_system: Vec<_> = messages.iter().filter(|m| m.role != Role::System).collect();

    for (i, msg) in non_system.iter().enumerate() {
        match msg.role {
            Role::User => {
                out.push_str("[INST] ");
                if i == 0 {
                    if let Some(sys) = system_msg {
                        out.push_str(&format!("<<SYS>>\n{sys}\n<</SYS>>\n\n"));
                    }
                }
                out.push_str(&msg.content);
                out.push_str(" [/INST] ");
            }
            Role::Assistant => {
                out.push_str(&msg.content);
                out.push_str(" </s>");
            }
            Role::System => {}
        }
    }
    out
}

fn format_llama3(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!(
            "<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.content
        ));
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn format_mistral_instruct(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    let mut pending_user = String::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                // Mistral doesn't have system role; prepend to first user message
                pending_user.push_str(&msg.content);
                pending_user.push_str("\n\n");
            }
            Role::User => {
                pending_user.push_str(&msg.content);
                out.push_str(&format!("[INST] {pending_user} [/INST]"));
                pending_user.clear();
            }
            Role::Assistant => {
                out.push_str(&msg.content);
                out.push_str("</s>");
            }
        }
    }

    // If there's an unanswered user message at the end
    if !pending_user.is_empty() {
        out.push_str(&format!("[INST] {pending_user} [/INST]"));
    }

    out
}

fn format_raw(messages: &[ChatMessage]) -> String {
    messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zephyr_format() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
        ];
        let prompt = ChatTemplate::Zephyr.apply(&messages);
        assert!(prompt.contains("<|user|>\nHello</s>"));
        assert!(prompt.contains("<|assistant|>\nHi there!</s>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_chatml_format() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hi"),
        ];
        let prompt = ChatTemplate::ChatML.apply(&messages);
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_llama2_format() {
        let messages = vec![
            ChatMessage::system("Be concise."),
            ChatMessage::user("Hello"),
        ];
        let prompt = ChatTemplate::Llama2.apply(&messages);
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("Be concise."));
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_llama3_format() {
        let messages = vec![ChatMessage::user("What is 2+2?")];
        let prompt = ChatTemplate::Llama3.apply(&messages);
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.contains("What is 2+2?"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_mistral_instruct_format() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi!"),
            ChatMessage::user("Bye"),
        ];
        let prompt = ChatTemplate::MistralInstruct.apply(&messages);
        assert!(prompt.contains("[INST] Hello [/INST]"));
        assert!(prompt.contains("Hi!</s>"));
        assert!(prompt.contains("[INST] Bye [/INST]"));
    }

    #[test]
    fn test_detect_from_template_string() {
        assert_eq!(
            ChatTemplate::from_template_string("{% for m in messages %}<|im_start|>..."),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::from_template_string("{% if messages %}<|start_header_id|>..."),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::from_template_string("{% for m %}<|user|>..."),
            ChatTemplate::Zephyr
        );
        assert_eq!(
            ChatTemplate::from_template_string("{% for m %}<<SYS>>..."),
            ChatTemplate::Llama2
        );
        assert_eq!(
            ChatTemplate::from_template_string("{% for m %}[INST]..."),
            ChatTemplate::MistralInstruct
        );
        assert_eq!(
            ChatTemplate::from_template_string("just plain text"),
            ChatTemplate::Raw
        );
    }
}
