use std::collections::HashMap;
use std::path::Path;

use crate::error::{LlmError, Result};

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data_offset: usize,
    pub data_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
}

/// A memory-mapped safetensors file.
pub struct SafeTensorsFile {
    _mmap: memmap2::Mmap,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
    raw: *const u8,
    raw_len: usize,
}

// Safety: Mmap is Send+Sync when the underlying file is not modified.
unsafe impl Send for SafeTensorsFile {}
unsafe impl Sync for SafeTensorsFile {}

impl SafeTensorsFile {
    /// Open and memory-map a safetensors file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(LlmError::SafeTensors("File too small".into()));
        }

        // First 8 bytes: little-endian u64 = header length
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if 8 + header_len > mmap.len() {
            return Err(LlmError::SafeTensors("Invalid header length".into()));
        }

        let header_bytes = &mmap[8..8 + header_len];
        let header: serde_json::Value =
            serde_json::from_slice(header_bytes).map_err(|e| LlmError::SafeTensors(e.to_string()))?;

        let data_start = 8 + header_len;
        let mut tensors = HashMap::new();

        let obj = header
            .as_object()
            .ok_or_else(|| LlmError::SafeTensors("Header is not an object".into()))?;

        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype_str = info["dtype"]
                .as_str()
                .ok_or_else(|| LlmError::SafeTensors(format!("Missing dtype for {name}")))?;

            let dtype = match dtype_str {
                "F32" => Dtype::F32,
                "F16" => Dtype::F16,
                "BF16" => Dtype::BF16,
                other => return Err(LlmError::UnsupportedDtype(other.to_string())),
            };

            let shape: Vec<usize> = info["shape"]
                .as_array()
                .ok_or_else(|| LlmError::SafeTensors(format!("Missing shape for {name}")))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();

            let offsets = info["data_offsets"]
                .as_array()
                .ok_or_else(|| LlmError::SafeTensors(format!("Missing data_offsets for {name}")))?;

            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name: name.clone(),
                    dtype,
                    shape,
                    data_offset: start,
                    data_len: end - start,
                },
            );
        }

        let raw = mmap.as_ptr();
        let raw_len = mmap.len();

        Ok(Self {
            _mmap: mmap,
            data_start,
            tensors,
            raw,
            raw_len,
        })
    }

    /// Get metadata for all tensors.
    pub fn tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }

    /// Get raw bytes for a tensor.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| LlmError::MissingWeight(name.to_string()))?;
        let start = self.data_start + info.data_offset;
        let end = start + info.data_len;
        if end > self.raw_len {
            return Err(LlmError::SafeTensors(format!(
                "Tensor {name} data out of bounds"
            )));
        }
        Ok(unsafe { std::slice::from_raw_parts(self.raw.add(start), info.data_len) })
    }

    /// Load tensor data as f32, converting from BF16/F16 if needed.
    pub fn tensor_to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| LlmError::MissingWeight(name.to_string()))?;
        let data = self.tensor_data(name)?;

        match info.dtype {
            Dtype::F32 => {
                let floats: &[f32] = bytemuck::cast_slice(data);
                Ok(floats.to_vec())
            }
            Dtype::BF16 => {
                let bf16s: &[half::bf16] = bytemuck::cast_slice(data);
                Ok(bf16s.iter().map(|v| v.to_f32()).collect())
            }
            Dtype::F16 => {
                let f16s: &[half::f16] = bytemuck::cast_slice(data);
                Ok(f16s.iter().map(|v| v.to_f32()).collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_parsing() {
        // Just test that enum variants exist and can be compared
        assert_eq!(Dtype::F32, Dtype::F32);
        assert_ne!(Dtype::F32, Dtype::BF16);
        assert_ne!(Dtype::F16, Dtype::BF16);
    }

    #[test]
    fn test_bf16_to_f32_conversion() {
        // Test BF16 round-trip via half crate
        let original = 3.25f32;
        let bf = half::bf16::from_f32(original);
        let back = bf.to_f32();
        // BF16 has lower precision
        assert!((back - original).abs() < 0.02);
    }

    #[test]
    fn test_f16_to_f32_conversion() {
        let original = 1.5f32;
        let f = half::f16::from_f32(original);
        let back = f.to_f32();
        assert!((back - original).abs() < 1e-3);
    }
}
