use std::io::BufRead;

use super::{Dataset, Example};
use crate::error::{KlearuError, Result};
use crate::tensor::SparseVector;

/// A dataset loaded from (or compatible with) the LibSVM file format.
///
/// The expected line format is:
/// ```text
/// label1,label2,... feature_idx:value feature_idx:value ...
/// ```
///
/// Single-label lines are also supported:
/// ```text
/// label feature_idx:value feature_idx:value ...
/// ```
///
/// Feature indices may be 0-based or 1-based. The loader auto-detects 1-based
/// indexing (when the minimum observed index is 1 and no index 0 is present)
/// and adjusts accordingly.
#[derive(Debug)]
pub struct LibSvmDataset {
    examples: Vec<Example>,
    feature_dim: usize,
    num_labels: usize,
}

impl LibSvmDataset {
    /// Load a LibSVM-format file from disk.
    pub fn load(path: &str, feature_dim: usize, num_labels: usize) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        Self::load_from_reader(reader, feature_dim, num_labels)
    }

    /// Load from any `BufRead` source (useful for testing with in-memory data).
    pub fn load_from_reader<R: BufRead>(
        reader: R,
        feature_dim: usize,
        num_labels: usize,
    ) -> Result<Self> {
        let mut min_feature_idx: Option<u32> = None;
        let mut has_zero_idx = false;

        // First pass: parse all lines.
        let mut raw_examples: Vec<(Vec<u32>, Vec<(u32, f32)>)> = Vec::new();

        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim().to_string();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let (labels, features) = Self::parse_line(&line, line_no)?;

            for &(idx, _) in &features {
                if idx == 0 {
                    has_zero_idx = true;
                }
                min_feature_idx = Some(match min_feature_idx {
                    Some(m) => m.min(idx),
                    None => idx,
                });
            }

            raw_examples.push((labels, features));
        }

        // Detect 1-based indexing: minimum index is 1 and no index 0 was seen.
        let offset: u32 = if !has_zero_idx && min_feature_idx == Some(1) {
            1
        } else {
            0
        };

        // Second pass: build Examples with adjusted indices.
        let mut examples = Vec::with_capacity(raw_examples.len());
        for (labels, features) in raw_examples {
            let pairs: Vec<(u32, f32)> = features
                .into_iter()
                .map(|(idx, val)| {
                    let adjusted = idx - offset;
                    if adjusted as usize >= feature_dim {
                        // Silently clamp -- the caller specified the dim.
                        // This matches common LibSVM loaders that ignore
                        // out-of-range features.
                        ((feature_dim - 1) as u32, val)
                    } else {
                        (adjusted, val)
                    }
                })
                .collect();

            let sv = SparseVector::from_pairs(feature_dim, pairs);
            examples.push(Example {
                features: sv,
                labels,
            });
        }

        Ok(Self {
            examples,
            feature_dim,
            num_labels,
        })
    }

    /// Build a dataset directly from pre-constructed examples.
    pub fn from_examples(
        examples: Vec<Example>,
        feature_dim: usize,
        num_labels: usize,
    ) -> Self {
        Self {
            examples,
            feature_dim,
            num_labels,
        }
    }

    /// Parse a single LibSVM line.
    ///
    /// Returns `(labels, feature_pairs)`.
    fn parse_line(line: &str, line_no: usize) -> Result<(Vec<u32>, Vec<(u32, f32)>)> {
        let mut tokens = line.split_whitespace();

        let label_token = tokens.next().ok_or_else(|| {
            KlearuError::Parse(format!("line {}: empty line", line_no + 1))
        })?;

        // Parse label(s): may be comma-separated.
        let labels: Vec<u32> = label_token
            .split(',')
            .map(|s| {
                s.parse::<u32>().map_err(|_| {
                    KlearuError::Parse(format!(
                        "line {}: invalid label '{}'",
                        line_no + 1,
                        s
                    ))
                })
            })
            .collect::<Result<Vec<u32>>>()?;

        // Parse feature:value pairs.
        let mut features = Vec::new();
        for token in tokens {
            // Skip comments at end of line.
            if token.starts_with('#') {
                break;
            }

            let (idx_str, val_str) = token.split_once(':').ok_or_else(|| {
                KlearuError::Parse(format!(
                    "line {}: expected 'index:value', got '{}'",
                    line_no + 1,
                    token
                ))
            })?;

            let idx: u32 = idx_str.parse().map_err(|_| {
                KlearuError::Parse(format!(
                    "line {}: invalid feature index '{}'",
                    line_no + 1,
                    idx_str
                ))
            })?;

            let val: f32 = val_str.parse().map_err(|_| {
                KlearuError::Parse(format!(
                    "line {}: invalid feature value '{}'",
                    line_no + 1,
                    val_str
                ))
            })?;

            features.push((idx, val));
        }

        Ok((labels, features))
    }
}

impl Dataset for LibSvmDataset {
    fn len(&self) -> usize {
        self.examples.len()
    }

    fn get(&self, index: usize) -> &Example {
        &self.examples[index]
    }

    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn load_from_str(data: &str, feature_dim: usize, num_labels: usize) -> Result<LibSvmDataset> {
        let cursor = Cursor::new(data.as_bytes().to_vec());
        LibSvmDataset::load_from_reader(cursor, feature_dim, num_labels)
    }

    #[test]
    fn test_single_label_0based() {
        let data = "0 0:1.0 2:3.5\n1 1:2.0\n";
        let ds = load_from_str(data, 4, 2).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.feature_dim(), 4);
        assert_eq!(ds.num_labels(), 2);

        let ex0 = ds.get(0);
        assert_eq!(ex0.labels, vec![0]);
        assert_eq!(ex0.features.indices, vec![0, 2]);
        assert_eq!(ex0.features.values, vec![1.0, 3.5]);

        let ex1 = ds.get(1);
        assert_eq!(ex1.labels, vec![1]);
        assert_eq!(ex1.features.indices, vec![1]);
        assert_eq!(ex1.features.values, vec![2.0]);
    }

    #[test]
    fn test_single_label_1based_autodetect() {
        // Features are 1-based (1, 2, 3) and no 0 index appears.
        let data = "0 1:1.0 3:2.0\n1 2:5.0\n";
        let ds = load_from_str(data, 4, 2).unwrap();

        let ex0 = ds.get(0);
        // After adjusting: 1->0, 3->2
        assert_eq!(ex0.features.indices, vec![0, 2]);
        assert_eq!(ex0.features.values, vec![1.0, 2.0]);

        let ex1 = ds.get(1);
        // 2->1
        assert_eq!(ex1.features.indices, vec![1]);
    }

    #[test]
    fn test_multi_label() {
        let data = "0,3,7 0:1.0 1:2.0\n";
        let ds = load_from_str(data, 5, 10).unwrap();
        let ex = ds.get(0);
        assert_eq!(ex.labels, vec![0, 3, 7]);
    }

    #[test]
    fn test_empty_features() {
        let data = "5\n";
        let ds = load_from_str(data, 10, 6).unwrap();
        let ex = ds.get(0);
        assert_eq!(ex.labels, vec![5]);
        assert!(ex.features.is_empty());
    }

    #[test]
    fn test_blank_lines_and_comments() {
        let data = "# This is a comment\n\n0 0:1.0\n   \n1 1:2.0\n";
        let ds = load_from_str(data, 5, 2).unwrap();
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_inline_comment() {
        let data = "0 0:1.0 1:2.0 # this is a comment\n";
        let ds = load_from_str(data, 5, 2).unwrap();
        let ex = ds.get(0);
        assert_eq!(ex.features.nnz(), 2);
    }

    #[test]
    fn test_empty_file() {
        let data = "";
        let ds = load_from_str(data, 10, 5).unwrap();
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
    }

    #[test]
    fn test_invalid_label() {
        let data = "abc 0:1.0\n";
        let result = load_from_str(data, 10, 5);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("invalid label"));
    }

    #[test]
    fn test_invalid_feature_format() {
        let data = "0 badformat\n";
        let result = load_from_str(data, 10, 5);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("expected 'index:value'"));
    }

    #[test]
    fn test_invalid_feature_index() {
        let data = "0 abc:1.0\n";
        let result = load_from_str(data, 10, 5);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("invalid feature index"));
    }

    #[test]
    fn test_invalid_feature_value() {
        let data = "0 0:not_a_number\n";
        let result = load_from_str(data, 10, 5);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("invalid feature value"));
    }

    #[test]
    fn test_from_examples() {
        let examples = vec![
            Example {
                features: SparseVector::from_pairs(10, vec![(0, 1.0)]),
                labels: vec![0],
            },
            Example {
                features: SparseVector::from_pairs(10, vec![(5, 2.0)]),
                labels: vec![1],
            },
        ];
        let ds = LibSvmDataset::from_examples(examples, 10, 2);
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.feature_dim(), 10);
        assert_eq!(ds.num_labels(), 2);
        assert_eq!(ds.get(0).labels, vec![0]);
        assert_eq!(ds.get(1).labels, vec![1]);
    }

    #[test]
    fn test_dataset_trait_is_empty() {
        let ds = LibSvmDataset::from_examples(vec![], 10, 5);
        assert!(ds.is_empty());
    }

    #[test]
    fn test_sparse_vector_dim_matches() {
        let data = "0 0:1.0 3:2.0\n";
        let ds = load_from_str(data, 100, 1).unwrap();
        assert_eq!(ds.get(0).features.dim, 100);
    }

    #[test]
    fn test_mixed_0based_keeps_0based() {
        // If index 0 is present, we stay 0-based even if min is 0.
        let data = "0 0:1.0 5:2.0\n";
        let ds = load_from_str(data, 10, 1).unwrap();
        let ex = ds.get(0);
        assert_eq!(ex.features.indices, vec![0, 5]);
    }

    #[test]
    fn test_multiple_features_sorted() {
        let data = "0 3:1.0 1:2.0 0:3.0\n";
        let ds = load_from_str(data, 10, 1).unwrap();
        let ex = ds.get(0);
        // SparseVector::from_pairs sorts by index.
        assert_eq!(ex.features.indices, vec![0, 1, 3]);
        assert_eq!(ex.features.values, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_large_feature_index_clamped() {
        // Feature index 999 with feature_dim=10 should be clamped to 9.
        let data = "0 0:1.0 999:2.0\n";
        let ds = load_from_str(data, 10, 1).unwrap();
        let ex = ds.get(0);
        // 999 -> clamped to 9
        assert!(ex.features.indices.iter().all(|&i| (i as usize) < 10));
    }

    #[test]
    fn test_negative_feature_values() {
        let data = "0 0:-1.5 2:3.0\n";
        let ds = load_from_str(data, 5, 1).unwrap();
        let ex = ds.get(0);
        assert_eq!(ex.features.values, vec![-1.5, 3.0]);
    }

    #[test]
    fn test_many_labels() {
        let data = "0,1,2,3,4,5,6,7,8,9 0:1.0\n";
        let ds = load_from_str(data, 5, 10).unwrap();
        let ex = ds.get(0);
        assert_eq!(ex.labels, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
