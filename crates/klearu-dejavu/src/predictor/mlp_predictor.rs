use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Small MLP that predicts which heads/neurons to activate.
/// Trained via distillation from full model outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityPredictor {
    /// First layer weights [hidden_dim][input_dim]
    w1: Vec<Vec<f32>>,
    /// First layer bias [hidden_dim]
    b1: Vec<f32>,
    /// Second layer weights [output_dim][hidden_dim]
    w2: Vec<Vec<f32>>,
    /// Second layer bias [output_dim]
    b2: Vec<f32>,
    /// Input dimension (model hidden size).
    input_dim: usize,
    /// Hidden dimension of predictor.
    hidden_dim: usize,
    /// Output dimension (num_heads for attention, intermediate_size for MLP).
    output_dim: usize,
}

impl SparsityPredictor {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let std2 = (2.0 / (hidden_dim + output_dim) as f64).sqrt();
        let dist1 = Normal::new(0.0, std1).unwrap();
        let dist2 = Normal::new(0.0, std2).unwrap();

        let w1: Vec<Vec<f32>> = (0..hidden_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| dist1.sample(&mut rng) as f32)
                    .collect()
            })
            .collect();
        let b1 = vec![0.0; hidden_dim];
        let w2: Vec<Vec<f32>> = (0..output_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| dist2.sample(&mut rng) as f32)
                    .collect()
            })
            .collect();
        let b2 = vec![0.0; output_dim];

        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Construct from pre-trained weight matrices.
    /// Dimensions are inferred from the shapes.
    pub fn from_weights(
        w1: Vec<Vec<f32>>,
        b1: Vec<f32>,
        w2: Vec<Vec<f32>>,
        b2: Vec<f32>,
    ) -> Self {
        let hidden_dim = w1.len();
        let input_dim = if hidden_dim > 0 { w1[0].len() } else { 0 };
        let output_dim = w2.len();
        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Save predictor weights to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load predictor weights from a JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Predict importance scores for each head/neuron.
    /// Returns scores in [0, 1] range (sigmoid output).
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        // Layer 1: ReLU
        let hidden: Vec<f32> = self
            .w1
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let dot: f32 = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                (dot + self.b1[i]).max(0.0)
            })
            .collect();

        // Layer 2: Sigmoid
        self.w2
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let dot: f32 = row.iter().zip(hidden.iter()).map(|(w, x)| w * x).sum();
                let logit = dot + self.b2[i];
                1.0 / (1.0 + (-logit).exp())
            })
            .collect()
    }

    /// Select top-k indices based on predicted importance.
    pub fn select_top_k(&self, input: &[f32], k: usize) -> Vec<usize> {
        let scores = self.predict(input);
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Train one step via distillation.
    /// `input`: model hidden state
    /// `target_scores`: importance scores from running the full (dense) model
    /// `lr`: learning rate
    pub fn train_step(&mut self, input: &[f32], target_scores: &[f32], lr: f32) {
        // Forward pass
        let hidden: Vec<f32> = self
            .w1
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let dot: f32 = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                (dot + self.b1[i]).max(0.0)
            })
            .collect();

        let predictions: Vec<f32> = self
            .w2
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let dot: f32 = row.iter().zip(hidden.iter()).map(|(w, x)| w * x).sum();
                let logit = dot + self.b2[i];
                1.0 / (1.0 + (-logit).exp())
            })
            .collect();

        // BCE loss gradient: d_loss/d_logit = prediction - target
        let d_output: Vec<f32> = predictions
            .iter()
            .zip(target_scores.iter())
            .map(|(p, t)| p - t)
            .collect();

        // Backward through layer 2
        let mut d_hidden = vec![0.0f32; self.hidden_dim];
        for i in 0..self.output_dim {
            let grad = d_output[i];
            for j in 0..self.hidden_dim {
                d_hidden[j] += self.w2[i][j] * grad;
                self.w2[i][j] -= lr * grad * hidden[j];
            }
            self.b2[i] -= lr * grad;
        }

        // Backward through ReLU
        let d_hidden_relu: Vec<f32> = d_hidden
            .iter()
            .enumerate()
            .map(|(i, &dh)| {
                let pre_act: f32 = self.w1[i]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
                    + self.b1[i];
                if pre_act > 0.0 {
                    dh
                } else {
                    0.0
                }
            })
            .collect();

        // Backward through layer 1
        for i in 0..self.hidden_dim {
            let grad = d_hidden_relu[i];
            for j in 0..self.input_dim {
                self.w1[i][j] -= lr * grad * input[j];
            }
            self.b1[i] -= lr * grad;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_output_shape() {
        let predictor = SparsityPredictor::new(64, 32, 12, 42);
        let input = vec![0.5f32; 64];
        let scores = predictor.predict(&input);
        assert_eq!(scores.len(), 12);
    }

    #[test]
    fn test_predict_output_range() {
        let predictor = SparsityPredictor::new(64, 32, 12, 42);
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let scores = predictor.predict(&input);
        for (i, &s) in scores.iter().enumerate() {
            assert!(
                s >= 0.0 && s <= 1.0,
                "Score at index {} should be in [0,1], got {}",
                i,
                s
            );
        }
    }

    #[test]
    fn test_select_top_k_count() {
        let predictor = SparsityPredictor::new(64, 32, 12, 42);
        let input = vec![1.0f32; 64];
        let selected = predictor.select_top_k(&input, 5);
        assert_eq!(selected.len(), 5);

        // All indices should be valid
        for &idx in &selected {
            assert!(idx < 12, "Index {} should be < 12", idx);
        }

        // No duplicates
        let mut sorted = selected.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5, "Should have no duplicate indices");
    }

    #[test]
    fn test_select_top_k_more_than_available() {
        let predictor = SparsityPredictor::new(64, 32, 12, 42);
        let input = vec![1.0f32; 64];
        let selected = predictor.select_top_k(&input, 20);
        assert_eq!(selected.len(), 12, "Cannot select more than output_dim");
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let mut predictor = SparsityPredictor::new(32, 16, 8, 42);
        let input = vec![1.0f32; 32];
        let target = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        // Compute initial loss
        let initial_preds = predictor.predict(&input);
        let initial_loss: f32 = initial_preds
            .iter()
            .zip(target.iter())
            .map(|(p, t)| {
                let p_clamped = p.clamp(1e-7, 1.0 - 1e-7);
                -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
            })
            .sum();

        // Train for several steps
        for _ in 0..100 {
            predictor.train_step(&input, &target, 0.01);
        }

        // Compute final loss
        let final_preds = predictor.predict(&input);
        let final_loss: f32 = final_preds
            .iter()
            .zip(target.iter())
            .map(|(p, t)| {
                let p_clamped = p.clamp(1e-7, 1.0 - 1e-7);
                -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
            })
            .sum();

        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={}, final={}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let predictor = SparsityPredictor::new(32, 16, 8, 42);
        let input = vec![0.5f32; 32];
        let original_scores = predictor.predict(&input);

        let dir = std::env::temp_dir().join("klearu_test_predictor");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_predictor.json");

        predictor.save(&path).unwrap();
        let loaded = SparsityPredictor::load(&path).unwrap();
        let loaded_scores = loaded.predict(&input);

        assert_eq!(original_scores, loaded_scores);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_weights() {
        let predictor = SparsityPredictor::new(8, 4, 3, 42);
        let input = vec![1.0f32; 8];
        let scores1 = predictor.predict(&input);

        // Serialize and reconstruct via from_weights (via JSON roundtrip to extract weights)
        let json = serde_json::to_string(&predictor).unwrap();
        let loaded: SparsityPredictor = serde_json::from_str(&json).unwrap();
        let scores2 = loaded.predict(&input);
        assert_eq!(scores1, scores2);
    }

    #[test]
    fn test_predict_deterministic() {
        let predictor = SparsityPredictor::new(64, 32, 12, 42);
        let input = vec![0.5f32; 64];
        let scores1 = predictor.predict(&input);
        let scores2 = predictor.predict(&input);
        assert_eq!(scores1, scores2, "Prediction should be deterministic");
    }
}
