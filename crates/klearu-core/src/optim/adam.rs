use super::{Optimizer, OptimizerState};

/// Adam optimizer (Kingma & Ba, 2014).
pub struct Adam {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

impl Adam {
    /// Create a new Adam optimizer with custom hyperparameters.
    pub fn new(beta1: f32, beta2: f32, eps: f32) -> Self {
        Self { beta1, beta2, eps }
    }
}

impl Optimizer for Adam {
    fn update(
        &self,
        weights: &mut [f32],
        grad_weights: &[f32],
        bias: &mut f32,
        grad_bias: f32,
        lr: f64,
        step: u64,
        state: &mut OptimizerState,
    ) {
        let lr = lr as f32;
        let step = step.max(1); // Avoid division by zero if step == 0.

        let (m_w, v_w, m_b, v_b) = match state {
            OptimizerState::Adam {
                m_w,
                v_w,
                m_b,
                v_b,
            } => (m_w, v_w, m_b, v_b),
            _ => panic!("Adam optimizer received non-Adam state"),
        };

        let beta1_t = self.beta1.powi(step as i32);
        let beta2_t = self.beta2.powi(step as i32);
        let correction1 = 1.0 - beta1_t;
        let correction2 = 1.0 - beta2_t;

        // Update weights.
        for i in 0..weights.len() {
            let g = grad_weights[i];
            // Update first moment.
            m_w[i] = self.beta1 * m_w[i] + (1.0 - self.beta1) * g;
            // Update second moment.
            v_w[i] = self.beta2 * v_w[i] + (1.0 - self.beta2) * g * g;
            // Bias-corrected moments.
            let m_hat = m_w[i] / correction1;
            let v_hat = v_w[i] / correction2;
            // Update weight.
            weights[i] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }

        // Update bias.
        *m_b = self.beta1 * *m_b + (1.0 - self.beta1) * grad_bias;
        *v_b = self.beta2 * *v_b + (1.0 - self.beta2) * grad_bias * grad_bias;
        let m_hat_b = *m_b / correction1;
        let v_hat_b = *v_b / correction2;
        *bias -= lr * m_hat_b / (v_hat_b.sqrt() + self.eps);
    }

    fn create_state(&self, dim: usize) -> OptimizerState {
        OptimizerState::Adam {
            m_w: vec![0.0; dim],
            v_w: vec![0.0; dim],
            m_b: 0.0,
            v_b: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_default() {
        let adam = Adam::default();
        assert!((adam.beta1 - 0.9).abs() < 1e-6);
        assert!((adam.beta2 - 0.999).abs() < 1e-6);
        assert!((adam.eps - 1e-8).abs() < 1e-12);
    }

    #[test]
    fn test_adam_custom() {
        let adam = Adam::new(0.8, 0.99, 1e-7);
        assert!((adam.beta1 - 0.8).abs() < 1e-6);
        assert!((adam.beta2 - 0.99).abs() < 1e-6);
        assert!((adam.eps - 1e-7).abs() < 1e-12);
    }

    #[test]
    fn test_adam_create_state() {
        let adam = Adam::default();
        let state = adam.create_state(5);
        match state {
            OptimizerState::Adam {
                m_w,
                v_w,
                m_b,
                v_b,
            } => {
                assert_eq!(m_w.len(), 5);
                assert_eq!(v_w.len(), 5);
                assert!(m_w.iter().all(|&v| v == 0.0));
                assert!(v_w.iter().all(|&v| v == 0.0));
                assert_eq!(m_b, 0.0);
                assert_eq!(v_b, 0.0);
            }
            _ => panic!("Expected Adam state"),
        }
    }

    #[test]
    fn test_adam_single_step() {
        let adam = Adam::default();
        let mut weights = vec![1.0];
        let grads = vec![0.1];
        let mut bias = 0.5;
        let grad_bias = 0.05;
        let lr = 0.001;
        let mut state = adam.create_state(1);

        adam.update(&mut weights, &grads, &mut bias, grad_bias, lr, 1, &mut state);

        // After one step, weights should have moved.
        assert!(weights[0] < 1.0, "weight should decrease with positive gradient");
        assert!(bias < 0.5, "bias should decrease with positive gradient");
    }

    #[test]
    fn test_adam_multiple_steps_converge() {
        let adam = Adam::default();
        let mut weights = vec![5.0];
        let mut bias = 2.0;
        let lr = 0.01;
        let mut state = adam.create_state(1);

        // Apply constant gradient for many steps, weights should move towards 0.
        for step in 1..=100 {
            let grads = vec![1.0]; // constant positive gradient
            adam.update(
                &mut weights,
                &grads,
                &mut bias,
                1.0,
                lr,
                step,
                &mut state,
            );
        }

        assert!(
            weights[0] < 5.0,
            "weight should have decreased from initial value"
        );
        assert!(bias < 2.0, "bias should have decreased from initial value");
    }

    #[test]
    fn test_adam_zero_gradient() {
        let adam = Adam::default();
        let mut weights = vec![1.0, 2.0];
        let grads = vec![0.0, 0.0];
        let mut bias = 0.5;
        let lr = 0.001;
        let mut state = adam.create_state(2);

        adam.update(&mut weights, &grads, &mut bias, 0.0, lr, 1, &mut state);

        // With zero gradient, moments stay zero, so no update occurs.
        assert!((weights[0] - 1.0).abs() < 1e-6);
        assert!((weights[1] - 2.0).abs() < 1e-6);
        assert!((bias - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_adam_negative_gradient() {
        let adam = Adam::default();
        let mut weights = vec![0.0];
        let grads = vec![-1.0];
        let mut bias = 0.0;
        let lr = 0.001;
        let mut state = adam.create_state(1);

        adam.update(&mut weights, &grads, &mut bias, -1.0, lr, 1, &mut state);

        // Negative gradient should increase weights.
        assert!(weights[0] > 0.0, "weight should increase with negative gradient");
        assert!(bias > 0.0, "bias should increase with negative gradient");
    }

    #[test]
    fn test_adam_bias_correction() {
        // Verify bias correction by checking that step 1 and step 100 produce
        // different effective learning rates for the same gradient.
        let adam = Adam::default();

        let mut w1 = vec![0.0];
        let mut b1 = 0.0;
        let mut s1 = adam.create_state(1);
        adam.update(&mut w1, &[1.0], &mut b1, 1.0, 0.001, 1, &mut s1);

        let mut w100 = vec![0.0];
        let mut b100 = 0.0;
        let mut s100 = adam.create_state(1);
        adam.update(&mut w100, &[1.0], &mut b100, 1.0, 0.001, 100, &mut s100);

        // Step 1 has larger bias correction than step 100, so the effective
        // update magnitude differs.
        assert!(
            (w1[0] - w100[0]).abs() > 1e-10,
            "bias correction should produce different updates at different steps"
        );
    }

    #[test]
    fn test_adam_momentum_accumulation() {
        let adam = Adam::default();
        let mut weights = vec![0.0];
        let mut bias = 0.0;
        let lr = 0.01;
        let mut state = adam.create_state(1);

        // Apply same gradient multiple times; due to momentum, the update
        // should accelerate.
        let mut prev_weights = 0.0_f32;
        let mut deltas = Vec::new();
        for step in 1..=5 {
            adam.update(&mut weights, &[1.0], &mut bias, 1.0, lr, step, &mut state);
            deltas.push(prev_weights - weights[0]); // positive means weight decreased
            prev_weights = weights[0];
        }

        // All deltas should be positive (weight decreasing with positive gradient).
        for (i, &d) in deltas.iter().enumerate() {
            assert!(d > 0.0, "step {}: delta should be positive, got {}", i + 1, d);
        }
    }
}
