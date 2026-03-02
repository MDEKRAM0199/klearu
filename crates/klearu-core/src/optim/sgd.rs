use super::{Optimizer, OptimizerState};

/// Stochastic Gradient Descent with optional momentum.
pub struct Sgd {
    pub momentum: f32,
}

impl Sgd {
    /// Create a new SGD optimizer.
    ///
    /// A momentum of 0.0 gives vanilla SGD.
    pub fn new(momentum: f32) -> Self {
        Self { momentum }
    }
}

impl Optimizer for Sgd {
    fn update(
        &self,
        weights: &mut [f32],
        grad_weights: &[f32],
        bias: &mut f32,
        grad_bias: f32,
        lr: f64,
        _step: u64,
        state: &mut OptimizerState,
    ) {
        let lr = lr as f32;

        if self.momentum > 0.0 {
            // Extract momentum buffers from state.
            let (momentum_w, momentum_b) = match state {
                OptimizerState::Sgd {
                    momentum_w,
                    momentum_b,
                } => (momentum_w, momentum_b),
                _ => panic!("SGD optimizer received non-SGD state"),
            };

            // Update weights with momentum.
            for (i, (w, gw)) in weights.iter_mut().zip(grad_weights.iter()).enumerate() {
                let v = self.momentum * momentum_w[i] - lr * gw;
                momentum_w[i] = v;
                *w += v;
            }

            // Update bias with momentum.
            let v_b = self.momentum * *momentum_b - lr * grad_bias;
            *momentum_b = v_b;
            *bias += v_b;
        } else {
            // Vanilla SGD: weight -= lr * grad
            for (w, gw) in weights.iter_mut().zip(grad_weights.iter()) {
                *w -= lr * gw;
            }
            *bias -= lr * grad_bias;
        }
    }

    fn create_state(&self, dim: usize) -> OptimizerState {
        if self.momentum > 0.0 {
            OptimizerState::Sgd {
                momentum_w: vec![0.0; dim],
                momentum_b: 0.0,
            }
        } else {
            OptimizerState::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vanilla_sgd_update() {
        let sgd = Sgd::new(0.0);
        let mut weights = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let mut bias = 0.5;
        let grad_bias = 0.05;
        let lr = 0.01;
        let mut state = sgd.create_state(3);

        sgd.update(&mut weights, &grads, &mut bias, grad_bias, lr, 1, &mut state);

        // weight -= lr * grad
        assert!((weights[0] - (1.0 - 0.01 * 0.1)).abs() < 1e-6);
        assert!((weights[1] - (2.0 - 0.01 * 0.2)).abs() < 1e-6);
        assert!((weights[2] - (3.0 - 0.01 * 0.3)).abs() < 1e-6);
        assert!((bias - (0.5 - 0.01 * 0.05)).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let sgd = Sgd::new(0.9);
        let mut weights = vec![1.0, 2.0];
        let grads = vec![1.0, 1.0];
        let mut bias = 0.5;
        let grad_bias = 1.0;
        let lr = 0.1;
        let mut state = sgd.create_state(2);

        // First step: velocity = 0.9*0 - 0.1*1.0 = -0.1; weight += -0.1
        sgd.update(&mut weights, &grads, &mut bias, grad_bias, lr, 1, &mut state);
        assert!((weights[0] - 0.9).abs() < 1e-6);
        assert!((weights[1] - 1.9).abs() < 1e-6);
        assert!((bias - 0.4).abs() < 1e-6);

        // Second step: velocity = 0.9*(-0.1) - 0.1*1.0 = -0.19; weight += -0.19
        sgd.update(&mut weights, &grads, &mut bias, grad_bias, lr, 2, &mut state);
        assert!((weights[0] - (0.9 - 0.19)).abs() < 1e-6);
        assert!((weights[1] - (1.9 - 0.19)).abs() < 1e-6);
        assert!((bias - (0.4 - 0.19)).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_zero_gradient() {
        let sgd = Sgd::new(0.0);
        let mut weights = vec![5.0, 3.0];
        let grads = vec![0.0, 0.0];
        let mut bias = 1.0;
        let mut state = sgd.create_state(2);

        sgd.update(&mut weights, &grads, &mut bias, 0.0, 0.1, 1, &mut state);
        assert_eq!(weights, vec![5.0, 3.0]);
        assert_eq!(bias, 1.0);
    }

    #[test]
    fn test_sgd_create_state_no_momentum() {
        let sgd = Sgd::new(0.0);
        let state = sgd.create_state(10);
        assert!(matches!(state, OptimizerState::None));
    }

    #[test]
    fn test_sgd_create_state_with_momentum() {
        let sgd = Sgd::new(0.9);
        let state = sgd.create_state(5);
        match state {
            OptimizerState::Sgd {
                momentum_w,
                momentum_b,
            } => {
                assert_eq!(momentum_w.len(), 5);
                assert!(momentum_w.iter().all(|&v| v == 0.0));
                assert_eq!(momentum_b, 0.0);
            }
            _ => panic!("Expected Sgd state"),
        }
    }

    #[test]
    fn test_sgd_large_lr() {
        let sgd = Sgd::new(0.0);
        let mut weights = vec![0.0];
        let grads = vec![1.0];
        let mut bias = 0.0;
        let mut state = sgd.create_state(1);

        sgd.update(&mut weights, &grads, &mut bias, 1.0, 10.0, 1, &mut state);
        assert!((weights[0] - (-10.0)).abs() < 1e-6);
        assert!((bias - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_negative_gradient() {
        let sgd = Sgd::new(0.0);
        let mut weights = vec![0.0];
        let grads = vec![-1.0];
        let mut bias = 0.0;
        let mut state = sgd.create_state(1);

        sgd.update(&mut weights, &grads, &mut bias, -1.0, 0.1, 1, &mut state);
        // weight -= lr * (-1) = weight + 0.1
        assert!((weights[0] - 0.1).abs() < 1e-6);
        assert!((bias - 0.1).abs() < 1e-6);
    }
}
