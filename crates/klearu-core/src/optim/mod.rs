mod sgd;
mod adam;
mod hogwild;

pub use sgd::Sgd;
pub use adam::Adam;
pub use hogwild::HogwildNetwork;

/// Trait for parameter optimizers.
pub trait Optimizer: Send + Sync {
    /// Update weights and bias given gradients.
    fn update(
        &self,
        weights: &mut [f32],
        grad_weights: &[f32],
        bias: &mut f32,
        grad_bias: f32,
        lr: f64,
        step: u64,
        state: &mut OptimizerState,
    );

    /// Create fresh optimizer state for a neuron with given dimension.
    fn create_state(&self, dim: usize) -> OptimizerState;
}

/// Per-neuron optimizer state (momentum, Adam moments, etc.)
#[derive(Debug, Clone)]
pub enum OptimizerState {
    None,
    Sgd {
        momentum_w: Vec<f32>,
        momentum_b: f32,
    },
    Adam {
        m_w: Vec<f32>,
        v_w: Vec<f32>,
        m_b: f32,
        v_b: f32,
    },
}
