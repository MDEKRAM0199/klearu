use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashFunctionType {
    SimHash,
    WtaHash,
    DwtaHash,
    MinHash,
    SparseRandomProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BucketType {
    Fifo,
    Reservoir,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingType {
    Vanilla,
    TopK,
    Threshold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Identity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    Sgd,
    Adam,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LshConfig {
    pub hash_function: HashFunctionType,
    pub bucket_type: BucketType,
    pub num_tables: usize,
    pub range_pow: usize,
    pub num_hashes: usize,
    pub bucket_capacity: usize,
    pub rebuild_interval_base: usize,
    pub rebuild_decay: f64,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            hash_function: HashFunctionType::SimHash,
            bucket_type: BucketType::Fifo,
            num_tables: 50,
            range_pow: 6,
            num_hashes: 6,
            bucket_capacity: 128,
            rebuild_interval_base: 100,
            rebuild_decay: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub input_dim: usize,
    pub num_neurons: usize,
    pub activation: ActivationType,
    pub lsh: LshConfig,
    pub sampling: SamplingType,
    pub sampling_threshold: usize,
    pub top_k: usize,
    pub is_output: bool,
}

impl LayerConfig {
    pub fn hidden(input_dim: usize, num_neurons: usize) -> Self {
        Self {
            input_dim,
            num_neurons,
            activation: ActivationType::Relu,
            lsh: LshConfig::default(),
            sampling: SamplingType::TopK,
            sampling_threshold: 2,
            top_k: num_neurons / 10,
            is_output: false,
        }
    }

    pub fn output(input_dim: usize, num_neurons: usize) -> Self {
        Self {
            input_dim,
            num_neurons,
            activation: ActivationType::Softmax,
            lsh: LshConfig::default(),
            sampling: SamplingType::TopK,
            sampling_threshold: 2,
            top_k: num_neurons / 10,
            is_output: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub layers: Vec<LayerConfig>,
    pub optimizer: OptimizerType,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_threads: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: OptimizerType::Adam,
            learning_rate: 0.001,
            batch_size: 128,
            num_threads: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlideConfig {
    pub network: NetworkConfig,
    pub seed: u64,
    pub hogwild: bool,
}

impl Default for SlideConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig::default(),
            seed: 42,
            hogwild: true,
        }
    }
}
