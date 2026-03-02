use std::path::PathBuf;

use klearu_llm::sparse::calibrate::CalibrationData;
use klearu_llm::weight::load_model;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model_dir> [--samples N] [--hidden-dim N] [--lr F] [--epochs N] [--output DIR]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);

    let mut num_samples = 16usize;
    let mut hidden_dim = 128usize;
    let mut lr = 0.01f32;
    let mut epochs = 100usize;
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--samples" => {
                i += 1;
                num_samples = args[i].parse().expect("Invalid --samples");
            }
            "--hidden-dim" => {
                i += 1;
                hidden_dim = args[i].parse().expect("Invalid --hidden-dim");
            }
            "--lr" => {
                i += 1;
                lr = args[i].parse().expect("Invalid --lr");
            }
            "--epochs" => {
                i += 1;
                epochs = args[i].parse().expect("Invalid --epochs");
            }
            "--output" => {
                i += 1;
                output_dir = Some(PathBuf::from(&args[i]));
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let out = output_dir.unwrap_or_else(|| model_dir.join("predictors"));

    eprintln!("Loading model from {}...", model_dir.display());
    let mut model = load_model(&model_dir).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    });
    let config = model.config.clone();

    eprintln!(
        "Model loaded: {} layers, hidden_size={}, vocab_size={}",
        config.num_layers, config.hidden_size, config.vocab_size
    );

    // Run calibration on synthetic sequences
    eprintln!("Running calibration with {num_samples} synthetic tokens...");
    let mut cal_data = CalibrationData::new(&config);

    // Generate synthetic token sequences for calibration.
    // Use simple sequential token IDs spread across the vocabulary.
    let tokens: Vec<u32> = (0..num_samples as u32)
        .map(|i| i % config.vocab_size as u32)
        .collect();

    cal_data.calibrate_sequence(&mut model, &tokens);
    cal_data.finalize();

    eprintln!("Training predictors (hidden_dim={hidden_dim}, lr={lr}, epochs={epochs})...");
    let store = cal_data.train_predictors(&config, hidden_dim, lr, epochs);

    eprintln!("Saving predictors to {}...", out.display());
    store.save(&out).unwrap_or_else(|e| {
        eprintln!("Failed to save predictors: {e}");
        std::process::exit(1);
    });

    eprintln!("Done! Saved {} layers of predictors.", config.num_layers);
}
