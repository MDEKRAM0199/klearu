use std::net::TcpListener;
use std::path::PathBuf;

use klearu_mpc::beaver::{DummyTripleGen, DummyTripleGen128};
use klearu_private::ferret_triples::{setup_ferret_server, FerretTripleGen, FerretTripleGen128};
use klearu_private::private_pipeline::{serve_private, serve_private_secure, SecurityLevel};
use klearu_private::tcp_transport::TcpTransport;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model_dir> [--port <port>] [--security <high|lower>] [--dummy-triples]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let mut port = 9000u16;
    let mut security = SecurityLevel::High;
    let mut dummy_triples = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args[i].parse().expect("Invalid port");
            }
            "--security" => {
                i += 1;
                security = match args[i].as_str() {
                    "high" | "High" => SecurityLevel::High,
                    "lower" | "Lower" => SecurityLevel::Lower,
                    other => {
                        eprintln!("Unknown security level: {other} (use 'high' or 'lower')");
                        std::process::exit(1);
                    }
                };
            }
            "--dummy-triples" => {
                dummy_triples = true;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if dummy_triples {
        eprintln!("WARNING: Using dummy (insecure) triple generation for development only.");
    }

    eprintln!("Loading model from {}...", model_dir.display());
    let mut model = klearu_llm::weight::load_model(&model_dir).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "Model loaded: {} layers, hidden_size={}, vocab_size={}",
        model.config.num_layers,
        model.config.hidden_size,
        model.config.vocab_size,
    );

    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("Failed to bind {addr}: {e}");
        std::process::exit(1);
    });
    eprintln!("Listening on {addr}...");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let peer = stream.peer_addr().ok();
                eprintln!("Client connected: {peer:?}");

                let mut transport = match TcpTransport::new(stream) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("Transport error: {e}");
                        continue;
                    }
                };

                let result = if dummy_triples {
                    match security {
                        SecurityLevel::High => {
                            let mut triples = DummyTripleGen128::new(1, 42);
                            serve_private_secure(&mut model, &mut triples, &mut transport)
                        }
                        SecurityLevel::Lower => {
                            let mut triples = DummyTripleGen::new(1, 42);
                            serve_private(&mut model, &mut triples, &mut transport)
                        }
                    }
                } else {
                    // Set up Ferret OT connection (server = ALICE = OT sender)
                    match setup_ferret_server(&mut transport) {
                        Ok((cot, corrections)) => {
                            match security {
                                SecurityLevel::High => {
                                    let mut triples = FerretTripleGen128::new(1, cot, corrections);
                                    serve_private_secure(&mut model, &mut triples, &mut transport)
                                }
                                SecurityLevel::Lower => {
                                    let mut triples = FerretTripleGen::new(1, cot, corrections);
                                    serve_private(&mut model, &mut triples, &mut transport)
                                }
                            }
                        }
                        Err(e) => Err(e),
                    }
                };

                match result {
                    Ok(()) => eprintln!("Session complete."),
                    Err(e) => eprintln!("Session error: {e}"),
                }

                eprintln!("Ready for next connection...");
            }
            Err(e) => {
                eprintln!("Accept error: {e}");
            }
        }
    }
}
