use std::io::{self, BufRead, Write};
use std::net::TcpStream;
use std::path::PathBuf;

use rand::SeedableRng;

use klearu_llm::generate::chat_template::{ChatMessage, ChatTemplate};
use klearu_llm::generate::sampler::SamplerConfig;
use klearu_llm::tokenizer::Tokenizer;
use klearu_mpc::beaver::{DummyTripleGen, DummyTripleGen128};
use klearu_llm::generate::pipeline::detect_eos_token;
use klearu_private::ferret_triples::{setup_ferret_client, FerretTripleGen, FerretTripleGen128};
use klearu_private::private_pipeline::{
    generate_private, generate_private_secure, PrivateConfig, SecurityLevel,
};
use klearu_private::tcp_transport::TcpTransport;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model_dir> [--host <host:port>] [--temp <f32>] [--top-k <n>] \
             [--top-p <f32>] [--max-tokens <n>] [--template <name>] [--system <msg>] \
             [--sparse] [--neuron-sparsity <f32>] [--security <high|lower>] [--dummy-triples]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);

    let mut host = "localhost:9000".to_string();
    let mut temperature = 0.7f32;
    let mut top_k = 40usize;
    let mut top_p = 0.9f32;
    let mut max_tokens = 256usize;
    let mut template_name = "auto".to_string();
    let mut system_msg: Option<String> = None;
    let mut sparse = false;
    let mut neuron_sparsity = 0.5f32;
    let mut security = SecurityLevel::High;
    let mut dummy_triples = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                i += 1;
                host = args[i].clone();
            }
            "--temp" | "--temperature" => {
                i += 1;
                temperature = args[i].parse().expect("Invalid temperature");
            }
            "--top-k" => {
                i += 1;
                top_k = args[i].parse().expect("Invalid top-k");
            }
            "--top-p" => {
                i += 1;
                top_p = args[i].parse().expect("Invalid top-p");
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("Invalid max-tokens");
            }
            "--template" => {
                i += 1;
                template_name = args[i].clone();
            }
            "--system" => {
                i += 1;
                system_msg = Some(args[i].clone());
            }
            "--sparse" => {
                sparse = true;
            }
            "--neuron-sparsity" => {
                i += 1;
                neuron_sparsity = args[i].parse().expect("Invalid neuron-sparsity");
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

    // Detect template
    let template = match template_name.as_str() {
        "auto" => {
            let detected = ChatTemplate::detect(&model_dir).unwrap_or(ChatTemplate::Raw);
            eprintln!("Auto-detected template: {detected:?}");
            detected
        }
        "zephyr" => ChatTemplate::Zephyr,
        "chatml" => ChatTemplate::ChatML,
        "llama2" => ChatTemplate::Llama2,
        "llama3" => ChatTemplate::Llama3,
        "mistral" => ChatTemplate::MistralInstruct,
        "raw" => ChatTemplate::Raw,
        other => {
            eprintln!("Unknown template: {other}");
            std::process::exit(1);
        }
    };

    // Detect EOS token
    let eos_token_id = detect_eos_token(&model_dir);
    eprintln!("EOS token ID: {eos_token_id:?}");

    // Load model and tokenizer
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

    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json")).unwrap_or_else(|e| {
        eprintln!("Failed to load tokenizer: {e}");
        std::process::exit(1);
    });

    let sampler = SamplerConfig {
        temperature,
        top_k,
        top_p,
        repetition_penalty: 1.1,
    };

    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = &system_msg {
        history.push(ChatMessage::system(sys.clone()));
        eprintln!("System: {sys}");
    }

    let mut rng = rand::rngs::StdRng::from_entropy();

    eprintln!("Ready. Type a message (Ctrl-D to quit).\n");

    let stdin = io::stdin();
    loop {
        eprint!("> ");
        io::stderr().flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).unwrap() == 0 {
            eprintln!();
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        history.push(ChatMessage::user(input));
        let prompt = template.apply(&history);

        // Tokenize
        let prompt_tokens = tokenizer.encode(&prompt).unwrap_or_else(|e| {
            eprintln!("Tokenization error: {e}");
            std::process::exit(1);
        });

        let stream = TcpStream::connect(&host).unwrap_or_else(|e| {
            eprintln!("Failed to connect to {host}: {e}");
            std::process::exit(1);
        });

        // Resolve the server's IP for Ferret NetIO (needs dotted-decimal, not hostname)
        let server_ip = stream.peer_addr()
            .map(|a| a.ip().to_string())
            .unwrap_or_else(|_| "127.0.0.1".to_string());
        let mut transport = TcpTransport::new(stream).unwrap_or_else(|e| {
            eprintln!("Transport error: {e}");
            std::process::exit(1);
        });

        let priv_config = PrivateConfig {
            max_new_tokens: max_tokens,
            sampler: sampler.clone(),
            eos_token_id,
            sparse,
            neuron_sparsity,
            security,
        };

        let mut response = String::new();
        let result = if dummy_triples {
            match security {
                SecurityLevel::High => {
                    let mut triples = DummyTripleGen128::new(0, 42);
                    generate_private_secure(
                        &mut model,
                        &prompt_tokens,
                        &priv_config,
                        &mut triples,
                        &mut transport,
                        &mut rng,
                        |token_text, _token_id| {
                            print!("{token_text}");
                            io::stdout().flush().unwrap();
                            response.push_str(token_text);
                            true
                        },
                        &tokenizer,
                    )
                }
                SecurityLevel::Lower => {
                    let mut triples = DummyTripleGen::new(0, 42);
                    generate_private(
                        &mut model,
                        &prompt_tokens,
                        &priv_config,
                        &mut triples,
                        &mut transport,
                        &mut rng,
                        |token_text, _token_id| {
                            print!("{token_text}");
                            io::stdout().flush().unwrap();
                            response.push_str(token_text);
                            true
                        },
                        &tokenizer,
                    )
                }
            }
        } else {
            // Set up Ferret OT connection (client = BOB = OT receiver)
            match setup_ferret_client(&server_ip, &mut transport) {
                Ok((cot, corrections)) => {
                    match security {
                        SecurityLevel::High => {
                            let mut triples = FerretTripleGen128::new(0, cot, corrections);
                            generate_private_secure(
                                &mut model,
                                &prompt_tokens,
                                &priv_config,
                                &mut triples,
                                &mut transport,
                                &mut rng,
                                |token_text, _token_id| {
                                    print!("{token_text}");
                                    io::stdout().flush().unwrap();
                                    response.push_str(token_text);
                                    true
                                },
                                &tokenizer,
                            )
                        }
                        SecurityLevel::Lower => {
                            let mut triples = FerretTripleGen::new(0, cot, corrections);
                            generate_private(
                                &mut model,
                                &prompt_tokens,
                                &priv_config,
                                &mut triples,
                                &mut transport,
                                &mut rng,
                                |token_text, _token_id| {
                                    print!("{token_text}");
                                    io::stdout().flush().unwrap();
                                    response.push_str(token_text);
                                    true
                                },
                                &tokenizer,
                            )
                        }
                    }
                }
                Err(e) => Err(e),
            }
        };
        match result {
            Ok(_) => {
                println!();
                history.push(ChatMessage::assistant(response));
            }
            Err(e) => {
                eprintln!("\nGeneration error: {e}");
            }
        }
    }
}
