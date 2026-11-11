#include "models/model.hpp"
#include "models/cuda_model.hpp"
#include "ops/cuda_backend.hpp"
#include "gguf/gguf_parser.hpp"
#include "tokenizer/tokenizer.hpp"
#include "core/tensor.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] <model.gguf> [prompt]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -t, --tokens N       Max tokens to generate (default: 512)\n";
    std::cout << "  --temp F             Temperature (default: 0.7)\n";
    std::cout << "  --top-k N            Top-k sampling (default: 40)\n";
    std::cout << "  --top-p F            Top-p sampling (default: 0.95)\n";
    std::cout << "  --seed N             Random seed (default: random)\n";
    std::cout << "  --device DEV         auto | cpu | cuda (default: auto)\n";
    std::cout << "  --gpu-precision P    int16 | fp32 GPU weights (default: int16)\n";
    std::cout << "  --chat               Apply the model's chat template\n";
    std::cout << "  --no-chat            Raw completion, even for instruct models\n";
    std::cout << "  --system TEXT        System message (implies --chat)\n";
    std::cout << "  --tokenizer PATH     Tokenizer file (tokenizer.json)\n";
    std::cout << "  -i, --interactive    Interactive chat mode\n";
    std::cout << "  -h, --help           Show this help\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " model.gguf \"Hello, how are you?\"\n";
    std::cout << "  " << prog << " --device cuda model.gguf \"Once upon a time\"\n";
    std::cout << "  " << prog << " -i model.gguf\n";
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string model_path;
    std::string tokenizer_path;
    std::string prompt;
    std::string device = "auto";
    std::string system_prompt;
    bool interactive = false;
    bool chat = false;
    bool chat_set = false;

    inference::InferenceConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--interactive") {
            interactive = true;
        } else if (arg == "-t" || arg == "--tokens") {
            if (++i < argc) config.max_tokens = std::stoi(argv[i]);
        } else if (arg == "--temp") {
            if (++i < argc) config.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i < argc) config.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i < argc) config.top_p = std::stof(argv[i]);
        } else if (arg == "--seed") {
            if (++i < argc) config.seed = std::stoi(argv[i]);
        } else if (arg == "--device") {
            if (++i < argc) device = argv[i];
        } else if (arg == "--gpu-precision") {
            if (++i < argc) {
                std::string p = argv[i];
                if (p == "int16") config.gpu_int16 = true;
                else if (p == "fp32") config.gpu_int16 = false;
                else std::cerr << "Warning: --gpu-precision expects int16 or fp32\n";
            }
        } else if (arg == "--chat") {
            chat = true;
            chat_set = true;
        } else if (arg == "--no-chat") {
            chat = false;
            chat_set = true;
        } else if (arg == "--system") {
            if (++i < argc) {
                system_prompt = argv[i];
                chat = true;
                chat_set = true;
            }
        } else if (arg == "--tokenizer") {
            if (++i < argc) tokenizer_path = argv[i];
        } else if (arg[0] != '-') {
            if (model_path.empty()) {
                model_path = arg;
            } else if (prompt.empty()) {
                prompt = arg;
            }
        }
    }

    if (model_path.empty() && !interactive) {
        std::cerr << "Error: Model path required\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        // Load model
        std::cout << "Loading model: " << model_path << "...\n";
        auto start = std::chrono::high_resolution_clock::now();

        inference::GGUFParser parser(model_path);
        std::cout << "  Architecture: " << parser.get_arch_name() << "\n";
        std::cout << "  Layers: " << parser.get_block_count() << "\n";
        std::cout << "  Embed dim: " << parser.get_embedding_dim() << "\n";
        std::cout << "  Heads: " << parser.get_num_heads() << "\n";
        std::cout << "  Vocab size: " << parser.get_vocab_size() << "\n";

        // Load weights
        auto weights = inference::load_model_weights(parser, inference::Device::cpu());

        // Create model (moves weights in; no extra copy)
        inference::Model model(std::move(weights), inference::Device::cpu());

        auto load_time = std::chrono::high_resolution_clock::now() - start;
        std::cout << "Model loaded in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(load_time).count()
                  << " ms\n";

        // Load tokenizer
        std::string tok_path = tokenizer_path;
        if (tok_path.empty()) {
            // Try to find tokenizer next to model
            std::filesystem::path model_dir = std::filesystem::path(model_path).parent_path();
            for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
                std::string name = entry.path().filename().string();
                if (name == "tokenizer.json" || name == "tokenizer.model") {
                    tok_path = entry.path().string();
                    break;
                }
            }
        }

        inference::Tokenizer tokenizer;
        bool tokenizer_ready = false;
        if (!tok_path.empty()) {
            try {
                std::cout << "Loading tokenizer: " << tok_path << "...\n";
                tokenizer.load(tok_path);
                tokenizer_ready = true;
            } catch (const std::exception& e) {
                std::cerr << "Tokenizer file failed (" << e.what() << "); trying GGUF metadata.\n";
            }
        }

        if (!tokenizer_ready) {
            try {
                tokenizer.load_from_gguf(parser);
                tokenizer_ready = true;
                std::cout << "Loaded tokenizer from GGUF metadata.\n";
            } catch (const std::exception& e) {
                std::cerr << "Warning: no usable tokenizer: " << e.what() << "\n";
            }
        }

        if (tokenizer_ready) {
            std::cout << "  Vocab size: " << tokenizer.vocab_size() << "\n";
            config.eos_token_id = tokenizer.eos_id();
        } else {
            std::cerr << "Falling back to byte tokenizer (output will be poor).\n";
        }

        // Interactive mode uses the model's chat template when one exists.
        // Single-prompt completion stays raw unless --chat is passed.
        if (!chat_set) chat = interactive && tokenizer_ready && tokenizer.has_chat_template();
        if (chat && tokenizer_ready) {
            std::cout << "Chat mode: "
                      << (tokenizer.has_chat_template() ? "model template"
                                                        : "generic fallback")
                      << "\n";
        }

        // Create CPU KV cache (also used as the fallback path)
        inference::KVCache kv_cache(model.n_layers(), model.max_seq_len(),
                                     model.n_heads(), model.head_dim());

        // Decide execution device
        bool use_cuda = false;
        if (device == "cuda" || device == "auto") {
            if (inference::cuda_model_available()) {
                if (device == "cuda") {
                    use_cuda = true;
                } else {
                    // auto: only use the GPU if the weights plausibly fit
                    const size_t need = inference::cuda_required_bytes(model, config.gpu_int16);
                    const size_t free_b = inference::cuda::free_memory();
                    use_cuda = need + (96ull << 20) < free_b;
                    if (!use_cuda) {
                        std::cout << "CUDA: model needs ~" << (need >> 20)
                                  << " MB (" << (config.gpu_int16 ? "int16" : "fp32")
                                  << " weights), only " << (free_b >> 20)
                                  << " MB VRAM free; using CPU.\n";
                    }
                }
            } else if (device == "cuda") {
                std::cerr << "CUDA backend unavailable; falling back to CPU.\n";
            }
        }

        if (use_cuda) {
            std::cout << "Device: CUDA (" << inference::cuda::device_name()
                      << ", " << (inference::cuda::free_memory() >> 20) << " MB free, "
                      << (config.gpu_int16 ? "int16" : "fp32") << " weights)\n";
        } else {
            std::cout << "Device: CPU\n";
        }

        // Generate on the selected device. In auto mode, a CUDA failure (e.g.
        // OOM) transparently falls back to CPU.
        inference::GenerationStats gpu_stats;
        auto generate_tokens = [&](const std::vector<int>& toks) {
            gpu_stats = {};
            if (use_cuda) {
                try {
                    return inference::generate_cuda(model, toks, config, &gpu_stats);
                } catch (const std::exception& e) {
                    if (device == "cuda") throw;
                    std::cerr << "CUDA generation failed (" << e.what()
                              << "); falling back to CPU.\n";
                    use_cuda = false;
                }
            }
            return model.generate(toks, config, kv_cache);
        };

        auto print_timing = [&](size_t new_tokens, std::chrono::duration<double> gen_time) {
            if (gpu_stats.decode_ms > 0.0 && gpu_stats.generated > 0) {
                const double decode_tps = gpu_stats.generated * 1000.0 / gpu_stats.decode_ms;
                std::cout << "[" << gpu_stats.generated << " tokens, "
                          << std::fixed << std::setprecision(2) << decode_tps
                          << " tok/s decode";
                if (gpu_stats.init_ms > 0.0) {
                    std::cout << ", GPU init " << std::setprecision(0)
                              << gpu_stats.init_ms << " ms";
                }
                std::cout << ", prefill " << std::setprecision(0)
                          << gpu_stats.prefill_ms << " ms]\n";
                return;
            }
            double tps = new_tokens / gen_time.count();
            std::cout << "[" << new_tokens << " tokens, " << std::fixed
                      << std::setprecision(2) << tps << " tok/s]\n";
        };

        if (interactive) {
            std::cout << "\n=== Interactive Mode ===\n";
            std::cout << "Type 'exit' or 'quit' to exit\n";
            std::cout << "Type 'clear' to clear context\n\n";

            std::vector<inference::ChatMessage> history;
            auto reset_history = [&] {
                history.clear();
                if (!system_prompt.empty()) history.push_back({"system", system_prompt});
            };
            reset_history();

            std::string line;
            while (true) {
                std::cout << "> ";
                std::getline(std::cin, line);

                if (line == "exit" || line == "quit") break;
                if (line == "clear") {
                    kv_cache.clear();
                    reset_history();
                    std::cout << "[Context cleared]\n";
                    continue;
                }
                if (line.empty()) continue;

                // Encode
                std::vector<int> prompt_tokens;
                if (tokenizer.vocab_size() > 0) {
                    if (chat) {
                        history.push_back({"user", line});
                        std::string rendered =
                            tokenizer.apply_chat_template(history, true);
                        prompt_tokens = tokenizer.encode(
                            rendered, tokenizer.chat_needs_bos(), false);
                        // Re-prefill the full conversation on the CPU path.
                        kv_cache.clear();
                    } else {
                        prompt_tokens = tokenizer.encode(line, true, false);
                    }
                } else {
                    // Dummy: each char = token
                    for (char c : line) prompt_tokens.push_back(static_cast<int>(c));
                }

                // Generate
                std::cout << "Generating...\n";
                auto gen_start = std::chrono::high_resolution_clock::now();
                auto generated = generate_tokens(prompt_tokens);
                auto gen_time = std::chrono::high_resolution_clock::now() - gen_start;

                // Decode (only new tokens)
                std::vector<int> new_tokens(generated.begin() + prompt_tokens.size(), generated.end());
                std::string output;
                if (tokenizer.vocab_size() > 0) {
                    output = tokenizer.decode(new_tokens);
                } else {
                    for (int t : new_tokens) output += static_cast<char>(t);
                }

                std::cout << output << "\n\n";
                print_timing(new_tokens.size(), std::chrono::duration<double>(gen_time));
                std::cout << "\n";

                if (chat) history.push_back({"assistant", output});
            }
        } else {
            // Single prompt
            if (prompt.empty()) {
                std::cerr << "Error: Prompt required\n";
                return 1;
            }

            std::cout << "\nPrompt: " << prompt << "\n\n";

            std::vector<int> prompt_tokens;
            if (tokenizer.vocab_size() > 0) {
                if (chat) {
                    std::vector<inference::ChatMessage> history;
                    if (!system_prompt.empty()) {
                        history.push_back({"system", system_prompt});
                    }
                    history.push_back({"user", prompt});
                    std::string rendered =
                        tokenizer.apply_chat_template(history, true);
                    prompt_tokens = tokenizer.encode(
                        rendered, tokenizer.chat_needs_bos(), false);
                } else {
                    prompt_tokens = tokenizer.encode(prompt, true, false);
                }
            } else {
                for (char c : prompt) prompt_tokens.push_back(static_cast<int>(c));
            }

            std::cout << "Prompt tokens: " << prompt_tokens.size() << "\n";

            auto gen_start = std::chrono::high_resolution_clock::now();
            auto generated = generate_tokens(prompt_tokens);
            auto gen_time = std::chrono::high_resolution_clock::now() - gen_start;

            std::vector<int> new_tokens(generated.begin() + prompt_tokens.size(), generated.end());
            std::string output;
            if (tokenizer.vocab_size() > 0) {
                output = tokenizer.decode(new_tokens);
            } else {
                for (int t : new_tokens) output += static_cast<char>(t);
            }

            std::cout << output << "\n\n";
            std::cout << "Generated " << new_tokens.size() << " tokens in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(gen_time).count()
                      << " ms total\n";
            print_timing(new_tokens.size(), std::chrono::duration<double>(gen_time));
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}