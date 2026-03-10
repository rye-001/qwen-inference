# qwen-inference-engine

A from-scratch LLM inference server in C++ using [ggml](https://github.com/ggerganov/llama.cpp/tree/master/ggml). Serve 10 concurrent users from a single 32GB Mac.

**~6,700 lines of C++.** The entire pipeline — GGUF loading, tokenization, forward pass, KV cache, sampling, HTTP streaming — in a codebase you can read in an afternoon.

## Why not just use llama.cpp?

You probably should! [llama.cpp](https://github.com/ggerganov/llama.cpp) supports more models, more hardware, and more features. This project exists because:

**If you need a small, self-contained inference server** — for an internal tool, a small team, a startup that wants to stop paying per-token API costs, or an on-prem deployment where data can't leave the building — this gives you exactly that with nothing extra.

**If you want to understand how LLM inference works** — llama.cpp is ~300K lines. This is 6,700. Every layer of the stack is visible and modifiable. You can trace a request from HTTP to tensor op without getting lost.

**If you want to hack on the model itself** — the forward pass lives in one file. Swap layers, add probes, prune attention heads, tap intermediate hidden states for [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) experiments. The kind of model surgery that's painful in a 300K-line codebase is straightforward here.

## What it does

- **Batched inference** — Slot-based KV cache with batched decode. ~4× throughput over sequential processing for 10 concurrent users.
- **Prefix caching** — System prompt KV state cloned per slot. 2.25× faster wall time, 5.3× faster time-to-first-token.
- **OpenAI-compatible API** — `/v1/completions` with SSE streaming. Drop-in for existing tools.
- **Grammar-constrained generation** — GBNF grammars for structured output (DSL, JSON).
- **Speculative decoding** — Prompt-lookup based, no draft model needed.
- **Metal acceleration** — Apple Silicon via ggml's Metal backend.

## Supported Models

| Model | Quant | Memory | Status |
|-------|-------|--------|--------|
| Qwen 2.5 Coder 14B Instruct | Q4_0 | ~9 GB | ✅ Primary target |
| Qwen 3 1.7B | BF16 | ~3.4 GB | ✅ Tested |
| Qwen 3 0.6B | Q8_0 | ~0.6 GB | ✅ Tested |

Supports both Qwen2 and Qwen3 architectures.

## Quick Start

```bash
# Build (macOS)
mkdir build && cd build
cmake -DGGML_USE_METAL=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)

# Build (Linux)
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Start the server

```bash
./bin/http_server --model path/to/model.gguf --port 8080
```

```bash
# Generate
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write hello world in Python:", "max_tokens": 100}'

# Stream
curl -N http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quicksort:", "max_tokens": 200, "stream": true}'

# With cached system prompt (subsequent requests skip system prompt prefill)
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "You are a coding assistant.", "prompt": "Fix this bug:", "max_tokens": 200}'
```

### Or use the CLI

```bash
./bin/qwen3 model.gguf --chat                  # interactive
./bin/qwen3 model.gguf --chat --speculative     # with speculative decoding
./bin/qwen3 model.gguf -p "Hello, world!"       # single prompt
```

## Architecture

```
src/
├── qwen3-core/              # Core inference engine
│   ├── gguf-loader.*        #   GGUF parsing + mmap tensor loading
│   ├── qwen3-model.*        #   Model weights + metadata
│   ├── forward-pass.*       #   Prefill & batched decode graphs
│   ├── tokenizer.*          #   BPE tokenizer (GPT-2/Qwen)
│   ├── sampling.*           #   Greedy / temperature / top-k / top-p
│   ├── grammar.*            #   GBNF constrained generation
│   └── speculative.h        #   Prompt-lookup speculative decoding
├── kv-cache/
│   └── simple-kv-cache.*    #   Slot-based KV cache (clone/clear)
├── server/
│   ├── inference_server.h   #   Request queue, slots, decode loop
│   └── http_server.cpp      #   OpenAI-compatible HTTP API
└── cli/
    └── main.cpp             #   Chat + single-prompt modes
```

One inference thread owns the model. HTTP threads push requests to a queue and block on per-request token queues. All active slots decode together in a single batched forward pass.

## Research & Experimentation

The small codebase makes this useful for research that's painful in larger frameworks:

- **Layer skipping** — Comment out layers in the forward pass loop, run your test suite, see what breaks
- **Logit lens** — Project hidden states through the LM head after each layer to watch the model "decide"
- **Attention head pruning** — Zero out heads and measure impact on downstream tasks
- **Activation patching** — Swap activations between runs to isolate what drives specific outputs
- **Custom sampling** — The sampler is ~180 lines; easy to add new strategies

## Performance

Measured on M1 MacBook Pro (32GB) with Qwen 2.5 Coder 14B Q4:

| Metric | Value |
|--------|-------|
| Throughput (10 concurrent) | ~4× vs sequential |
| System prompt cache hit | 2.25× wall time, 5.3× TTFT |
| DSL code generation | 100% pass rate (10 concurrent) |

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `GGML_USE_METAL` | OFF | Metal GPU acceleration (macOS) |
| `QWEN3_BUILD_TESTS` | OFF | Build unit + integration tests |
| `QWEN3_ENABLE_OPENMP` | ON | OpenMP parallelism |

## Tests

```bash
cmake -DQWEN3_BUILD_TESTS=ON -DGGML_USE_METAL=ON -DCMAKE_BUILD_TYPE=Debug ..
make -j8

./bin/qwen3-unit-tests                    # tokenizer, vocab (no model needed)
./bin/qwen3-integration-tests             # e2e generation (needs model files)
./bin/qwen3-parallel-order-mngmt-test      # 10-concurrent DSL benchmark
```

Python benchmarks in `py/` for load testing and DSL validation.

## Dependencies

Fetched automatically via CMake:

- [ggml](https://github.com/ggerganov/llama.cpp) — tensor ops + Metal backend
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) — HTTP server
- [nlohmann/json](https://github.com/nlohmann/json) — JSON
- [Google Test](https://github.com/google/googletest) — testing

## Acknowledgements

Inspired by and built on concepts from [llama.cpp](https://github.com/ggerganov/llama.cpp) and [ggml](https://github.com/ggerganov/ggml). This project wouldn't exist without the foundational work by Georgi Gerganov and the ggml community.

## Contributing

Issues and PRs welcome. The codebase is intentionally small — please keep it that way.

## License

[MIT](LICENSE)