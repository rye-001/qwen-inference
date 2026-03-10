The architecture: 
- QwenGGUFLoader reads, 
- Qwen3Model holds data, 
- Tokenizer translates text to IDs (numbers), and 
- Qwen3ForwardPass will run inference.
- main.cpp parses CLI args, detects CPU features, and loads the Qwen3 model in two passes: first metadata, then tensors into a ggml_context.