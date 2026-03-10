"""
Qwen3 RoPE + GQA Attention Golden Value Generator
This script generates golden reference values for testing your C++ RoPE and GQA Attention
implementations against the official Qwen3 transformers implementation using EXACT GGUF parameters.
"""

import torch
import numpy as np
import math

def create_qwen3_rope_golden_values():
    """Generate golden values for Qwen3 RoPE implementation using EXACT GGUF parameters"""
    
    # From your debug: Qwen3-0.6B-Q8_0.gguf metadata
    hidden_size = 1024           # qwen3.embedding_length
    num_attention_heads = 16     # qwen3.attention.head_count  
    num_key_value_heads = 8      # qwen3.attention.head_count_kv
    head_dim = 128              # qwen3.attention.key_length/value_length
    rope_theta = 1000000.0      # qwen3.rope.freq_base = 1e+06
    max_position_embeddings = 40960  # qwen3.context_length
    
    print("=== EXACT GGUF Configuration ===")
    print(f"hidden_size: {hidden_size}")
    print(f"num_attention_heads: {num_attention_heads}")
    print(f"num_key_value_heads: {num_key_value_heads}")
    print(f"head_dim: {head_dim}")
    print(f"rope_theta: {rope_theta}")
    print(f"max_position_embeddings: {max_position_embeddings}")
    
    # === Create Test Tensors ===
    seq_len = 4  # 4 tokens like your test
    test_head_dim = 8  # Your test uses 8-dim for simplicity
    
    # Use deterministic values for reproducible testing
    torch.manual_seed(42)
    
    # Create Q and K tensors: [seq_len, head_dim] (matching your test layout)
    Q = torch.randn(seq_len, test_head_dim, dtype=torch.float32)
    K = torch.randn(seq_len, test_head_dim, dtype=torch.float32)
    
    # Position IDs [0, 1, 2, 3] - matching your debug output
    position_ids = torch.arange(seq_len, dtype=torch.long)
    
    print(f"\n=== Input Tensors ===")
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"position_ids: {position_ids.tolist()}")
    
    # === Manual Qwen3 RoPE Implementation (NeoX style) ===
    # Based on your debug: mode=LLAMA_ROPE_TYPE_NEOX = 2
    def compute_rope_embeddings(head_dim, seq_len, rope_theta, position_ids):
        # Compute frequency for each dimension pair
        # NeoX RoPE: frequencies for pairs (0,head_dim/2), (1,head_dim/2+1), etc.
        dim_indices = torch.arange(0, head_dim // 2, dtype=torch.float32)
        freqs = 1.0 / (rope_theta ** (dim_indices / (head_dim // 2)))
        
        # Compute angles: position * frequency
        # position_ids: [seq_len], freqs: [head_dim//2]
        # angles: [seq_len, head_dim//2]
        angles = position_ids.float().unsqueeze(-1) * freqs.unsqueeze(0)
        
        # Compute cos and sin
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        
        return cos_vals, sin_vals
    
    def apply_neox_rope(tensor, cos_vals, sin_vals):
        """Apply NeoX-style RoPE rotation"""
        # tensor: [seq_len, head_dim]
        # cos_vals, sin_vals: [seq_len, head_dim//2]
        
        seq_len, head_dim = tensor.shape
        half_dim = head_dim // 2
        
        # Split tensor into two halves
        x1 = tensor[:, :half_dim]        # First half
        x2 = tensor[:, half_dim:]        # Second half
        
        # Apply NeoX rotation: 
        # x1_new = x1 * cos - x2 * sin
        # x2_new = x1 * sin + x2 * cos
        x1_new = x1 * cos_vals - x2 * sin_vals
        x2_new = x1 * sin_vals + x2 * cos_vals
        
        # Concatenate back
        rotated = torch.cat([x1_new, x2_new], dim=-1)
        return rotated
    
    # === Apply RoPE Using Exact Qwen3 Parameters ===
    cos_vals, sin_vals = compute_rope_embeddings(test_head_dim, seq_len, rope_theta, position_ids)
    
    Q_rope = apply_neox_rope(Q, cos_vals, sin_vals)
    K_rope = apply_neox_rope(K, cos_vals, sin_vals)
    
    # === Debug Information ===
    print(f"\n=== RoPE Computation Debug ===")
    print(f"Frequencies (first 4): {1.0 / (rope_theta ** (torch.arange(4).float() / (test_head_dim // 2)))}")
    print(f"Angles for position 0: {(0 * 1.0 / (rope_theta ** (torch.arange(test_head_dim//2).float() / (test_head_dim // 2))))}")
    print(f"Angles for position 1: {(1 * 1.0 / (rope_theta ** (torch.arange(test_head_dim//2).float() / (test_head_dim // 2))))}")
    
    # === Print Golden Values in C++ Format ===
    print(f"\n=== GOLDEN VALUES FOR C++ TESTING ===")
    
    print("\n// Input Q data (4 tokens, 8 dimensions)")
    print("float Q_data[n_tokens][head_dim] = {")
    for i in range(seq_len):
        values = Q[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    print("\n// Input K data (4 tokens, 8 dimensions)")  
    print("float K_data[n_tokens][head_dim] = {")
    for i in range(seq_len):
        values = K[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    print("\n// Expected Q after RoPE (4 tokens, 8 dimensions)")
    print("float Q_rope_expected[n_tokens][head_dim] = {")
    for i in range(seq_len):
        values = Q_rope[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    print("\n// Expected K after RoPE (4 tokens, 8 dimensions)")
    print("float K_rope_expected[n_tokens][head_dim] = {")
    for i in range(seq_len):
        values = K_rope[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    # === Validation Parameters ===
    print(f"\n=== C++ VALIDATION PARAMETERS ===")
    print(f"const float rope_freq_base = {rope_theta:.1f}f;")
    print(f"const int head_dim = {test_head_dim};")
    print(f"const int seq_len = {seq_len};")
    print(f"const int context_length = {max_position_embeddings};")
    print(f"int position_ids[{seq_len}] = {{{', '.join(map(str, position_ids.tolist()))}}};")
    
    # === Expected RoPE Parameters (from your debug) ===
    print(f"\n=== EXPECTED GGML_ROPE_EXT PARAMETERS ===")
    print(f"mode = LLAMA_ROPE_TYPE_NEOX (2)")
    print(f"n_ctx_orig = {max_position_embeddings}")
    print(f"freq_base = {rope_theta}")
    print(f"freq_scale = 1.0f")
    print(f"ext_factor = 0.0f")
    print(f"attn_factor = 1.0f") 
    print(f"beta_fast = 32.0f")
    print(f"beta_slow = 1.0f")
    
    # === Verification Test ===
    print(f"\n=== VERIFICATION ===")
    print("First token should have NO rotation (position 0):")
    print(f"Q[0] original:  {Q[0].numpy()}")
    print(f"Q[0] after RoPE: {Q_rope[0].numpy()}")
    print(f"Difference: {(Q_rope[0] - Q[0]).numpy()}")
    print("Should be near-zero for position 0")
    
    return {
        'input_Q': Q.numpy(),
        'input_K': K.numpy(), 
        'expected_Q_rope': Q_rope.numpy(),
        'expected_K_rope': K_rope.numpy(),
        'parameters': {
            'rope_freq_base': rope_theta,
            'head_dim': test_head_dim,
            'context_length': max_position_embeddings,
            'position_ids': position_ids.tolist()
        }
    }

def create_qwen3_gqa_attention_golden_values():
    """Generate golden values for Qwen3 GQA Attention implementation"""
    
    # === GQA Configuration (matching Qwen3 exact specs) ===
    seq_len = 4
    n_tokens = 4
    test_head_dim = 8  # Keep same as RoPE test for consistency
    num_q_heads = 16   # Q heads (from GGUF: qwen3.attention.head_count)
    num_kv_heads = 8   # KV heads (from GGUF: qwen3.attention.head_count_kv)
    rope_theta = 1000000.0
    
    print("\n" + "="*60)
    print("=== QWEN3 GQA ATTENTION GOLDEN VALUES ===")
    print("="*60)
    
    print(f"\n=== GQA Configuration ===")
    print(f"seq_len: {seq_len}")
    print(f"test_head_dim: {test_head_dim}")
    print(f"num_q_heads: {num_q_heads}")
    print(f"num_kv_heads: {num_kv_heads}")
    print(f"GQA ratio: {num_q_heads // num_kv_heads} (each KV head serves {num_q_heads // num_kv_heads} Q heads)")
    
    # === Create Test Tensors ===
    torch.manual_seed(42)  # Same seed for reproducibility
    
    # For testing, we'll use simplified tensors that match your function signature:
    # Q_rope: [seq_len, num_q_heads, head_dim] -> for simplicity: [seq_len, head_dim] (single head)
    # K_rope: [seq_len, num_kv_heads, head_dim] -> for simplicity: [seq_len, head_dim] (single head) 
    # V: [seq_len, num_kv_heads, head_dim] -> for simplicity: [seq_len, head_dim] (single head)
    
    # Generate position-aware Q and K using RoPE (reuse from above)
    position_ids = torch.arange(seq_len, dtype=torch.long)
    
    # Create base Q, K, V tensors
    Q_base = torch.randn(seq_len, test_head_dim, dtype=torch.float32) * 0.5  # Scale for numerical stability
    K_base = torch.randn(seq_len, test_head_dim, dtype=torch.float32) * 0.5
    V = torch.randn(seq_len, test_head_dim, dtype=torch.float32) * 0.5

    # Add output projection weight matrix [head_dim, head_dim] 
    attn_output_weight = torch.randn(test_head_dim, test_head_dim, dtype=torch.float32) * 0.1  # Small values for stability

    
    # Apply RoPE to Q and K
    def compute_rope_embeddings(head_dim, seq_len, rope_theta, position_ids):
        dim_indices = torch.arange(0, head_dim // 2, dtype=torch.float32)
        freqs = 1.0 / (rope_theta ** (dim_indices / (head_dim // 2)))
        angles = position_ids.float().unsqueeze(-1) * freqs.unsqueeze(0)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        return cos_vals, sin_vals
    
    def apply_neox_rope(tensor, cos_vals, sin_vals):
        seq_len, head_dim = tensor.shape
        half_dim = head_dim // 2
        x1 = tensor[:, :half_dim]
        x2 = tensor[:, half_dim:]
        x1_new = x1 * cos_vals - x2 * sin_vals
        x2_new = x1 * sin_vals + x2 * cos_vals
        return torch.cat([x1_new, x2_new], dim=-1)
    
    cos_vals, sin_vals = compute_rope_embeddings(test_head_dim, seq_len, rope_theta, position_ids)
    Q_rope = apply_neox_rope(Q_base, cos_vals, sin_vals)
    K_rope = apply_neox_rope(K_base, cos_vals, sin_vals)
    
    print(f"\n=== Input Tensors After RoPE ===")
    print(f"Q_rope shape: {Q_rope.shape}")  # [seq_len, head_dim]
    print(f"K_rope shape: {K_rope.shape}")  # [seq_len, head_dim]
    print(f"V shape: {V.shape}")            # [seq_len, head_dim]
    
    # === GQA Attention Implementation ===
    
    # Step 1: For testing, we simulate the "repeat" operation conceptually
    # In real GQA, K and V would be repeated from 8 heads to 16 heads
    # For our test: K_rope and V represent the "repeated" versions already
    K_repeated = K_rope  # In real implementation: ggml_repeat(K_rope, repeat_factor=2)
    V_repeated = V       # In real implementation: ggml_repeat(V, repeat_factor=2)
    
    # Step 2: QK^T - Matrix multiplication Q * K^T
    # Q_rope: [seq_len, head_dim], K_repeated: [seq_len, head_dim]
    # QK^T: [seq_len, seq_len]
    QK_scores = torch.matmul(Q_rope, K_repeated.transpose(-2, -1))
    
    # Step 3: Scale by 1/sqrt(d_k)
    d_k = test_head_dim
    scale_factor = 1.0 / math.sqrt(d_k)
    scaled_scores = QK_scores * scale_factor
    
    # Step 4: Apply Softmax
    attention_weights = torch.softmax(scaled_scores, dim=-1)
    
    # Step 5: Apply attention weights to V
    # attention_weights: [seq_len, seq_len], V_repeated: [seq_len, head_dim]
    # attention_output: [seq_len, head_dim]
    attention_output = torch.matmul(attention_weights, V_repeated)

    # Step 6: Apply output projection (NEW)
    # attention_output: [n_tokens, head_dim], attn_output_weight: [head_dim, head_dim]
    final_output = torch.matmul(attention_output, attn_output_weight)

    
    # === Debug Information ===
    print(f"\n=== GQA Attention Computation Debug ===")
    print(f"Scale factor (1/sqrt(d_k)): {scale_factor:.6f}")
    print(f"QK_scores shape: {QK_scores.shape}")
    print(f"QK_scores[0,0]: {QK_scores[0,0].item():.6f}")
    print(f"Scaled_scores[0,0]: {scaled_scores[0,0].item():.6f}")
    print(f"Attention_weights sum per row: {attention_weights.sum(dim=-1)}")  # Should be ~1.0
    print(f"Attention_output shape: {attention_output.shape}")
    
    # === Print Golden Values in C++ Format ===
    print(f"\n=== GQA ATTENTION GOLDEN VALUES FOR C++ TESTING ===")
    
    print("\n// Input Q after RoPE (4 tokens, 8 dimensions)")
    print("float Q_rope_data[n_tokens][head_dim] = {")
    for i in range(seq_len):
        values = Q_rope[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    print("\n// Input K after RoPE (4 tokens, 8 dimensions)")
    print("float K_rope_data[n_tokens][head_dim] = {")
    for i in range(seq_len):
        values = K_rope[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    print("\n// Input V (4 tokens, 8 dimensions)")
    print("float V_data[n_tokens][head_dim] = {")
    for i in range(seq_len):
        values = V[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    print("\n// Expected QK^T scores (4x4 attention matrix)")
    print("float QK_scores_expected[n_tokens][n_tokens] = {")
    for i in range(seq_len):
        values = QK_scores[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    print("\n// Expected scaled scores (4x4 attention matrix)")
    print("float scaled_scores_expected[n_tokens][n_tokens] = {")
    for i in range(seq_len):
        values = scaled_scores[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    print("\n// Expected attention weights after softmax (4x4 attention matrix)")
    print("float attention_weights_expected[n_tokens][n_tokens] = {")
    for i in range(seq_len):
        values = attention_weights[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")
    
    print("\n// Expected final attention output (4 tokens, 8 dimensions)")
    print("float attention_output_expected[n_tokens][head_dim] = {")
    for i in range(seq_len):
        values = attention_output[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")

    print("\n// Output projection weight matrix (8x8)")
    print("float attn_output_weight_data[head_dim][head_dim] = {")
    for i in range(test_head_dim):
        values = attn_output_weight[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")

    print("\n// Expected final output after projection (4 tokens, 8 dimensions)")
    print("float final_output_expected[n_tokens][head_dim] = {")
    for i in range(n_tokens):
        values = final_output[i].numpy()
        print(f"    {{{', '.join(f'{x:.6f}f' for x in values)}}},")
    print("};")

    # === Test Parameters ===
    print(f"\n=== GQA ATTENTION TEST PARAMETERS ===")
    print(f"const float scale_factor = {scale_factor:.6f}f;  // 1/sqrt({d_k})")
    print(f"const int seq_len = {seq_len};")
    print(f"const int head_dim = {test_head_dim};")
    print(f"const int num_q_heads = {num_q_heads};")
    print(f"const int num_kv_heads = {num_kv_heads};")
    print(f"const int gqa_ratio = {num_q_heads // num_kv_heads};")
    
    # === Validation Checks ===
    print(f"\n=== VALIDATION CHECKS ===")
    print("Attention weights should sum to 1.0 for each token:")
    for i in range(seq_len):
        row_sum = attention_weights[i].sum().item()
        print(f"  Token {i}: sum = {row_sum:.6f}")
    
    print(f"\nFirst attention weight row: {attention_weights[0].numpy()}")
    print(f"Sum: {attention_weights[0].sum().item():.6f} (should be ≈ 1.0)")
    
    return {
        'input_Q_rope': Q_rope.numpy(),
        'input_K_rope': K_rope.numpy(),
        'input_V': V.numpy(),
        'attn_output_weight': attn_output_weight.numpy(),
        'expected_QK_scores': QK_scores.numpy(),
        'expected_scaled_scores': scaled_scores.numpy(),
        'expected_attention_weights': attention_weights.numpy(),
        'expected_attention_output': attention_output.numpy(),
        'expected_final_output': final_output.numpy(),
        'parameters': {
            'scale_factor': scale_factor,
            'seq_len': seq_len,
            'head_dim': test_head_dim,
            'num_q_heads': num_q_heads,
            'num_kv_heads': num_kv_heads,
            'gqa_ratio': num_q_heads // num_kv_heads
        }
    }

if __name__ == "__main__":
    # Generate RoPE golden values
    # rope_values = create_qwen3_rope_golden_values()
    
    # Generate GQA Attention golden values
    gqa_values = create_qwen3_gqa_attention_golden_values()
    
    print(f"\n" + "="*60)
    print("=== SUMMARY ===")
    print("="*60)
    print("Golden values generated for:")
    print("1. Qwen3 RoPE (NeoX style)")
    print("2. Qwen3 GQA Attention (16 Q heads, 8 KV heads)")
    print("All values use the same random seed (42) for reproducibility.")
    print("Copy the C++ arrays above into your test files.")