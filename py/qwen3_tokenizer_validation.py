#!/usr/bin/env python3
"""
Qwen3 HuggingFace Tokenizer Validation Script
Validates token IDs and mappings against your C++ tokenizer test results
"""

from transformers import AutoTokenizer
import sys

def main():
    print("Loading official Qwen3 tokenizer from HuggingFace...")
    
    try:
        # Load the official Qwen3 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        print(f"✅ Tokenizer loaded successfully")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        sys.exit(1)
    
    print("=" * 60)
    print("TESTING TOKEN MAPPINGS FROM YOUR C++ TEST")
    print("=" * 60)
    
    # Test cases from your C++ code and Python analysis
    test_tokens = [
        (0, "!"),
        (1, "\""), 
        (10, "+"),
        (32, "A"),
        (64, "a"),
        (65, "b"),
        (66, "c"),
        (100, "§"),
        (200, "Č"),
    ]
    
    print("\n=== DECODING TOKEN IDS ===")
    for token_id, expected in test_tokens:
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            status = "✅" if decoded == expected else "❌"
            print(f"Token {token_id:3d}: '{decoded}' (expected: '{expected}') {status}")
        except Exception as e:
            print(f"Token {token_id:3d}: ERROR - {e}")
    
    print("\n=== SPECIAL TOKENS VALIDATION ===")
    special_tokens_test = [
        (151645, "<|im_end|>"),
        (151643, "<|endoftext|>"),
    ]
    
    for token_id, expected in special_tokens_test:
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            status = "✅" if decoded == expected else "❌"
            print(f"Token {token_id}: '{decoded}' (expected: '{expected}') {status}")
        except Exception as e:
            print(f"Token {token_id}: ERROR - {e}")
    
    print("\n=== ENCODING SPECIFIC STRINGS ===")
    encode_tests = [
        "A",
        "b", 
        "!",
        "§",
        "<|endoftext|>",
        "<|im_end|>",
        "Hello world!"
    ]
    
    for text in encode_tests:
        try:
            # For special tokens, we need to allow them
            if text.startswith("<|") and text.endswith("|>"):
                encoded = tokenizer.encode(text, allowed_special={text})
            else:
                encoded = tokenizer.encode(text)
            
            print(f"'{text}' -> {encoded}")
            
            # Verify round-trip
            decoded = tokenizer.decode(encoded, skip_special_tokens=False)
            round_trip_ok = "✅" if decoded == text else f"❌ (got: '{decoded}')"
            print(f"  Round-trip: {round_trip_ok}")
            
        except Exception as e:
            print(f"'{text}' -> ERROR: {e}")
        print()
    
    print("=" * 60)
    print("TOKENIZER METADATA")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    # Check if we can access special tokens
    print(f"\nSpecial tokens map:")
    for name, token in tokenizer.special_tokens_map.items():
        if hasattr(tokenizer, f"{name}_id"):
            token_id = getattr(tokenizer, f"{name}_id")
            print(f"  {name}: '{token}' (ID: {token_id})")
        else:
            print(f"  {name}: '{token}' (ID: unknown)")
    
    print("\n=== TESTING YOUR SPECIFIC EXAMPLES ===")
    # Test the exact sequence from your round-trip test
    original_text = "This is a test."
    encoded_ids = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(encoded_ids)
    
    print(f"Original: '{original_text}'")
    print(f"Encoded:  {encoded_ids}")
    print(f"Decoded:  '{decoded_text}'")
    print(f"Round-trip match: {'✅' if original_text == decoded_text else '❌'}")
    
    # Test the "Hello world!" from your encode test
    hello_text = "Hello world!"
    hello_encoded = tokenizer.encode(hello_text)
    print(f"\n'{hello_text}' -> {hello_encoded}")
    print(f"Token count: {len(hello_encoded)} (your test expects < 10)")

if __name__ == "__main__":
    main()
