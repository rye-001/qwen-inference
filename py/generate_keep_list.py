#!/usr/bin/env python3
"""
Vocabulary Extraction Script for Qwen Model Pruning

Extracts all unique tokens from conversation logs and generates:
- keep_list.bin - Sorted list of token IDs to keep


# Basic — vocab from conversation only
python generate_keep_list.py -i conversation.json

# With low-ID tokens (recommended)
python generate_keep_list.py -i conversation.json -t 10000

# Custom output dir
python generate_keep_list.py -i conversation.json -t 10000 -o ./models/

# Different model tokenizer
python generate_keep_list.py -i conversation.json -m Qwen/Qwen2.5-7B-Instruct

"""

import json
import struct
from pathlib import Path
from typing import Set, List
from transformers import AutoTokenizer


def load_conversation(file_path: str) -> List[str]:
    """Extract text content from JSON or plain text files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try JSON first
        if file_path.endswith('.json'):
            data = json.load(f)
            all_content = []
            messages = data.get('messages', [])
            if not messages and 'conv' in data:
                messages = data['conv'].get('messages', [])
            for msg in messages:
                content = msg.get('content', '').strip()
                if content:
                    all_content.append(content)
            print(f"Extracted {len(all_content)} messages from JSON")
            return all_content
        else:
            # Plain text / source code — return as single string
            content = f.read()
            print(f"Read {len(content)} chars from plain text file")
            return [content]

def tokenize_and_collect(texts: List[str], tokenizer) -> Set[int]:
    """Tokenize all texts and collect unique token IDs."""
    token_ids = set()
    
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_ids.update(tokens)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(texts)} messages, {len(token_ids)} unique tokens so far")
    
    print(f"\nTotal unique tokens from content: {len(token_ids)}")
    return token_ids


def add_base_vocabulary(token_ids: Set[int], tokenizer) -> Set[int]:
    """Adds a base vocabulary for English and JavaScript to ensure robustness."""
    
    js_keywords = [
        # Control flow
        "function", "const", "let", "var", "if", "else", "for", "while", "do",
        "return", "class", "extends", "constructor", "async", "await", "new",
        "Promise", "try", "catch", "finally", "import", "export", "from", "default",
        "true", "false", "null", "undefined", "NaN", "Infinity", "typeof", "instanceof",
        "this", "super", "static", "get", "set", "break", "continue", "switch", "case",
        "throw", "delete", "in", "of", "void", "yield", "debugger", "with",
        
        # Built-ins
        "document", "window", "console", "log", "JSON", "parse", "stringify",
        "Array", "Object", "Map", "Set", "Date", "Math", "String", "Number", 
        "Boolean", "RegExp", "Error", "Promise", "Symbol", "BigInt",
        
        # Array methods
        "isArray", "forEach", "map", "filter", "reduce", "reduceRight", "find", 
        "findIndex", "indexOf", "lastIndexOf", "includes", "some", "every",
        "push", "pop", "shift", "unshift", "slice", "splice", "concat", "join",
        "reverse", "sort", "fill", "flat", "flatMap", "from", "keys", "values", 
        "entries", "length",
        
        # Object methods
        "assign", "create", "defineProperty", "freeze", "seal", "hasOwnProperty",
        "toString", "valueOf", "toJSON",
        
        # String methods
        "charAt", "charCodeAt", "substring", "substr", "trim", "split", "replace",
        "replaceAll", "match", "search", "toLowerCase", "toUpperCase", "startsWith",
        "endsWith", "padStart", "padEnd", "repeat",
        
        # Math
        "abs", "ceil", "floor", "round", "max", "min", "random", "sqrt", "pow",
        
        # Async
        "then", "catch", "finally", "resolve", "reject", "all", "race", "allSettled"
    ]    

    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it",
        "not", "on", "with", "he", "as", "you", "at", "this", "but", "his",
        "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
        "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
        "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
        "time", "no", "just", "him", "know", "take", "people", "into", "year",
        "your", "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also", "back",
        "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "want", "because", "any", "these", "give", "day", "most", "us",
        "find", "more", "may", "very", "should", "each", "where", "between",
        "without", "through", "same", "another", "such", "own", "under", "does"
    ]
    
    punctuation = [
        ".", ",", ":", ";", "!", "?", "'", '"', "(", ")", "[", "]", "{", "}",
        "<", ">", "=", "+", "-", "*", "/", "%", "&", "|", "^", "~", "_",
        "==", "===", "!=", "!==", "<=", ">=", "&&", "||", "++", "--", "=>",
        "...", "//", "/*", "*/"
    ]
    
    all_base_words = js_keywords + common_words + punctuation
    original_size = len(token_ids)
    
    for word in all_base_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        token_ids.update(tokens)
        tokens_with_space = tokenizer.encode(" " + word, add_special_tokens=False)
        token_ids.update(tokens_with_space)
        tokens_trailing = tokenizer.encode(word + " ", add_special_tokens=False)
        token_ids.update(tokens_trailing)
    
    for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        tokens = tokenizer.encode(char, add_special_tokens=False)
        token_ids.update(tokens)
    
    added_count = len(token_ids) - original_size
    print(f"Added {added_count} base vocabulary tokens (total now: {len(token_ids)})")
    
    return token_ids


def add_special_tokens(token_ids: Set[int], tokenizer) -> Set[int]:
    """Add essential control/special tokens."""
    special_tokens = {
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    }
    
    special_tokens = {tid for tid in special_tokens if tid is not None}
    
    print(f"Adding {len(special_tokens)} special tokens: {special_tokens}")
    token_ids.update(special_tokens)
    
    return token_ids


def add_low_token_ids(token_ids: Set[int], threshold: int) -> Set[int]:
    """Add all token IDs below a specified threshold."""
    if threshold <= 0:
        return token_ids
    
    original_size = len(token_ids)
    
    print(f"\nAdding all tokens with ID < {threshold}...")
    low_tokens = set(range(threshold))
    token_ids.update(low_tokens)
    
    added_count = len(token_ids) - original_size
    print(f"Added {added_count} low-ID tokens (total now: {len(token_ids)})")
    
    return token_ids


def save_keep_list(token_ids: Set[int], output_path: str):
    """Save sorted token IDs as binary file (uint32 array)."""
    sorted_tokens = sorted(token_ids)
    
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', len(sorted_tokens)))
        for token_id in sorted_tokens:
            f.write(struct.pack('I', token_id))
    
    print(f"Saved {len(sorted_tokens)} tokens to {output_path}")


def print_statistics(token_ids: Set[int], tokenizer):
    """Print vocabulary statistics."""
    print("\n" + "="*60)
    print("VOCABULARY STATISTICS")
    print("="*60)
    print(f"Original vocabulary size: {tokenizer.vocab_size}")
    print(f"Kept tokens: {len(token_ids)}")
    print(f"Reduction: {(1 - len(token_ids)/tokenizer.vocab_size)*100:.1f}%")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate vocabulary keep list from conversation logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with low token threshold
  python generate_keep_list.py -i conv.json -t 10000
  
  # Minimal vocabulary from conversation only
  python generate_keep_list.py -i conv.json
        """
    )
    
    parser.add_argument('--input', '-i', default=None, help='Path to conversation JSON file')
    parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-Coder-14B-Instruct', 
                        help='HuggingFace model ID for tokenizer')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory for binary files')
    parser.add_argument('--low-token-threshold', '-t', type=int, default=0,
                        help='Include all token IDs below this threshold')
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")

    token_ids = set()

    if args.input:
        print(f"\nLoading conversation from {args.input}...")
        texts = load_conversation(args.input)
        print("\nTokenizing all messages...")
        token_ids = tokenize_and_collect(texts, tokenizer)

    print("\nAdding base vocabulary...")
    token_ids = add_base_vocabulary(token_ids, tokenizer)
    
    print("\nAdding special tokens...")
    token_ids = add_special_tokens(token_ids, tokenizer)
    
    if args.low_token_threshold > 0:
        token_ids = add_low_token_ids(token_ids, args.low_token_threshold)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    keep_list_path = output_dir / "keep_list.bin"
    
    print("\nSaving binary file...")
    save_keep_list(token_ids, str(keep_list_path))
    
    print_statistics(token_ids, tokenizer)
    
    print(f"\n✅ Done! Generated: {keep_list_path}")


if __name__ == "__main__":
    main()