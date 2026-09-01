import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

tokenizer = None
tokenizer_dir = os.environ.get('TOKENIZER_DIR', 'deepseek_v3_tokenizer')

try:
    import transformers
    if os.path.exists(tokenizer_dir):
        print(f"Loading local tokenizer from '{tokenizer_dir}'...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        print("Loading fallback tokenizer (gpt2)...")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
except Exception as e:
    print(f"Warning: Could not load HuggingFace tokenizer ({e}). Falling back to char approximation.")
    tokenizer = None


def count_text_tokens(text: str) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    # Fallback approximation: ~4 characters per token
    return max(1, len(text) // 4)


def count_file_tokens(filename: str) -> int:
    target = filename
    if not os.path.exists(target):
        target = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(target):
        return 0

    with open(target, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_tokens = 0

    # Case 1: HuggingFace Dataset dictionary format: {"user_input": [...], "response": [...], "retrieved_contexts": [...]}
    if isinstance(data, dict):
        user_inputs = data.get('user_input', [])
        responses = data.get('response', [])
        contexts_list = data.get('retrieved_contexts', [])
        max_len = max(len(user_inputs), len(responses), len(contexts_list))

        for i in range(max_len):
            q = user_inputs[i] if i < len(user_inputs) else ''
            a = responses[i] if i < len(responses) else ''
            ctxs = contexts_list[i] if i < len(contexts_list) else []

            parts = [q, a]
            if isinstance(ctxs, list):
                parts.extend(str(c) for c in ctxs)
            elif isinstance(ctxs, str):
                parts.append(ctxs)

            text_to_tokenize = '\n'.join(parts)
            total_tokens += count_text_tokens(text_to_tokenize)

    # Case 2: List of row objects format: [{"question": ..., "answer": ..., "contexts": [...]}, ...]
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            q = row.get('question', row.get('user_input', ''))
            a = row.get('answer', row.get('response', ''))
            ctxs = row.get('contexts', row.get('retrieved_contexts', []))

            parts = [str(q), str(a)]
            if isinstance(ctxs, list):
                parts.extend(str(c) for c in ctxs)
            elif isinstance(ctxs, str):
                parts.append(ctxs)

            text_to_tokenize = '\n'.join(parts)
            total_tokens += count_text_tokens(text_to_tokenize)

    return total_tokens


if __name__ == '__main__':
    print('Counting tokens in datasets...')
    naive_tokens = count_file_tokens('eval/checkpoint_naive.json')
    graph_tokens = count_file_tokens('eval/checkpoint_graph.json')
    agentic_tokens = count_file_tokens('eval/checkpoint_agentic.json')

    total = naive_tokens + graph_tokens + agentic_tokens

    print(f'Naive RAG Dataset: {naive_tokens:,} tokens')
    print(f'GraphRAG Dataset: {graph_tokens:,} tokens')
    print(f'Agentic RAG Dataset: {agentic_tokens:,} tokens')
    print('---')
    print(f'Total context size fed to Evaluator: {total:,} tokens')
