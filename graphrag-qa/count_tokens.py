import json
import transformers
import os

tokenizer_dir = os.environ.get('TOKENIZER_DIR', 'deepseek_v3_tokenizer')
print('Loading tokenizer...')
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

def count_file_tokens(filename):
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_tokens = 0
    for row in data:
        parts = [row.get('question', ''), row.get('answer', '')]
        parts.extend(row.get('contexts', []))
        text_to_tokenize = '\n'.join(parts)
        
        tokens = tokenizer.encode(text_to_tokenize)
        total_tokens += len(tokens)
    return total_tokens

print('Counting tokens in datasets...')
naive_tokens = count_file_tokens('eval/checkpoint_naive.json')
graph_tokens = count_file_tokens('eval/checkpoint_graph.json')
agentic_tokens = count_file_tokens('eval/checkpoint_agentic.json')

total = naive_tokens + graph_tokens + agentic_tokens

print(f'Naive RAG Dataset: {naive_tokens:,} tokens')
print(f'GraphRAG Dataset: {graph_tokens:,} tokens')
print(f'Agentic RAG Dataset: {agentic_tokens:,} tokens')
print(f'---')
print(f'Total context size fed to Evaluator: {total:,} tokens')
