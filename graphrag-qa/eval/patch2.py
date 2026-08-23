
import os
p = 'benchmark.py'
with open(p) as f: lines = f.readlines()
# insert right after imports
insert_idx = 0
for i, l in enumerate(lines):
    if 'import pandas as pd' in l:
        insert_idx = i + 1
        break

patch = '''
# Monkey-patch to enforce global 4.2s rate limit (14 RPM max) across Agentic RAG & Ragas
from langchain_google_genai import ChatGoogleGenerativeAI
_orig_gen = ChatGoogleGenerativeAI._generate
_orig_agen = ChatGoogleGenerativeAI._agenerate
def _rl_gen(*args, **kwargs):
    import time
    time.sleep(4.2)
    return _orig_gen(*args, **kwargs)
async def _rl_agen(*args, **kwargs):
    import asyncio
    await asyncio.sleep(4.2)
    return await _orig_agen(*args, **kwargs)
ChatGoogleGenerativeAI._generate = _rl_gen
ChatGoogleGenerativeAI._agenerate = _rl_agen
'''
lines.insert(insert_idx, patch)
with open(p, 'w') as f: f.writelines(lines)

