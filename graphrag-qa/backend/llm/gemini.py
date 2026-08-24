from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import time
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash-lite"
# Free tier: 500 RPD / 15 RPM / 250K TPM. Paid (Tier 1) is much higher on RPM/RPD —
# see graphrag-qa/README.md for notes on enabling billing for eval runs.

# Used for final answer generation — some creative variation in phrasing is fine here.
model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.3,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Used for structured/classification calls (agent planner + critic in llm/agent.py)
# where we want deterministic, repeatable decisions rather than creative variation.
# temperature=0.3 on a True/False "is context sufficient?" classifier would make the
# agent's iteration count — and therefore the benchmark numbers — vary run to run.
structured_model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

SYSTEM_PROMPT = """You are an expert code assistant. Answer questions using ONLY the provided code chunks.
1. End every response with a ## Citations section.
2. Format citations strictly as: `filepath:startline-endline`
"""

def stream_answer(question: str, context: str):
    """Streams the answer chunk by chunk — used by the live /query endpoint."""
    prompt = f"{SYSTEM_PROMPT}\n\n## Code Context\n{context}\n\n## Question\n{question}\n"
    for chunk in model.stream([HumanMessage(content=prompt)]):
        content = chunk.content
        if content:
            if isinstance(content, list):
                yield "".join(c.get("text", "") for c in content if isinstance(c, dict))
            else:
                yield str(content)

def safe_stream_answer(question: str, context: str, retries: int = 3) -> str:
    """stream_answer wrapped with exponential backoff, collected into one string.

    Shared by every pipeline in the benchmark harness (naive, graph, AND agentic)
    so a single transient error doesn't silently poison one row's answer with
    "Error: ..." while the other two pipelines retry and succeed — which would
    otherwise make the pipeline comparison unfair.
    """
    for attempt in range(retries):
        try:
            return "".join(stream_answer(question, context))
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(2 ** attempt)