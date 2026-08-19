from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.3,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

SYSTEM_PROMPT = """You are an expert code assistant. Answer questions using ONLY the provided code chunks.
1. End every response with a ## Citations section.
2. Format citations strictly as: `filepath:startline-endline`
"""

def stream_answer(question: str, context: str):
    prompt = f"{SYSTEM_PROMPT}\n\n## Code Context\n{context}\n\n## Question\n{question}\n"
    for chunk in model.stream([HumanMessage(content=prompt)]):
        content = chunk.content
        if content:
            if isinstance(content, list):
                yield "".join(c.get("text", "") for c in content if isinstance(c, dict))
            else:
                yield str(content)
