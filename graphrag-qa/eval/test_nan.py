
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', google_api_key='BAD_KEY'))
emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key='BAD_KEY'))
ds = Dataset.from_dict({'question': ['test'], 'contexts': [['test']], 'answer': ['test'], 'reference': ['test']})
res = evaluate(ds, metrics=[context_precision], llm=llm, embeddings=emb)
df = res.to_pandas()
print('Value:', df['context_precision'].iloc[0])

