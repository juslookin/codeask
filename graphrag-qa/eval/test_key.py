
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load the key from the backend .env
load_dotenv('../backend/.env')

llm = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', google_api_key=os.getenv('GEMINI_API_KEY'))
try:
    res = llm.invoke([HumanMessage(content='Say the word SUCCESS if the API key works.')])
    print('\nAPI Key is working! Response:', res.content)
except Exception as e:
    print('\nAPI Key failed:', str(e))

