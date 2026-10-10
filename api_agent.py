from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import requests

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv(r"D:\python\.env")

app = FastAPI()

# ========== 启动时只建一次向量库和 Agent ==========
agent = None

@app.on_event("startup")
def load_agent():
    global agent
    loader = PyPDFLoader("RAG_demo.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    @tool
    def search_pdf(query: str) -> str:
        """当用户问关于 RAG、文档内容的问题时，用这个工具检索 PDF。"""
        results = vectorstore.similarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in results)

    @tool
    def get_weather(city: str) -> str:
        """当用户问某个城市的天气时，用这个工具。"""
        url = f"https://wttr.in/{city}?format=3&lang=zh"
        try:
            resp = requests.get(url, timeout=5)
            return resp.text.strip() if resp.status_code == 200 else f"查询失败：{resp.status_code}"
        except Exception as e:
            return f"查天气出错：{e}"

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0
    )
    agent = create_react_agent(llm, [search_pdf, get_weather])

# ========== 接口 ==========
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = agent.invoke({"messages": [HumanMessage(content=req.question)]})
    answer = result["messages"][-1].content
    return AskResponse(answer=answer)