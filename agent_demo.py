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

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

load_dotenv(r"D:\python\.env")

# ========== 1. 先搭好 RAG 的检索部分（复用你之前的） ==========
loader = PyPDFLoader("RAG_demo.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# ========== 2. 把检索包装成一个工具 ==========
@tool
def search_pdf(query: str) -> str:
    """当用户问关于 RAG、文档内容的问题时，用这个工具检索 PDF。"""
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in results)

# ========== 3. 再加一个天气工具（假数据） ==========
@tool
def get_weather(city: str) -> str:
    """当用户问某个城市的天气时，用这个工具。"""
    fake_weather = {
        "北京": "晴，25°C，微风",
        "上海": "多云，28°C，湿度 70%",
        "杭州": "小雨，22°C，东南风 3 级",
    }
    return fake_weather.get(city, f"暂时查不到 {city} 的天气")

# ========== 4. 把两个工具交给模型 ==========
tools = [search_pdf, get_weather]

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)

# 用 LangGraph 的 create_react_agent 创建一个会自主调工具的 Agent
agent = create_react_agent(llm, tools)

# ========== 5. 循环提问 ==========
while True:
    question = input("\n请输入你的问题（输入 q 退出）：")
    if question.lower() == "q":
        break
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    # 取最后一条消息作为回答
    answer = result["messages"][-1].content
    print("\n===== 回答 =====")
    print(answer)