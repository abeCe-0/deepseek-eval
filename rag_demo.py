from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

load_dotenv(r"D:\python\.env")

loader = PyPDFLoader("RAG_demo.pdf")
docs = loader.load()
print(f"PDF 共 {len(docs)} 页")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"切成 {len(chunks)} 块")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
print("向量库建好了")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
根据以下文档内容回答问题。如果文档里没有相关信息，就说"文档里没有提到"。

文档内容：
{context}

问题：{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    question = input("\n请输入你的问题（输入 q 退出）：")
    if question.lower() == "q":
        break
    answer = chain.invoke(question)
    print("\n===== 回答 =====")
    print(answer)