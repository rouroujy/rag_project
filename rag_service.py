import json
import os
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.service.langchain_llm_adapter import DashScopeLLM

BASE_MODEL_DIR = "/mnt/d/ai_models"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(BASE_MODEL_DIR,"hf_cache")

class RAGService:
    def __init__(self, documents: list[Document]):
        print("开始加载embedding...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="/mnt/d/ai_models/hf_cache"
        )
        # embedding_model = HuggingFaceEmbeddings(
        #     model_name = EMBED_MODEL_NAME,
        #     cache_folder = CACHE_DIR,
        #     # 第一次下载模型后，设置离线模式，不用每次都走代理检查最新版本
        #     model_kwargs={
        #         "local_files_only": True
        #     }
        # )
        print("embedding加载完成")

        self.vectorstore = FAISS.from_documents(
            documents,
            embedding_model
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type = 'mmr',
            search_kwargs = {"k":3,"fetch_k":10}
        )

        self.llm = DashScopeLLM()

        self.prompt = ChatPromptTemplate.from_template(
            """
你是一个严谨的企业级问答助手。
仅根据上下文回答。
必须输出JSON格式：
{{
    "answer":"...",
    "sources":["...", "..."]
}}
上下文：
{context}
问题：
{question}
"""
        )
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n".join(
            f"[{i}] {doc.page_content}" 
            for i, doc in enumerate(docs)
        )
    
    def query(self, question: str):
        result = self.chain.invoke(question)

        try:
            return json.loads(result)
        except:
            return{
                "answer":result,
                "sources":[]
            }