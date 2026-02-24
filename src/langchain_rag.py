import os
import argparse
import pathlib
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.minirag_logging_config import minirag_setup_logging
from src.service.langchain_llm_adapter import DashScopeLLM

logger = logging.getLogger(__name__)

BASE_MODEL_DIR = "/mnt/d/ai_models"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(BASE_MODEL_DIR,"hf_cache")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type = str,
        default = "data/docs.txt",
        help = "要加载的文档路径"
    )
    return parser.parse_args()

def load_document(file_path:pathlib.Path):
    suffix = file_path.suffix.lower()

    if suffix in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        
    elif suffix == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        raise ValueError("不支持的文件格式")
    
def chunk_text(text:str,chunk_size:int = 300,overlap:int=50):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def main():
    args = parse_args()
    data_path = pathlib.Path(args.file)

    # 1.读取文档
    full_text = load_document(data_path)
    logger.info(f"文档加载成功：{data_path}")

    docs = chunk_text(full_text)
    logger.info(f"文档切分完成，共{len(docs)}个chunks")

    # 2.转成LangChain Document
    documents = [Document(page_content = doc) for doc in docs]

    # 3.Embedding模型
    embedding_model = HuggingFaceEmbeddings(
        model_name = EMBED_MODEL_NAME,
        cache_folder = CACHE_DIR
    )

    # 4.构建向量数据库（数据存储）
    vectorstore = FAISS.from_documents(
        documents,
        embedding_model
    )

    logger.info("向量库构建完成")

    # 5.构建Retriever：一个可调用的搜索器（数据查询接口）
    retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":3,"fetch_k":10}
        )

    # 6.构建 DashScope LLM
    llm = DashScopeLLM()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7.构造prompt
    prompt = ChatPromptTemplate.from_template(
        """
        你是一个严谨的问答助手。
        仅根据提供的上下文回答问题。
        若上下文未提及，请回答“无法从给定内容确定”。
        上下文：
        {context}
        问题：
        {question}
    """
    )

    # 8.构造RAG pipeline(LCEL方式)
    # RunnableParallel()：并行执行多个 Runnable，然后把结果合并成一个 dict，即一个输入多个输出
    # | 是串行（有先后顺序）
    rag_chain = (
        {
            "context":retriever | format_docs,
            "question":RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


    # 9.查询
    query = "关于AI Agent工程师需要具备什么条件"

    result = rag_chain.invoke(query)

    print("\n=========LLM回答=========")
    print(result)

    # print("\n=========召回文档=========")
    # for doc in result["context"]:
    #     print(doc.page_content[:300])
    #     print("-"*50)


if __name__ == "__main__":
    minirag_setup_logging()
    logger.info("启动LangChain版RAG")
    main()