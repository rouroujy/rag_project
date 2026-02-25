from fastapi import FastAPI
from pydantic import BaseModel
import pathlib

from langchain_core.documents import Document
from src.langchain_rag import load_document, chunk_text
from rag_service import RAGService

app = FastAPI()
rag_service = None

@app.on_event("startup")
def startup():
    global rag_service
    # 初始化RAG
    file_path = pathlib.Path("data/chunk_test.md")
    text = load_document(file_path)
    chunks = chunk_text(text)
    documents = [Document(page_content=c) for c in chunks]

    rag_service = RAGService(documents)
    print("=========RAG初始化完成！=========")


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query_rag(req: QueryRequest):
    result = rag_service.query(req.question)
    return result