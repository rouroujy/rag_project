from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
from src.minirag_logging_config import minirag_setup_logging
import asyncio
import os
import pathlib
import argparse

BASE_MODEL_DIR = "/mnt/d/ai_models"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(BASE_MODEL_DIR, "hf_cache")

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type = str,
        default="data/docs.txt",
        help="要加载的文档路径"
    )
    return parser.parse_args()

# 新增：支持PDF Markdown
def load_document(file_path:pathlib.Path):
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".txt" or suffix == ".md":
            with open(file_path,"r",encoding="utf-8") as f:
                return f.read()
        elif suffix == ".pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        else:
            logger.error("不支持的文件格式")
    except Exception as e:
        logger.error(f"文件读取出错：{e}")


# 新增chunk函数
def chunk_text(text:str, chunk_size: int = 200, overlap: int = 50):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

async def main():
    # 1.加载模型
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer(
        EMBED_MODEL_NAME,
        cache_folder=CACHE_DIR,
        # 第一次下载模型后，设置离线模式，不用每次都走代理检查最新版本
        local_files_only=True
        )
    logger.info("加载模型成功")


    # 使用命令行参数来定义读取文件路径
    args = parse_args()
    data_path = pathlib.Path(args.file)

    # 2.文档读取（进阶版：支持chunk overlap）
    # data_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "docs.txt"
    try:
        with open(data_path,"r",encoding="utf-8") as f:
            # full_text = f.read()
            full_text = load_document(data_path)
        logger.info(f"文档加载成功：{data_path}")
        chunk_size = 500
        overlap = 50
        docs = chunk_text(full_text,chunk_size=chunk_size,overlap=overlap)
        if not docs:
            logger.error("文档为空，程序终止")
            return
        logger.info(f"文档切分完成，共{len(docs)}个chunks")
    except Exception as e:
        logger.exception(f"文档读取失败：{data_path}, 错误信息：{e}")
        return


    # 2.文档读取（基础版：一行为一个文档）
    # data_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "docs.txt"
    # try:
    #     with open(data_path,"r",encoding = 'utf-8') as f:
    #         docs = [line.strip() for line in f if line.strip()]
    #     logger.info(f"文档加载成功：{data_path}，共{len(docs)}条")
    #     if not docs:
    #         logger.warning("文档文件为空，请检查内容")
    # except Exception as e:
    #     logger.exception(f"文档读取失败：{data_path}，错误信息{e}")

    # 2.文档硬编写
    # docs = [
    #     "人工智能是一门研究智能行为的科学。",
    #     "RAG是检索增强生成的简称。",
    #     "FastAPI是一个现代Python Web框架。"
    # ]
    # logger.info("文档输入成功")

    try:
        # 3.向量化
        embedings = model.encode(docs)
        # 4.建立向量索引
        dimension = embedings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embedings).astype("float32"))
        logger.info("向量化、建立向量索引成功！")
    except Exception as e:
        logger.error(f"出现异常：{e}")
    

    # 5.查询
    logger.info("开始查询问题")
    query = "关于AI Agent工程师需要具备什么条件"
    query_vector = model.encode([query])
    k = 5
    D,I = index.search(np.array(query_vector).astype("float32"), k = k)
    logger.info("查询结束")
    print(f"\n===== k={k} 实验结果 =====")
    for idx in I[0]:
        print("候选文档：", docs[idx][:500])

    # for idx in I[0]:
    #     print("候选文档：", docs[idx])
    # print("最相关文档：", docs[I[0][0]])
     

if __name__=="__main__":
     minirag_setup_logging()
     logger.info(f"开始运行mini_rag：{__name__}")
     asyncio.run(main())