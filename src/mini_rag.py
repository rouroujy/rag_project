from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
from src.minirag_logging_config import minirag_setup_logging
import asyncio
import os
import pathlib

BASE_MODEL_DIR = "/mnt/d/ai_models"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(BASE_MODEL_DIR, "hf_cache")


logger = logging.getLogger(__name__)

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

    # 2.文档读取
    data_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "docs.txt"
    try:
        with open(data_path,"r",encoding = 'utf-8') as f:
            docs = [line.strip() for line in f if line.strip()]
        logger.info(f"文档加载成功：{data_path}，共{len(docs)}条")
        if not docs:
            logger.warning("文档文件为空，请检查内容")
    except Exception as e:
        logger.exception(f"文档读取失败：{data_path}，错误信息{e}")

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
    query = "什么是RAG"
    query_vector = model.encode([query])
    D,I = index.search(np.array(query_vector).astype("float32"), k =2)
    logger.info("查询结束")
    for idx in I[0]:
        print("候选文档：", docs[idx])
    # print("最相关文档：", docs[I[0][0]])
     

if __name__=="__main__":
     minirag_setup_logging()
     logger.info(f"开始运行mini_rag：{__name__}")
     asyncio.run(main())