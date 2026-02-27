# RAG Project

一个基于 SentenceTransformers + FAISS + DashScope(Qwen) + FastAPI + Docker 构建的企业级 RAG（Retrieval-Augmented Generation）系统。

## 环境
Python 3.10.12
WSL2 Ubuntu

## 支持

多格式文档加载（txt / md / pdf）
文本分块 + overlap
向量检索（FAISS）
百炼大模型（Qwen）生成
FastAPI API 服务化
Docker 镜像化部署

---

## 项目结构

```
rag_project
├── Dockerfile
├── rag_api.py              # FastAPI 入口
├── rag_service.py          # RAG 服务封装
├── requirements.txt
├── data/                   # 测试数据
├── logs/                   # 日志
└── src/
    ├── mini_rag.py         # 原生版 RAG（手写 FAISS）
    ├── langchain_rag.py    # LangChain 版 RAG
    ├── config.py           # 配置管理
    ├── minirag_logging_config.py
    └── service/
        ├── dashscope_client.py
        └── langchain_llm_adapter.py
```

---

## 系统架构

```
                ┌─────────────────────┐
                │     Client (HTTP)    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     FastAPI         │
                │    rag_api.py       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     RAGService      │
                │   (核心调度模块)     │
                └──────┬──────┬───────┘
                       │      │
                       ▼      ▼
              ┌────────────┐  ┌──────────────┐
              │  Retriever │  │     LLM       │
              │  (FAISS)   │  │ DashScope(Qwen)│
              └──────┬─────┘  └──────┬───────┘
                     │               │
                     ▼               ▼
            Embedding Model     Prompt + 生成
     (all-MiniLM-L6-v2)
```

---

## RAG Pipeline（完整数据流）

### 1. 文档预处理阶段

```
加载文档
  ↓
chunk 切分（带 overlap）
  ↓
生成 embeddings
  ↓
构建 FAISS 向量索引
```

关键实现：

* Embedding: `all-MiniLM-L6-v2`
* 向量库: FAISS
* 检索策略: MMR

---

### 2. 查询阶段（在线）

```
用户问题
   ↓
Embedding 编码
   ↓
向量检索 Top-K
   ↓
构造 Prompt
   ↓
DashScope LLM 生成
   ↓
返回 JSON 结果
```

在 `rag_service.py` 中构建 LCEL Pipeline：

```python
self.chain = (
    {
        "context": self.retriever | self._format_docs,
        "question": RunnablePassthrough()
    }
    | self.prompt
    | self.llm
    | StrOutputParser()
)
```

---

## 模块划分说明

---

### 1. mini_rag.py（原生实现）

手写版本流程：

* SentenceTransformer 向量化
* FAISS 手动构建
* 手写 prompt
* 手动调用 DashScope

👉 用于理解底层原理

---

### 2. langchain_rag.py（框架版）

使用：

* HuggingFaceEmbeddings
* FAISS (LangChain封装)
* Retriever
* LCEL Runnable Pipeline

👉 更工程化、可扩展

---

### 3.  rag_service.py（服务层抽象）

这是最终 API 用的核心模块：

* 初始化 embedding
* 构建 vectorstore
* 构建 retriever
* 构建 LLM
* 构建 prompt
* 封装 chain

对外暴露：

```python
def query(self, question: str)
```

---

### 4.  rag_api.py（接口层）

FastAPI 服务：

```
POST /query
```

输入：

```json
{
  "question": "什么是AI Agent？"
}
```

输出：

```json
{
  "answer": "...",
  "sources": ["..."]
}
```

---

### 5.  config.py（配置管理）

* 使用 Pydantic Settings
* 支持 dev / prod
* 校验 DashScope API Key

---

## 本地运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

```
DASHSCOPE_API_KEY=your_key
```

### 3. 启动 API

```bash
uvicorn rag_api:app --reload
```

访问：

```
http://localhost:8000/docs
```

---

## Docker 部署

### 1. 构建镜像

```bash
docker build -t rag-service .
```

### 2. 运行容器

```bash
docker run -it -p 8000:8000 \
  -e DASHSCOPE_API_KEY=your_key \
  rag-service
```

访问：

```
http://localhost:8000/docs
```

---

## 技术栈总结

| 模块        | 技术               |
| --------- | ---------------- |
| Embedding | all-MiniLM-L6-v2 |
| 向量库       | FAISS            |
| 框架        | LangChain        |
| LLM       | DashScope (Qwen) |
| API       | FastAPI          |
| 容器化       | Docker           |

---

## 可优化方向

---

### 1. 检索优化

* 调整 chunk_size
* 动态 overlap
* Hybrid Search（BM25 + 向量）
* Rerank 模型

---

### 2. 向量存储优化

当前使用：

```
IndexFlatL2（全量扫描）
```

可升级：

* IVF
* HNSW
* Milvus / Qdrant / Weaviate

---

### 3.  Prompt 优化

* 增加 citation 模板
* 增加 system 指令控制
* 加入防 hallucination 规则

---

### 4. 性能优化

* 预构建向量库（持久化）
* embedding 批量化
* 异步调用 LLM
* 缓存 query 结果

---

### 5. 工程化升级

* 支持多文档上传
* 分库管理
* 多租户隔离
* 日志埋点
* OpenTelemetry 监控

---

## 本项目 RAG 架构总结

这是一个标准的：单机向量库 + 云端大模型 的 RAG 架构

特点：

* Embedding 本地
* VectorStore 本地
* LLM 云端
* API 服务化
* Docker 可部署

---

