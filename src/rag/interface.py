"""
测试用例生成器的RAG（检索增强生成）接口。

本模块提供可插拔的RAG功能接口。
实际实现可以使用各种向量存储（Chroma、Pinecone等）和嵌入模型。

目前提供用于未来实现的骨架/接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RAGConfig:
    """
    RAG功能的配置。
    
    属性:
        enabled: 是否启用RAG
        collection_name: 向量存储集合名称
        embedding_model: 用于嵌入的模型
        embedding_api_key: 嵌入模型的API密钥
        embedding_base_url: 嵌入API的基础URL
        vector_store_type: 向量存储类型（chroma、pinecone、faiss等）
        vector_store_config: 额外的向量存储配置
        top_k: 要检索的文档数量
        similarity_threshold: 检索的最小相似度分数
    """
    enabled: bool = False
    collection_name: str = "test_case_knowledge"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    vector_store_type: str = "chroma"
    vector_store_config: dict = field(default_factory=dict)
    top_k: int = 5
    similarity_threshold: float = 0.7


@dataclass
class RetrievedDocument:
    """
    从向量存储检索的文档。
    
    属性:
        content: 文档内容
        metadata: 文档元数据
        score: 相似度分数
        source: 来源标识
    """
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    source: str = ""
    
    def __str__(self) -> str:
        """文档的字符串表示。"""
        source_info = f" (来源: {self.source})" if self.source else ""
        return f"{self.content}{source_info}"


class BaseVectorStore(ABC):
    """
    向量存储实现的抽象基类。
    
    实现此类以添加对不同向量存储的支持。
    """
    
    @abstractmethod
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None
    ) -> list[str]:
        """
        向向量存储添加文档。
        
        参数:
            documents: 文档内容列表
            metadatas: 可选的元数据字典列表
            ids: 可选的文档ID列表
            
        返回:
            文档ID列表
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> list[RetrievedDocument]:
        """
        搜索相似文档。
        
        参数:
            query: 搜索查询
            top_k: 返回的结果数量
            filter_dict: 可选的元数据过滤器
            
        返回:
            检索到的文档列表
        """
        pass
    
    @abstractmethod
    def delete(self, ids: list[str]) -> bool:
        """
        按ID删除文档。
        
        参数:
            ids: 要删除的文档ID列表
            
        返回:
            成功则返回True
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        清除存储中的所有文档。
        
        返回:
            成功则返回True
        """
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma向量存储实现。
    
    注意：这是一个占位实现。需要时安装chromadb并实现实际功能。
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._client = None
        self._collection = None
        self._embeddings = None
    
    def _initialize(self):
        """初始化Chroma客户端和集合。"""
        try:
            import chromadb
            from langchain_openai import OpenAIEmbeddings
            
            # 初始化嵌入
            self._embeddings = OpenAIEmbeddings(
                api_key=self.config.embedding_api_key,
                base_url=self.config.embedding_base_url,
                model=self.config.embedding_model
            )
            
            # 初始化Chroma客户端
            persist_dir = self.config.vector_store_config.get('persist_directory')
            if persist_dir:
                self._client = chromadb.PersistentClient(path=persist_dir)
            else:
                self._client = chromadb.Client()
            
            # 获取或创建集合
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "测试用例知识库"}
            )
            
        except ImportError:
            raise ImportError(
                "Chroma向量存储需要chromadb。"
                "使用以下命令安装: pip install chromadb"
            )
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None
    ) -> list[str]:
        if self._collection is None:
            self._initialize()
        
        # 如果未提供则生成ID
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # 生成嵌入
        embeddings = self._embeddings.embed_documents(documents)
        
        # 添加到集合
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in documents],
            ids=ids
        )
        
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> list[RetrievedDocument]:
        if self._collection is None:
            self._initialize()
        
        # 生成查询嵌入
        query_embedding = self._embeddings.embed_query(query)
        
        # 搜索
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # 转换为RetrievedDocument对象
        documents = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            score = 1.0 - results['distances'][0][i] if results['distances'] else 0.0
            
            documents.append(RetrievedDocument(
                content=doc,
                metadata=metadata,
                score=score,
                source=metadata.get('source', '')
            ))
        
        return documents
    
    def delete(self, ids: list[str]) -> bool:
        if self._collection is None:
            self._initialize()
        
        self._collection.delete(ids=ids)
        return True
    
    def clear(self) -> bool:
        if self._client is None:
            self._initialize()
        
        self._client.delete_collection(self.config.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name
        )
        return True


class InMemoryVectorStore(BaseVectorStore):
    """
    简单的内存向量存储，用于测试/开发。
    
    使用基本的余弦相似度，无需外部依赖。
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._documents: dict[str, dict] = {}
        self._embeddings = None
    
    def _initialize(self):
        """初始化嵌入模型。"""
        try:
            from langchain_openai import OpenAIEmbeddings
            
            self._embeddings = OpenAIEmbeddings(
                api_key=self.config.embedding_api_key,
                base_url=self.config.embedding_base_url,
                model=self.config.embedding_model
            )
        except ImportError:
            # 回退到无嵌入（仅文本匹配）
            self._embeddings = None
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算两个向量之间的余弦相似度。"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None
    ) -> list[str]:
        if self._embeddings is None:
            self._initialize()
        
        # 如果未提供则生成ID
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # 如果可用则生成嵌入
        embeddings = None
        if self._embeddings:
            embeddings = self._embeddings.embed_documents(documents)
        
        # 存储文档
        for i, doc in enumerate(documents):
            self._documents[ids[i]] = {
                'content': doc,
                'metadata': metadatas[i] if metadatas else {},
                'embedding': embeddings[i] if embeddings else None
            }
        
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> list[RetrievedDocument]:
        if not self._documents:
            return []
        
        if self._embeddings is None:
            self._initialize()
        
        results = []
        
        if self._embeddings:
            # 向量相似度搜索
            query_embedding = self._embeddings.embed_query(query)
            
            for doc_id, doc_data in self._documents.items():
                # 应用元数据过滤
                if filter_dict:
                    match = all(
                        doc_data['metadata'].get(k) == v
                        for k, v in filter_dict.items()
                    )
                    if not match:
                        continue
                
                if doc_data['embedding']:
                    score = self._cosine_similarity(
                        query_embedding,
                        doc_data['embedding']
                    )
                else:
                    score = 0.0
                
                results.append(RetrievedDocument(
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    score=score,
                    source=doc_data['metadata'].get('source', doc_id)
                ))
        else:
            # 简单文本匹配回退
            query_lower = query.lower()
            for doc_id, doc_data in self._documents.items():
                if filter_dict:
                    match = all(
                        doc_data['metadata'].get(k) == v
                        for k, v in filter_dict.items()
                    )
                    if not match:
                        continue
                
                # 简单关键词匹配分数
                content_lower = doc_data['content'].lower()
                words = query_lower.split()
                matched = sum(1 for w in words if w in content_lower)
                score = matched / len(words) if words else 0.0
                
                results.append(RetrievedDocument(
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    score=score,
                    source=doc_data['metadata'].get('source', doc_id)
                ))
        
        # 按分数排序并返回top_k
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 按阈值过滤
        results = [r for r in results if r.score >= self.config.similarity_threshold]
        
        return results[:top_k]
    
    def delete(self, ids: list[str]) -> bool:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
        return True
    
    def clear(self) -> bool:
        self._documents.clear()
        return True


class RAGInterface:
    """
    测试用例生成器的主RAG接口。
    
    提供统一接口用于：
    - 向知识库添加文档
    - 检索相关文档
    - 管理向量存储
    
    使用方式:
        # 使用配置初始化
        config = RAGConfig(enabled=True, embedding_api_key="sk-...")
        rag = RAGInterface(config)
        
        # 添加文档
        rag.add_documents([
            "登录测试用例需求...",
            "支付流程测试指南..."
        ])
        
        # 检索相关文档
        docs = rag.retrieve("用户认证测试")
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化RAG接口。
        
        参数:
            config: RAG配置，未提供时使用默认值
        """
        self.config = config or RAGConfig()
        self._vector_store: Optional[BaseVectorStore] = None
    
    def _get_vector_store(self) -> BaseVectorStore:
        """获取或创建向量存储实例。"""
        if self._vector_store is None:
            store_type = self.config.vector_store_type.lower()
            
            if store_type == "chroma":
                self._vector_store = ChromaVectorStore(self.config)
            elif store_type == "memory":
                self._vector_store = InMemoryVectorStore(self.config)
            else:
                # 默认使用内存存储
                self._vector_store = InMemoryVectorStore(self.config)
        
        return self._vector_store
    
    def is_enabled(self) -> bool:
        """检查RAG是否启用。"""
        return self.config.enabled
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        source: Optional[str] = None
    ) -> list[str]:
        """
        向知识库添加文档。
        
        参数:
            documents: 文档内容列表
            metadatas: 每个文档的可选元数据
            source: 要添加到所有文档的可选来源标识
            
        返回:
            文档ID列表
        """
        if not self.is_enabled():
            return []
        
        # 如果提供了来源则添加到元数据
        if source and metadatas is None:
            metadatas = [{"source": source} for _ in documents]
        elif source and metadatas:
            for m in metadatas:
                m["source"] = source
        
        store = self._get_vector_store()
        return store.add_documents(documents, metadatas)
    
    def add_from_file(self, file_path: str) -> list[str]:
        """
        从文件添加文档。
        
        参数:
            file_path: 文件路径
            
        返回:
            文档ID列表
        """
        if not self.is_enabled():
            return []
        
        from src.input_handler.handlers import InputHandler
        
        handler = InputHandler()
        processed = handler.process_file(file_path)
        
        # 将内容分割为块（简单方法）
        content = processed.text_content
        chunks = self._chunk_text(content)
        
        metadatas = [{"source": file_path, "chunk": i} for i in range(len(chunks))]
        
        return self.add_documents(chunks, metadatas)
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> list[str]:
        """
        将文本分割为带重叠的块。
        
        参数:
            text: 要分块的文本
            chunk_size: 目标块大小（字符数）
            overlap: 块之间的重叠
            
        返回:
            文本块列表
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句子边界断开
            if end < len(text):
                # 查找句子结尾
                for sep in ['. ', '.\n', '\n\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + len(sep)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return chunks
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[dict] = None
    ) -> list[str]:
        """
        为查询检索相关文档。
        
        参数:
            query: 搜索查询
            top_k: 结果数量（未指定时使用配置默认值）
            filter_dict: 可选的元数据过滤器
            
        返回:
            文档内容列表
        """
        if not self.is_enabled():
            return []
        
        store = self._get_vector_store()
        docs = store.search(
            query,
            top_k=top_k or self.config.top_k,
            filter_dict=filter_dict
        )
        
        return [doc.content for doc in docs]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[dict] = None
    ) -> list[RetrievedDocument]:
        """
        检索带相似度分数的文档。
        
        参数:
            query: 搜索查询
            top_k: 结果数量
            filter_dict: 可选的元数据过滤器
            
        返回:
            RetrievedDocument对象列表
        """
        if not self.is_enabled():
            return []
        
        store = self._get_vector_store()
        return store.search(
            query,
            top_k=top_k or self.config.top_k,
            filter_dict=filter_dict
        )
    
    def delete_documents(self, ids: list[str]) -> bool:
        """
        按ID删除文档。
        
        参数:
            ids: 文档ID列表
            
        返回:
            成功则返回True
        """
        if not self.is_enabled():
            return False
        
        store = self._get_vector_store()
        return store.delete(ids)
    
    def clear(self) -> bool:
        """
        清除知识库中的所有文档。
        
        返回:
            成功则返回True
        """
        if not self.is_enabled():
            return False
        
        store = self._get_vector_store()
        return store.clear()
    
    def get_stats(self) -> dict:
        """
        获取知识库的统计信息。
        
        返回:
            包含统计信息的字典
        """
        if not self.is_enabled():
            return {"enabled": False}
        
        store = self._get_vector_store()
        
        # 获取文档数量（实现取决于存储）
        if isinstance(store, InMemoryVectorStore):
            doc_count = len(store._documents)
        else:
            doc_count = -1  # 未知
        
        return {
            "enabled": True,
            "vector_store_type": self.config.vector_store_type,
            "collection_name": self.config.collection_name,
            "document_count": doc_count,
            "embedding_model": self.config.embedding_model
        }
