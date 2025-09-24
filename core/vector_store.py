import os
import pickle
from typing import List, Optional
from langchain_chroma import Chroma  # 新的导入方式
from config.models import Chunk,SearchResult
from models.base import BaseEmbedding,BaseVectorStore
from utils.logger import get_logger
from utils.exceptions import SearchError

class ChromaVectorStore(BaseVectorStore):
    """
    Chroma向量存储实现
    """
    def __init__(self, embedding_model: BaseEmbedding,persist_directory: str, collection_name: str ="documents"):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = get_logger(__name__)
        self.chroma_db = None
        self.chunks_metadata = {}

    def add_chunks(self, chunks:List[Chunk])-> None:
        """
        添加chunks到向量存储中
        """
        self.logger.info(f"开始添加{len(chunks)}个chunks到向量存储中...")
        try:
            #准备文档内容和元数据
            texts = [chunk.content for chunk in chunks]
            metadatas = []

            for chunk in chunks:
                #为chorma准备元数据
                metadata = chunk.metadata.copy()
                metadata.update( {
                    "chunk_id" :chunk.chunk_id,
                    "content_length": len(chunk.content),
                    "parent_doc_id": chunk.parent_doc_id or ''
                })
                metadatas.append(metadata)

                #保存完整的chunk信息
                self.chunks_metadata[chunk.chunk_id] = chunk

            #创建向量数据库chroma
            self.chroma_db = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding_model.client,# # 使用langchain兼容的embedding
                metadatas=metadatas,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name

            )
            self.logger.info("向量存储添加完成")

        except Exception as e:
            self.logger.error(f"添加chunks到向量存储时出错：{e}")
            raise SearchError(f"添加chunks到向量存储时出错：{e}")

    def search(self, query: str, k: int = 4)-> List[SearchResult]:
        """
        搜索相似chunks
        """
        if not self.chroma_db:
            raise SearchError("向量存储未初始化")
        try:
            #使用chorma向量数据库进行相似性搜索
            results = self.chroma_db.similarity_search_with_score(query, k=k)
            search_results = []
            for i,(doc, score) in enumerate(results):
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id and chunk_id in self.chunks_metadata:
                    chunk = self.chunks_metadata[chunk_id]
                    search_results.append(SearchResult(
                        chunk = chunk,
                        score = score,
                        rank = i+1
                    ))
            return search_results

        except Exception as e:
            self.logger.error(f"搜索相似chunks时出错：{e}")
            raise SearchError(f"搜索相似chunks时出错：{e}")

    def save(self,path: str)-> None:
        """
        保存额外的metadata
        """
        try:
            with open(os.path.join(path, "chunks_metadata.pkl"), "wb") as f:
                pickle.dump(self.chunks_metadata, f)
            self.logger.info("保存额外的metadata完成,向量存储元数据保存成功")
        except Exception as e:
            self.logger.error(f"保存metadata时出错：{e}")

    def load(self, path: str) -> None:
        """
        加载向量存储，确保元数据为可变字典
        """
        try:
            # 修复拼写错误：chorma_db → chroma_db（避免后续引用错误）
            # 修复拼写错误：chorma_db → chroma_db（避免后续引用错误）
            self.chroma_db = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model.client
            )

            # 验证chroma_db是否初始化成功
            if self.chroma_db is None:
                raise SearchError("Chroma数据库初始化失败")

            # 验证客户端是否正确初始化
            if not hasattr(self.chroma_db, 'client') or self.chroma_db.client is None:
                raise SearchError("Chroma数据库客户端初始化失败")

            # 加载metadata并确保可变
            metadata_path = os.path.join(path, "chunks_metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    loaded_metadata = pickle.load(f)

                    # 关键修复：强制转换为可变字典
                    if isinstance(loaded_metadata, dict):
                        # 对顶层字典转换
                        self.chunks_metadata = dict(loaded_metadata)
                        # 如果有嵌套字典，递归转换（根据实际数据结构调整）
                        for key, value in self.chunks_metadata.items():
                            if isinstance(value, dict):
                                self.chunks_metadata[key] = dict(value)
                    else:
                        # 若加载的不是字典，初始化空字典避免报错
                        self.chunks_metadata = {}
                        self.logger.warning(f"元数据格式异常，已初始化为空字典")
            else:
                # 元数据文件不存在时，初始化空字典
                self.chunks_metadata = {}
                self.logger.info("未找到元数据文件，初始化为空字典")

            self.logger.info("向量存储加载成功")

        except Exception as e:
            self.logger.error(f"加载向量存储时出错：{e}")
            raise SearchError(f"加载向量存储时出错：{e}")

class VectorStoreManager:
    """
    向量存储管理器
    """
    def __init__(self, embedding_model: BaseEmbedding, config):
        if embedding_model is None:
            raise ValueError("embedding_model 不能为空")
        self.embedding_model = embedding_model
        self.config = config
        self.logger = get_logger(__name__)
        self.vector_store = None

    def get_vector_store(self)-> BaseVectorStore:
        """
        获取向量存储实例
        """
        if self.vector_store:
            return self.vector_store

        if self.config.provider == "chroma":
            self.vector_store = ChromaVectorStore(
                embedding_model=self.embedding_model,
                persist_directory=self.config.persist_directory,
                collection_name=self.config.collection_name
            )
        else:
            raise ValueError(f"不支持的向量存储提供者：{self.config.provider}")
        return self.vector_store

















































