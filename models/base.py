from abc import ABC,abstractmethod
from typing import List,Dict,Any, Optional, Tuple
from config.models import Document, Chunk, SearchResult

class BaseDocument(ABC):
    """
    嵌入模型基类
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        将文本转换为向量
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量将文本转换为向量"""
        pass

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: str = None) -> str:
        """
        抽象方法：生成回答（子类必须实现）
        参数：
            prompt: 用户问题（字符串）
            context: 检索到的上下文（字符串，可选，默认None）
        返回：
            回答内容（字符串）
        """
        pass

    @abstractmethod
    def generate_with_context(self, question: str, context: str, **kwargs) -> str:
        """
        基于上下文回答
        """
        pass
    """
    比如你之前可能写了 def generate(self, prompt: str, **kwargs) -> str，虽然加了 **kwargs，但父类要求的 context 参数没「显式出现」，ABC 就不认。
    """


class BaseEmbedding(ABC):
    """嵌入模型的基类/接口"""

    @abstractmethod
    def embed_documents(self, texts: List[str])-> List[List[float]]:
        """
        对文档列表进行嵌入
        参数: texts - 字符串列表
        返回: 嵌入向量列表
        """
        pass

    @abstractmethod
    def embed_query(self, text: str)-> List[float]:
        """
        对单个查询进行嵌入
        参数: text - 字符串
        返回: 嵌入向量
        """
        pass

        # 保持向后兼容的方法
    def embed_text(self, text: str) -> List[float]:
        """将单个文本转换为向量 - 向后兼容方法"""
        return self.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量转换文本为向量 - 向后兼容方法"""
        return self.embed_documents(texts)

@abstractmethod
class BaseVectorStore(ABC):
    """
    向量数据库基类
    """

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        添加文档块到向量存储
        """
        pass

    @abstractmethod
    def search(self,query_embedding: List[float], k: int = 4) ->List[SearchResult]:
        """
        搜索相似向量
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存向量存储
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        加载向量存储
        """
        pass













