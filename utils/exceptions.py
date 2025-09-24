class RAGSystemError(Exception):
    """
    RAG系统基础异常类
    """
    pass

class DocumentLoadError(RAGSystemError):
    """
    文档加载异常类
    """
    pass

class EmbeddingError(RAGSystemError):
    """
    嵌入异常类
    """
    pass

class SearchError(RAGSystemError):
    """
    搜索异常类
    """
    pass

class ConfigurationError(RAGSystemError):
    """
    配置异常类
    """
    pass
