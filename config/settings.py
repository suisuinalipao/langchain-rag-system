import os
from typing import Dict,Any,Optional
from dataclasses import dataclass,field

# 添加 dotenv 支持
try:
    from dotenv import load_dotenv
    import os as os_path
    # 获取当前文件所在的目录
    current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
    # 构建 .env 文件的路径
    env_path = os_path.path.join(current_dir, '..', '.env')
    # 加载 .env 文件
    load_dotenv(env_path)
except ImportError:
    pass

@dataclass
class ModelConfig:
    """模型配置类"""
    provider: str  # Model provider 'openai', 'deepseek', 'local'
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None

@dataclass
class EmbeddingConfig:
    """嵌入模型配置类"""
    provider: str # 'openai', 'deepseek', 'huggingface'
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None

@dataclass
class VectorStoreConfig:
    """向量数据库配置类"""
    provider: str # 'chroma', 'faiss', 'pinecone'
    persist_directory: Optional[str] =  None
    collection_name: str = "langchain_docs"

@dataclass
class ProcessingConfig:
    """文档处理配置类"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = field(default_factory=lambda: ['\n\n', '\n', ' ', ''])

@dataclass
class RetrievalConfig:
    """向量检索配置类"""
    serch_type: str = "similarity"
    k: int = 4
    score_threshold: Optional[float] = None

class Settings:
    """全局配置管理类"""
    def __init__(self,config_path:Optional[str]=None):
        self.config_path = config_path
        self.load_default_config()

    def load_default_config(self):
        """
        加载默认配置
        """
        #LLM配置
        self.llm_config = ModelConfig(
            provider=os.getenv("LLM_PROVIDER", "deepseek"),
            model_name=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1"))
        )

        #嵌入模型配置
        self.embedding_config = EmbeddingConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )

        #向量数据库配置
        self.vector_store_config = VectorStoreConfig(
            provider=os.getenv("VECTOR_STORE_PROVIDER", "chroma"),
            persist_directory=os.getenv("VECTOR_STORE_PATH", "chroma_db"),
            collection_name=os.getenv("COLLECTION_NAME", "langchain_docs")
        )

        #文档处理配置
        self.processing_config = ProcessingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        )

        #向量检索配置
        self.retrieval_config = RetrievalConfig(
            k = int(os.getenv("RETRIEVAL_K", "4")),
            score_threshold = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.0"))  if os.getenv("SCORE_THRESHOLD") else None

        )

        # 文档路径
        # 使用项目根目录作为基准来定位文档路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 回到项目根目录
        project_root = os.path.dirname(project_root)
        default_docs_path = os.path.join(project_root, "langchain_konwledge_base", "docs", "docs")
        self.docs_path = os.getenv("DOCS_PATH", default_docs_path)
        # 确保路径为绝对路径
        self.docs_path = os.path.abspath(self.docs_path)
    def validate_config(self):
        """
        验证配置的完整性
        """
        errors = []
        if self.llm_config.provider in['openai','deepseek'] and not self.llm_config.api_key :
            errors.append(f"缺少{self.llm_config.provider}的API密钥")

        if self.embedding_config.provider in['openai','deepseek'] and not self.embedding_config.api_key :
            errors.append(f"缺少{self.embedding_config.provider}的API密钥")

        #检查文档路径
        if not os.path.exists(self.docs_path):
            errors.append(f"文档路径{self.docs_path}不存在")
        if errors:
            for error in errors:
                print(f"配置错误: {error}")

        return  True
    































