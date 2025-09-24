import torch
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from models.base import BaseEmbedding
from config.settings import EmbeddingConfig
from utils.logger import get_logger


class HuggingFaceEmbedding(BaseEmbedding):
    """Hugging Face嵌入模型实现"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # 检查是否有GPU可用
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"使用设备: {self.device}")

        # 初始化模型参数
        model_kwargs = {
            'device': self.device,
            'trust_remote_code': True
        }

        encode_kwargs = {
            'normalize_embeddings': True,  # 归一化嵌入向量
            'batch_size': 32 if self.device == "cuda" else 8  # CPU使用较小的batch size
        }

        try:
            # 使用LangChain的HuggingFace嵌入
            self.client = HuggingFaceEmbeddings(
                model_name=config.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                show_progress=True
            )

            self.logger.info(f"成功加载Hugging Face嵌入模型: {config.model_name}")

        except Exception as e:
            self.logger.error(f"加载Hugging Face模型失败: {e}")
            # 尝试使用更简单的配置
            try:
                self.client = HuggingFaceEmbeddings(
                    model_name=config.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.logger.warning("使用CPU后备模式加载模型")
            except Exception as backup_error:
                self.logger.error(f"后备模式也失败: {backup_error}")
                raise backup_error

    def embed_text(self, text: str) -> List[float]:
        """将单个文本转换为向量 - 兼容旧接口"""
        return self.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量转换文本为向量 - 兼容旧接口"""
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为向量 - BaseEmbedding要求的方法"""
        if not text or not text.strip():
            self.logger.warning("输入文本为空，返回零向量")
            return [0.0] * 384  # all-MiniLM-L6-v2的维度是384

        try:
            embedding = self.client.embed_query(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Hugging Face查询嵌入失败: {e}")
            # 返回零向量作为后备
            return [0.0] * 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量转换文档为向量 - BaseEmbedding要求的方法"""
        if not texts:
            return []

        # 过滤空文本
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return [[0.0] * 384] * len(texts)

        try:
            embeddings = self.client.embed_documents(valid_texts)

            # 如果原始列表中有空文本，需要补充零向量
            if len(valid_texts) != len(texts):
                result = []
                valid_idx = 0
                for text in texts:
                    if text and text.strip():
                        result.append(embeddings[valid_idx])
                        valid_idx += 1
                    else:
                        result.append([0.0] * 384)
                return result

            return embeddings

        except Exception as e:
            self.logger.error(f"Hugging Face批量嵌入失败: {e}")
            # 返回零向量列表作为后备
            return [[0.0] * 384] * len(texts)

