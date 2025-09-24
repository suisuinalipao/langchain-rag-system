import time
from datetime import datetime
from typing import List, Optional
from config.models import QAResult, SearchResult
from models.base import BaseLLM,BaseVectorStore
from config .settings import RetrievalConfig
from utils.logger import get_logger

class QAEngine:
    """
    问答引擎
    """
    def __init__(self, llm: BaseLLM, vector_store: BaseVectorStore, config: RetrievalConfig):
        self.llm = llm
        self.vector_store = vector_store
        self.config = config
        self.logger = get_logger(__name__)

    def answer_question(self, question: str) -> QAResult:
        """
        回答问题
        """
        start_time = time.time()
        self.logger.info(f"开始处理问题: {question}")

        try:
            #1.检索相关文档。
            search_results = self.vector_store.search(question, k=self.config.k)

            if not search_results:
                return QAResult(
                    question=question,
                    answer="抱歉，没有找到相关信息。",
                    source_chunk=[],
                    processing_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
            self.logger.info(f"检索到 {len(search_results)} 个相关chunks")

            #2.准备上下文
            context = self._prepare_context(search_results)

            #3.生成回答
            answer = self.llm.generate_with_context(question, context)

            processing_time = time.time() - start_time
            self.logger.info(f"问题回答完成，耗时 {processing_time:.2f} 秒")

            return QAResult(
                question=question,
                answer=answer,
                source_chunk=search_results,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"回答问题时出错: {e}")
            return QAResult(
                question=question,
                answer="抱歉，处理问题时出错。",
                source_chunk=[],
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )


    def _prepare_context(self, search_results: List[SearchResult])-> str:
        """
        准备上下文信息
        """
        context_parts = []

        for i ,result in enumerate(search_results):
            chunk = result.chunk
            source = chunk.metadata.get("source", "未知来源")

            #获取标题信息
            headers = []
            for key, value in chunk.metadata.items():
                if key.startswith("Header"):
                    headers.append(f"{key}: {value}")

            header_info = " | ".join(headers) if headers else "无标题信息"
            context_part = f"""
            来源 {i + 1}: {source}
            标题: {header_info}
            相关度: {result.score:.3f}

            内容:
            {chunk.content}
            """
            context_parts.append(context_part)

        return "\n" + "="*80 + "\n".join(context_parts)

    def batch_answer_questions(self, questions: List[str]) -> List[QAResult]:
        """
        批量处理问题
        """
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)
        return results










































