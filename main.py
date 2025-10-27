import os
import sys
from config.settings import Settings
from core.document_loader import DocumentLoader
from core.text_processor import  TextProcessor
from core.vector_store import VectorStoreManager
from core.qa_engine import QAEngine
from core.qa_engine  import *
from utils.logger import get_logger,setup_logger
from models.deepseek_models import DeepSeekLLM
from models.huggingface_models import HuggingFaceEmbedding
from utils.exceptions import RAGSystemError, ConfigurationError, SearchError
from typing import List, Optional

class RAGSystem:
    """
    RAG系统主类
    """
    def __init__(self,config_path:Optional[str]= None):
        #设置日志
        self.logger= setup_logger()

        #加载配置
        self.settings = Settings(config_path)

        if not self.settings.validate_config():
            raise ConfigurationError("配置验证失败")

        #初始化组件
        self.embeddings_model = None
        self.llm_model = None
        self.vector_store_manager = None
        self.qa_engine = None

        self.logger.info("RAG系统初始化完成")

    def _validata_custom_config(self)-> bool:
        """
        自定义配置验证
        """
        # 检查LLM的API密钥
        errors = []
        if self.settings.llm_config.provider == "deepseek" and not self.settings.llm_config.api_key:
            errors.append("缺少DeepSeek的API密钥")

        # 检查文档路径
        if not os.path.exists(self.settings.docs_path):
            errors.append(f"文档路径不存在: {self.settings.docs_path}")

        if errors:
            for error in errors:
                print(f"配置错误: {error}")
            return False

        return True

    def _initialize_models(self):
        """
        初始化模型
        """
        # 初始化嵌入模型
        if self.settings.embedding_config.provider == "huggingface":
            self.logger.info("正在加载HuggingFace嵌入模型（首次加载需要下载模型，请耐心等待）...")
            self.embedding_model = HuggingFaceEmbedding(self.settings.embedding_config)
        else:
            raise ValueError(f"不支持的嵌入模型: {self.settings.embedding_config.provider}")

        #初始化LLM模型
        if self.settings.llm_config.provider == "deepseek":
            self.llm_model = DeepSeekLLM(self.settings.llm_config)
        else:
            raise ValueError(f"不支持的LLM模型: {self.settings.llm_config.provider}")
        self.logger.info("模型初始化完成")

    def build_knowledge_base(self, force_rebuild: bool = False):
        self.logger.info("开始构建知识库")
        self._initialize_models()  # 确保模型先初始化

        # 修复判断逻辑：如果存在且不强制重建，则加载；否则创建新的
        if os.path.exists(self.settings.vector_store_config.persist_directory) and not force_rebuild:
            self.logger.info("发现已存在的向量数据库，直接加载")
            self._load_existing_knowledge_base()
        else:
            # 原代码错误地打印了"发现已存在..."，这里修正日志
            self.logger.info("未发现向量数据库或需要强制重建，开始创建新知识库")
            self._create_new_knowledge_base()
        self.logger.info("知识库构建完成")
        if not self.vector_store_manager or not self.qa_engine:
            raise RAGSystemError("知识库构建不完整，向量存储未初始化")

    def _load_existing_knowledge_base(self):
        self.vector_store_manager = VectorStoreManager(
            self.embedding_model,
            self.settings.vector_store_config
        )
        # 获取向量存储实例
        self.vector_store = self.vector_store_manager.get_vector_store()
        # 加载向量数据
        self.vector_store.load(self.settings.vector_store_config.persist_directory)
        # 新增验证
        if not hasattr(self.vector_store, 'chroma_db') or self.vector_store.chroma_db is None:
            raise SearchError("向量存储加载失败，chroma_db未初始化")

        self.qa_engine = QAEngine(
            self.llm_model,
            self.vector_store,  # 使用已加载的向量存储实例
            self.settings.retrieval_config
        )

    def _create_new_knowledge_base(self):
        """
        创建新的知识库
        """
        #1.加载文档
        loader = DocumentLoader(self.settings.docs_path)
        documents = loader.load_documents()
        
        # 检查是否成功加载文档
        if not documents:
            raise RAGSystemError(f"未能从路径 {self.settings.docs_path} 加载任何文档")

        #2.处理文档
        processor = TextProcessor(self.settings.processing_config)
        chunks = processor.process_documents(documents)
        
        # 检查是否有生成chunks
        if not chunks:
            raise RAGSystemError("文档处理后未生成任何文本块")

        #3.向量化(创建向量存储)
        self.vector_store_manager = VectorStoreManager(
            self.embedding_model,
            self.settings.vector_store_config
        )
        vector_store = self.vector_store_manager.get_vector_store()
        vector_store.add_chunks(chunks)
        vector_store.save(self.settings.vector_store_config.persist_directory)

        #4.创建QA引擎
        self.qa_engine = QAEngine(
            self.llm_model,
            vector_store,
            self.settings.retrieval_config
        )
    def answer_question(self,question:str)-> str:
        """
        回答问题
        """
        if not self.qa_engine:
            raise RAGSystemError("知识库未构建，请先调用 build_knowledge_base()")
        return self.qa_engine.answer_question(question)

    def interactive_mode(self):
        """交互式问答模式"""
        print("\n" + "=" * 60)
        print("🤖 LangChain RAG问答系统")
        print(
            f"📚 使用模型: {self.settings.llm_config.provider} - {self.settings.llm_config.model_name}")
        print(
            f"🔍 嵌入模型: {self.settings.embedding_config.provider} - {self.settings.embedding_config.model_name}")
        print("💬 输入问题开始对话，输入 'quit' 退出")
        print("=" * 60)

        sample_questions = [
            "LangSmith 的核心功能是什么？它如何提升 AI 应用的可观测性与可评估性？",
            "angChain 如何帮助开发者在不同 LLM 模型之间做出平衡决策（准确率、延迟、成本）？",
            "LangChain 生态系统中有哪些主要组件？它们之间是什么关系？",
            "angChain 的主要目标是什么？它希望帮助开发者解决哪些问题？",
            "LangChain 的 “retriever（检索器）” 接口在 RAG 应用中扮演了什么角色？"
        ]

        print("\n🔍 示例问题:")
        for i, q in enumerate(sample_questions, 1):
            print(f"{i}. {q}")
        print()

        while True:
            try:
                question = input("💬 请输入问题: ").strip()

                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break

                if not question:
                    continue

                print(f"\n🤔 正在思考问题: {question}")
                result = self.answer_question(question)

                print(f"\n🤖 回答:")
                print(result.answer)

                print(f"\n📊 检索信息:")
                print(f"处理时间: {result.processing_time:.2f}秒")
                print(f"参考文档数量: {len(result.source_chunk)}")

                if result.source_chunk:
                    print("\n📚 参考来源:")
                    for i, source in enumerate(result.source_chunk, 1):
                        source_file = source.chunk.metadata.get('source', '未知')
                        print(
                            f"{i}. {os.path.basename(source_file)} (相关度: {source.score:.3f})")

                print("\n" + "-" * 60)

            except KeyboardInterrupt:
                print("\n👋 程序被中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 处理问题时出错: {e}")
                continue

def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RAGSystem()

        # 构建知识库
        print("正在初始化RAG系统...")
        rag_system.build_knowledge_base(force_rebuild=True)

        # 启动交互模式
        rag_system.interactive_mode()

    except ConfigurationError as e:
        print(f"配置错误: {e}")
        print("请检查环境变量和配置文件")
    except RAGSystemError as e:
        print(f"系统错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()









































