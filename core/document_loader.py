import os
from typing import Any, Dict, List
from langchain_community.document_loaders import DirectoryLoader,UnstructuredMarkdownLoader
from langchain.schema import Document as LangChainDocument
from config.models import  Document
from utils.logger import get_logger
from utils.exceptions import DocumentLoadError

class DocumentLoader:
    """
    文档加载器
    """
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.logger = get_logger(__name__)# __name__ 在这个文件中的值是 'core.document_loader'

    def load_documents(self)-> List[Document]:
        """
        加载所有文档
        """
        self.logger.info(f"开始从{self.docs_path}加载文档...")

        if not os.path.exists(self.docs_path):
            raise DocumentLoadError(f"文档路径不存在: {self.docs_path}")

        try:
            # 检查目录是否为空
            docs_files = [f for f in os.listdir(self.docs_path) if f.endswith('.md') or f.endswith('.mdx')]
            if not docs_files:
                self.logger.warning(f"文档路径 {self.docs_path} 中没有找到 .md 或 .mdx 文件")
            
            self.logger.info(f"在 {self.docs_path} 中找到 {len(docs_files)} 个文档文件")
            
            # 使用langchain的DirectLoader加载所有文档
            loader = DirectoryLoader(
                self.docs_path,
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader,# 指定用什么加载器处理每个文件
                recursive=True,# 递归搜索所有子目录
                show_progress=True
            )
            # ** = 匹配任意深度的目录
            # *.md* = 匹配以.md开头的文件

            # 会匹配：
            # docs/intro.md
            # docs/concepts/memory.mdx
            # docs/tutorials/basic/setup.md
            # docs/advanced/chains/custom.mdx


            # UnstructuredMarkdownLoader 专门处理Markdown文件
            # 它能解析Markdown结构，提取文本内容
            langchain_docs = loader.load()
            
            self.logger.info(f"DirectoryLoader 加载了 {len(langchain_docs)} 个文档")

            # 尝试使用不同的glob模式加载.md和.mdx文件
            if len(langchain_docs) == 0:
                self.logger.info("未找到.md文件，尝试加载.mdx文件...")
                loader_mdx = DirectoryLoader(
                    self.docs_path,
                    glob="**/*.mdx",
                    loader_cls=UnstructuredMarkdownLoader,
                    recursive=True,
                    show_progress=True
                )
                langchain_docs = loader_mdx.load()
                self.logger.info(f"DirectoryLoader (mdx) 加载了 {len(langchain_docs)} 个文档")

            # 将langchain的文档转换为Document对象
            documents = []
            for i ,doc in enumerate(langchain_docs):
                documents.append(
                    Document(
                        content=doc.page_content,
                        doc_id=f"doc_{i}",
                        metadata=doc.metadata
                    )
                )
            self.logger.info(f"成功加载{len(documents)}个文档")
            return documents

        except Exception as e:
            self.logger.error(f"加载文档时出错: {e}")
            raise DocumentLoadError(f"加载文档时出错: {e}")

    def load_single_document(self, file_path: str)-> Document:
        """
        加载单个文档
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return Document(
                content=content,
                doc_id=os.path.basename(file_path),
                metadata={"source": file_path}
            )

        except Exception as e:
            raise DocumentLoadError(f"加载文档{file_path}时出错: {e}")
