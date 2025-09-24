import hashlib
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from config.models import Document, Chunk
from utils.logger import get_logger
from config.settings import ProcessingConfig


class TextProcessor:
    """文本处理器：负责文档分割、Chunk 创建"""

    def __init__(self, config: ProcessingConfig):
        self.logger = get_logger(__name__)
        self.config = config

        # 基础文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=self.config.separators,
        )

        # Markdown 标题分割器
        self.header_splitter = MarkdownTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            keep_separator=True
        )

    def process_documents(self, documents: List[Document]) -> List[Chunk]:
        self.logger.info(f"开始处理 {len(documents)} 个文档")
        all_chunks = []
        for doc in documents:
            chunks = self.process_single_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def process_single_document(self, document: Document) -> List[Chunk]:
        chunks = []
        try:
            header_based_chunks = self.header_splitter.split_text(document.content)

            for chunk_idx, chunk_content in enumerate(header_based_chunks):
                if len(chunk_content) > self.config.chunk_size:
                    sub_chunks = self.text_splitter.split_text(chunk_content)
                    for sub_idx, sub_content in enumerate(sub_chunks):
                        # 关键修复1：用 dict() 包装 document.metadata，确保是可变字典
                        metadata = dict(document.metadata)  # 先转换为可变字典
                        # 再添加新字段（替代 **展开，避免继承只读特性）
                        metadata.update({
                            "split_type": "markdown_header + recursive",
                            "parent_header_chunk_idx": chunk_idx
                        })
                        chunk = self._create_chunk(
                            content=sub_content,
                            metadata=metadata,
                            parent_doc_id=document.doc_id,
                            chunk_index=f"{chunk_idx}_{sub_idx}"
                        )
                        chunks.append(chunk)
                else:
                    # 关键修复2：同样转换为可变字典
                    metadata = dict(document.metadata)
                    metadata.update({
                        "split_type": "markdown_header",
                        "header_chunk_idx": chunk_idx
                    })
                    chunk = self._create_chunk(
                        content=chunk_content,
                        metadata=metadata,
                        parent_doc_id=document.doc_id,
                        chunk_index=chunk_idx
                    )
                    chunks.append(chunk)

        except Exception as e:
            self.logger.error(f"处理文档 {document.doc_id} 时出错：{e}，使用基础分割器降级处理")
            simple_chunks = self.text_splitter.split_text(document.content)
            for simple_idx, simple_content in enumerate(simple_chunks):
                # 关键修复3：异常处理中也确保可变
                metadata = dict(document.metadata)
                metadata.update({"split_type": "recursive_fallback"})
                chunk = self._create_chunk(
                    content=simple_content,
                    metadata=metadata,
                    parent_doc_id=document.doc_id,
                    chunk_index=simple_idx
                )
                chunks.append(chunk)

        return chunks

    def _create_chunk(self, content: str, metadata: Dict, parent_doc_id: str, chunk_index: str or int) -> Chunk:
        unique_key = f"{parent_doc_id}_{chunk_index}_{content[:50]}".encode("utf-8")
        chunk_id = hashlib.md5(unique_key).hexdigest()
        # 关键修复：创建全新的字典，彻底摆脱原字典的只读关联
        mutable_metadata = {}
        # 逐个复制键值对（避免直接引用原字典）
        for key, value in metadata.items():
            mutable_metadata[key] = value
        return Chunk(
            content=content,
            metadata=metadata,  # 此时 metadata 已是可变字典
            chunk_id=chunk_id,
            parent_doc_id=parent_doc_id
        )
