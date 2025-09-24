from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

@dataclass
class Document:
    """
    文档数据模型
    """
    content : str
    metadata:Dict[str,Any]
    doc_id:Optional[str] =  None

@dataclass
class Chunk:
    """
    文档分块数据模型
    """
    content : str
    metadata:Dict[str,Any]
    chunk_id:str
    embedding:Optional[List[float]] =   None
    parent_doc_id:Optional[str] =   None # 父文档id,溯源功能，上下文扩展，去重，可以知道当前文档块来自哪一个Document

@dataclass
class SearchResult:
    """
    搜索结果数据模型
    """
    chunk:Chunk
    score:float
    rank: int

@dataclass
class QAResult:
    """
    问答结果数据模型
    """
    question:str
    answer:str
    source_chunk:List[SearchResult]
    processing_time:float
    timestamp:datetime
    confidence:Optional[float] = None




































