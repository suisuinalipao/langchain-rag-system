# LangChain RAG智能问答系统

一个基于LangChain文档的智能问答系统，使用RAG（检索增强生成）技术。

## 功能特点

- 🔧 **模块化设计**: 可扩展的架构，支持多种模型和向量数据库
- 🚀 **多模型支持**: 支持OpenAI、HuggingFace等多种LLM和嵌入模型
- 📚 **智能文档处理**: 按标题层级智能切分文档
- 🔍 **高效检索**: 基于向量相似度的智能检索
- 💾 **持久化存储**: 向量数据库可持久化，避免重复构建
- 🎯 **准确回答**: 基于官方文档的准确回答

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd rag_system

# 安装依赖
pip install -r requirements.txt

# 克隆LangChain文档
git clone https://github.com/langchain-ai/langchain.git
```

### 2. 配置设置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，填入你的API密钥
# DEEPSEEK_API_KEY=sk-your-api-key-here
```

### 3. 运行系统

```bash
python main.py
```

## 项目架构

### 核心模块

- **config/**: 配置管理，支持环境变量和配置文件
- **models/**: 模型抽象层，支持多种LLM和嵌入模型
- **core/**: 核心业务逻辑
  - `document_loader`: 文档加载
  - `text_processor`: 文本处理和切分
  - `vector_store`: 向量数据库管理
  - `qa_engine`: 问答引擎
- **utils/**: 工具函数，日志和异常处理

### 扩展性

系统采用接口抽象设计，可以轻松扩展：

- 新的LLM模型：实现`BaseLLM`接口
- 新的嵌入模型：实现`BaseEmbedding`接口  
- 新的向量数据库：实现`BaseVectorStore`接口

## 使用示例

```python
from rag_system import RAGSystem

# 创建系统实例
rag = RAGSystem()

# 构建知识库
rag.build_knowledge_base()

# 问答
result = rag.ask_question("什么是LangChain？")
print(result.answer)
```

## 配置选项

所有配置都可以通过环境变量设置，详见`.env.example`文件。

主要配置项：
- 模型选择（OpenAI/DeepSeek）
- 文档路径
- 切分参数
- 检索参数

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License
