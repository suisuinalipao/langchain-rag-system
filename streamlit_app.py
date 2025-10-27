import subprocess
import sys
from pathlib import Path

# 自动下载LangChain文档
if not Path("langchain").exists():
    import streamlit as st
    st.info("📥 首次运行，正在下载LangChain文档...")
    with st.spinner("下载中，请稍候（约1-2分钟）..."):
        result = subprocess.run([sys.executable, "setup_docs.py"])
        if result.returncode == 0:
            st.success("✅ 文档下载完成！")
            st.rerun()
        else:
            st.error("❌ 文档下载失败")
            st.stop()

import streamlit as st
import os
import sys
import time

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录到Python路径
sys.path.insert(0, current_dir)

# 设置当前工作目录为项目根目录
os.chdir(current_dir)

from main import RAGSystem
from utils.logger import setup_logger

# 初始化logger
logger = setup_logger()

# 设置页面配置
st.set_page_config(
    page_title="RAG问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面标题
st.title("🤖 LangChain RAG问答系统")

# 在侧边栏添加系统信息
st.sidebar.header("系统信息")

# 初始化session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# 在侧边栏添加初始化按钮
with st.sidebar:
    st.subheader("系统初始化")
    
    force_rebuild = st.checkbox("强制重建知识库", value=False)
    
    if st.button("初始化系统"):
        with st.spinner("正在初始化RAG系统..."):
            try:
                # 创建RAG系统实例
                rag_system = RAGSystem()
                
                # 构建知识库
                rag_system.build_knowledge_base(force_rebuild=force_rebuild)
                
                # 保存到session state
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                
                st.success("✅ 系统初始化成功!")
                logger.info("RAG系统在Streamlit中初始化成功")
                
            except Exception as e:
                st.error(f"❌ 系统初始化失败: {str(e)}")
                logger.error(f"RAG系统初始化失败: {str(e)}")

# 主要内容区域
if not st.session_state.initialized:
    st.info("📢 请在左侧边栏点击'初始化系统'按钮以启动RAG问答系统")
    st.warning("⚠️ 首次运行可能需要较长时间来下载模型和构建知识库，请耐心等待")
    
    st.markdown("""
    ## 使用说明
    
    1. 点击左侧边栏的"初始化系统"按钮
    2. 等待系统初始化完成（首次运行可能需要几分钟）
    3. 初始化成功后，即可在下方输入问题进行问答
    
    ## 系统特性
    
    - 📚 基于LangChain的知识库问答系统
    - 🔍 支持语义搜索和相似度匹配
    - 🧠 利用大语言模型生成准确答案
    - 📊 显示答案来源和相关度评分
    """)
else:
    st.success("✅ RAG系统已就绪，可以开始提问了！")
    
    # 创建两列布局
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 用户输入问题
        question = st.text_input("请输入您的问题:", placeholder="例如: 什么是LangChain?")
        
        # 添加示例问题按钮
        st.subheader("常见问题示例:")
        example_questions = [
            "什么是LangChain？",
            "如何创建一个LLM链？",
            "LangChain中的Memory组件有哪些？"
        ]
        
        # 创建示例行
        example_cols = st.columns(len(example_questions))
        for idx, (col, example_q) in enumerate(zip(example_cols, example_questions)):
            if col.button(example_q, key=f"example_{idx}"):
                question = example_q
                
    with col2:
        # 参数设置
        st.subheader("参数设置")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 2000, 500, 50)
    
    # 处理问题回答
    if question:
        with st.spinner("正在思考中..."):
            try:
                start_time = time.time()
                result = st.session_state.rag_system.answer_question(question)
                end_time = time.time()
                
                # 显示答案
                st.subheader("🤖 回答:")
                st.write(result.answer)
                
                # 显示相关信息
                st.subheader("📊 相关信息:")
                col1, col2, col3 = st.columns(3)
                col1.metric("处理时间", f"{end_time - start_time:.2f}秒")
                col2.metric("参考文档数", len(result.source_chunk))
                col3.metric("模型温度", temperature)
                
                # 显示参考来源
                if result.source_chunk:
                    st.subheader("📚 参考来源:")
                    for i, source in enumerate(result.source_chunk[:3], 1):  # 只显示前3个
                        source_file = source.chunk.metadata.get('source', '未知')
                        similarity = source.score
                        
                        with st.expander(f"来源 {i}: {os.path.basename(source_file)} (相关度: {similarity:.3f})"):
                            st.write(f"**文件名:** {os.path.basename(source_file)}")
                            st.write(f"**相关度:** {similarity:.3f}")
                            st.write("**内容预览:**")
                            st.text(source.chunk.content[:500] + "..." if len(source.chunk.content) > 500 else source.chunk.content)
                            
            except Exception as e:
                st.error(f"❌ 回答问题时出错: {str(e)}")
                logger.error(f"回答问题时出错: {str(e)}")
    
    # 显示系统状态
    st.sidebar.subheader("系统状态")
    st.sidebar.info("🟢 运行中")
    
    # 显示版本信息
    st.sidebar.subheader("版本信息")
    st.sidebar.text("RAG问答系统 v1.0")

if __name__ == "__main__":
    pass