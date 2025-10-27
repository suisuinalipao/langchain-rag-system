"""
自动下载LangChain文档（用于Streamlit Cloud）
"""
import os
import subprocess
import sys


def setup_langchain_docs():
    """下载LangChain文档"""
    docs_path = "langchain"

    if os.path.exists(docs_path):
        print(f"✅ 文档已存在: {docs_path}")
        return True

    print("📥 正在下载LangChain文档...")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/langchain-ai/langchain.git"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("✅ 文档下载成功！")
            return True
        else:
            print(f"❌ 文档下载失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ 下载出错: {e}")
        return False


if __name__ == "__main__":
    success = setup_langchain_docs()
    sys.exit(0 if success else 1)