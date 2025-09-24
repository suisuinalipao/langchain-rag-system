# test_embedding_fix.py
# 用于测试修复后的嵌入模型

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_huggingface_embedding():
    """测试Hugging Face嵌入模型是否正常工作"""
    try:
        from config.settings import EmbeddingConfig
        from models.huggingface_models import HuggingFaceEmbedding

        print("正在测试Hugging Face嵌入模型...")

        # 创建配置
        config = EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 初始化模型
        print("正在初始化模型...")
        embedding = HuggingFaceEmbedding(config)
        print("✅ 模型初始化成功")

        # 测试 embed_query 方法
        print("测试 embed_query 方法...")
        query_vector = embedding.embed_query("这是一个测试查询")
        print(f"✅ embed_query 成功，维度: {len(query_vector)}")

        # 测试 embed_documents 方法
        print("测试 embed_documents 方法...")
        docs = ["文档1", "文档2", "文档3"]
        doc_vectors = embedding.embed_documents(docs)
        print(f"✅ embed_documents 成功，处理了 {len(doc_vectors)} 个文档")

        # 测试向后兼容方法
        print("测试向后兼容方法...")
        text_vector = embedding.embed_text("测试文本")
        texts_vectors = embedding.embed_texts(["文本1", "文本2"])
        print(f"✅ 向后兼容方法正常工作")

        # 测试空文本处理
        print("测试空文本处理...")
        empty_vector = embedding.embed_query("")
        empty_docs = embedding.embed_documents(["", "正常文本", ""])
        print(f"✅ 空文本处理正常")

        print("\n🎉 所有测试通过！Hugging Face嵌入模型工作正常。")
        return True

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保安装了所需依赖：")
        print("pip install langchain-huggingface transformers sentence-transformers torch")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_class():
    """测试基类接口"""
    try:
        from models.base import BaseEmbedding
        from models.huggingface_models import HuggingFaceEmbedding

        print("测试基类接口...")

        # 检查抽象方法
        required_methods = ['embed_query', 'embed_documents']

        for method in required_methods:
            if not hasattr(BaseEmbedding, method):
                print(f"❌ 基类缺少方法: {method}")
                return False
            print(f"✅ 基类包含方法: {method}")

        # 检查实现类
        for method in required_methods:
            if not hasattr(HuggingFaceEmbedding, method):
                print(f"❌ 实现类缺少方法: {method}")
                return False
            print(f"✅ 实现类包含方法: {method}")

        print("✅ 基类接口测试通过")
        return True

    except Exception as e:
        print(f"❌ 基类测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("Hugging Face 嵌入模型修复测试")
    print("=" * 60)

    # 1. 测试基类接口
    print("\n1. 测试基类接口...")
    base_test = test_base_class()

    if not base_test:
        print("❌ 基类测试失败，请检查基类定义")
        return

    # 2. 测试Hugging Face嵌入模型
    print("\n2. 测试Hugging Face嵌入模型...")
    embedding_test = test_huggingface_embedding()

    if embedding_test:
        print("\n🎉 所有测试通过！现在可以在主程序中使用 Hugging Face 嵌入模型了。")
        print("\n使用方法：")
        print("1. 在 .env 文件中设置:")
        print("   EMBEDDING_PROVIDER=huggingface")
        print("   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2")
        print("\n2. 运行主程序:")
        print("   python main.py")
    else:
        print("\n❌ 测试失败，请检查错误信息并修复")


if __name__ == "__main__":
    main()