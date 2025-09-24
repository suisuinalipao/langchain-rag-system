import os
import subprocess
import sys


def setup_and_run():
    print("正在设置环境...")

    # 设置环境变量
    os.environ['EMBEDDING_MODEL'] = 'sentence-transformers/all-MiniLM-L6-v2'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

    print("环境变量已设置")

    # 安装依赖
    try:
        print("检查依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "sentence-transformers",
                               "langchain-huggingface"])
    except:
        print("依赖安装可能失败，但继续运行...")

    # 运行主程序
    print("启动RAG系统...")
    try:
        from main import main
        main()
    except Exception as e:
        print(f"运行失败：{e}")


if __name__ == "__main__":
    setup_and_run()