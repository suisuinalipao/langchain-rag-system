import logging
import os
from datetime import datetime

def setup_logger(name:str = "rag_systm" , level :str = "INFO")->logging.Logger:
    # 如果设置level="INFO"，只会记录INFO及以上级别的日志
    """
    设置日志器
    """
    logger = logging.getLogger(name)  # 想象Logger是一个"广播站"，Handlers是"接收器"
    if logger.handlers:
        return logger
    # 第一次调用
    # logger1 = setup_logger("rag_system")  # 完整配置
    # print(len(logger1.handlers))  # 输出: 2 (文件+控制台处理器)

    # 第二次调用同样的名字
    # logger2 = setup_logger("rag_system")  # 直接返回，不重复配置
    # print(logger1 is logger2)  # 输出: True (同一个对象)

    logger.setLevel(getattr(logging, level.upper()))# 等价于以下操作 level = "INFO" level_value = getattr(logging, level.upper())
    # 为什么不直接写
    # logging.INFO？
    # 因为level是一个字符串参数，需要动态获取对应的常量

    # 示例
    # getattr(logging, "DEBUG")  # 返回 10
    # getattr(logging, "INFO")  # 返回 20
    # getattr(logging, "ERROR")  # 返回 40
    # 创建日志目录
    log_dir ="logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #文件处理器""
    file_handler = logging.FileHandler(       # 文件接收器
        os.path.join(log_dir,f"rag_system_{datetime.now().strftime('%Y-%m-%d')}.log"),
        encoding = 'utf-8'
    )
    # your_project /
    # ├── main.py
    # ├── logs /  # 这个目录会被自动创建
    # │   ├── rag_system_20241215.log
    # │   ├── rag_system_20241216.log
    # │   └── rag_system_20241217.log
    # └── ...

    # 控制台处理器
    console_handler = logging.StreamHandler()  # 控制台接收器
    # 设置日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler) # 现在日志会发送到文件
    logger.addHandler(console_handler) # 现在日志也会发送到控制台

    return logger

# 同一条日志会被发送到两个地方：
# logger.info("用户提问: 什么是LangChain?")

# 1. 保存到文件 logs/rag_system_20241215.log
# 2. 显示在控制台终端上

# 用户看到终端输出，开发者可以查看历史文件


def get_logger(name:str = "rag_systm")->logging.Logger:
    """
    获取日志器
    """
    return logging.getLogger(name)



































