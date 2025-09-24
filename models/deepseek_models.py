import time
from typing import List, Dict, Any
from openai import OpenAI
from config.settings import ModelConfig
from models.base import BaseEmbedding, BaseLLM

class DeepSeekLLM(BaseLLM):
    """
    DeepSeek LLM 实现（使用 OpenAI Python SDK 指向 DeepSeek endpoint）
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        # 推荐 base_url 不带 /v1；/v1 也可用，但通常使用 https://api.deepseek.com
        self.client = OpenAI(
            api_key=config.api_key,
            base_url="https://api.deepseek.com"
        )

    def _chat_completion(self, messages: list, model: str = "deepseek-chat", **kwargs) -> str:
        """
        发送 chat completion 请求并返回文本内容（封装了请求与常见错误处理）
        """
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            # DeepSeek/OpenAI-style 响应通常在 choices[0].message.content
            return resp.choices[0].message.content
        except Exception as e:
            # 打印完整返回便于调试（可以在调试时打开）
            print("DeepSeek 请求异常：", e)
            # 若希望打印原始响应结构以便定位问题，可取消下面注释（慎用生产）
            # import traceback; traceback.print_exc()
            return "抱歉，生成回答时出现错误。"

    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成回答（简单 prompt -> chat 格式）
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return self._chat_completion(messages, **kwargs)

    def generate_with_context(self, question: str, context: str, **kwargs) -> str:
        """
        基于上下文生成回答
        """
        prompt = (
            "基于以下上下文信息回答用户问题。如果文档中没有相关信息，请明确说明。\n\n"
            f"上下文信息:\n{context}\n\n用户问题:\n{question}"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return self._chat_completion(messages, **kwargs)
