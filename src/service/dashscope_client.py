import logging
import dashscope
from src.config import settings
from dashscope import Generation

logger = logging.getLogger(__name__)

dashscope.api_key = settings.DASHSCOPE_API_KEY

class DashscpoeClient:
    def __init__(self):
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("没有配置百炼PAI KEY")
    
    def call_llm(self, prompt:str) -> str:
        try:
            response = Generation.call(
                model = "qwen-plus",
                prompt = prompt,
                temperature = 0.7
            )

            if response.status_code == 200:
                logger.info("百炼API请求成功！")
                return response.output.text
            else:
                logger.error(f"百炼LLM调用失败：{response}")
                return "百炼LLM调用失败"

        except Exception as e:
            logger.exception(f"调用百炼API时出错{e}")
            raise