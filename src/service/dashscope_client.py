import logging
from dashscope import Generation
from src.config import settings

logger = logging.getLogger(__name__)


class DashScopeClient:
    def __init__(self, model="qwen-plus", temperature=0.7):
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("没有配置百炼 API KEY")

        self.api_key = settings.DASHSCOPE_API_KEY
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        try:
            response = Generation.call(
                api_key=self.api_key,
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
            )

            if response.status_code != 200:
                logger.error(f"百炼调用失败: {response}")
                raise RuntimeError("百炼 LLM 调用失败")

            logger.info("百炼API请求成功")

            return response.output["text"]

        except Exception as e:
            logger.exception("调用百炼API时出错")
            raise