from typing import Optional, List
import requests
import os

from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

from src.service.dashscope_client import DashScopeClient


class DashScopeLLM(LLM):
    model: str = "qwen-plus"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "dashscope"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        client = DashScopeClient(
            model = self.model,
            temperature=self.temperature
        )

        return client.generate(prompt)