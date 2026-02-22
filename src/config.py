from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache

load_dotenv()

class Settings(BaseSettings):
    ENV: str = "dev"
    APP_NAME: str = "RAG Project"
    DASHSCOPE_API_KEY: str | None = None
    LOG_LEVEL: str = "INFO"

    @field_validator("ENV")
    @classmethod
    def validate_env(cls,v):
        if v not in {"dev","prod"}:
            raise ValueError("环境必须是env或者prod")
        return v
    
    @field_validator("DASHSCOPE_API_KEY")
    @classmethod
    def validate_api_key(cls,v,values):
        if values.data.get("ENV") == "prod" and not v:
            raise ValueError("生产环境必须要配置DASHSCOPO API KEY")
        return v
    
    def safe_dic(self):
        data = self.model_dump()
        if "DASHSCOPE_API_KEY" in data and data["DASHSCPOE"]:
            data["DASHSCOPE_API_KEY"] = "***"
        return data
    
@lru_cache
def get_settings():
    return Settings()
    
settings = get_settings()