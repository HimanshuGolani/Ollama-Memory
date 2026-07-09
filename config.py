from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_url: str = Field("http://localhost:11434", validation_alias="OLLAMA_URL")
    embed_model: str = Field("nomic-embed-text", validation_alias="OLLAMA_BRAIN_EMBED_MODEL")
    chat_model: str = Field("llama3.2", validation_alias="OLLAMA_BRAIN_CHAT_MODEL")
    server_port: int = Field(11435, validation_alias="OLLAMA_BRAIN_PORT")
    data_dir: Path = Field(Path.home() / ".ollama-brain", validation_alias="OLLAMA_BRAIN_DATA_DIR")

    model_config = {"env_file": ".env", "populate_by_name": True}

settings = Settings()
