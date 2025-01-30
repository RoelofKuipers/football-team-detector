from pathlib import Path
from pydantic_settings import BaseSettings
import tempfile


class Settings(BaseSettings):
    MODEL_PATH: str = "checkpoints/yolo_football.pt"
    TEMP_BASE_DIR: Path = Path(tempfile.mkdtemp(prefix="football_api_"))
    OUTPUT_DIR: Path = Path("output")

    def cleanup_temp_dir(self):
        """Remove all temporary files"""
        if self.TEMP_BASE_DIR.exists():
            shutil.rmtree(self.TEMP_BASE_DIR)

    class Config:
        env_prefix = "APP_"


settings = Settings()

# Register cleanup on exit
import atexit
import shutil

atexit.register(settings.cleanup_temp_dir)
