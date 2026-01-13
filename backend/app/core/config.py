"""
Application Configuration: Centralized settings and path management.
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import List


class Settings:
    """Application settings with environment variable support."""
    
    # Base directories
    BACKEND_DIR: Path = Path(__file__).parent.parent.parent.resolve()
    DATA_DIR: Path = BACKEND_DIR / "data"
    RUNS_DIR: Path = BACKEND_DIR / "runs"
    TMP_R_DIR: Path = BACKEND_DIR / "tmp_r"
    
    # API settings
    API_TITLE: str = "ModelLab API"
    API_VERSION: str = "v1"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Metadata storage
    METADATA_FILE: Path = DATA_DIR / "metadata.json"
    
    def __init__(self):
        """Ensure required directories exist on initialization."""
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directories if they don't exist."""
        for directory in [self.DATA_DIR, self.RUNS_DIR, self.TMP_R_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()
