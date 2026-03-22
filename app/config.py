"""Application configuration."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

OUTPUTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
    
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    INTRADAY_CACHE_TTL = int(os.getenv("INTRADAY_CACHE_TTL", "60"))
    
    MAX_CODE_EXECUTION_TIME = int(os.getenv("MAX_CODE_EXECUTION_TIME", "20"))
    MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "512"))
    
    DB_PATH = str(BASE_DIR / "data" / "stock_agent.db")
    
    APPROVED_IMPORTS = [
        "pandas", "numpy", "yfinance", "plotly", "ta", "pandas_ta",
        "scipy", "math", "statistics", "datetime", "typing"
    ]
    
    BASE_DIR = BASE_DIR
    OUTPUTS_DIR = OUTPUTS_DIR
    DATA_DIR = DATA_DIR

config = Config()
