from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Use Pydantic BaseSettings for robust settings management
# Pydantic will automatically read from environment variables.
class Settings(BaseSettings):
    # Core App Settings
    PROJECT_NAME: str = "OCR Upload API"
    CORS_ORIGINS: list[str] = ["*"]

    # DigitalOcean Spaces Configuration
    SPACES_REGION: str
    SPACES_NAME: str
    SPACES_ENDPOINT: str
    ACCESS_KEY: str
    SECRET_KEY: str

    # OCR Model Paths
    YOLO_MODEL_PATH: str = "ai_model/yolo_best.onnx"
    CRNN_MODEL_PATH: str = "ai_model/crnn_best.onnx"
    OCR_MODE: str = "production" # e.g., 'dev' or 'prod' (pytorch or onnx)

    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL")

    # JWT Authentication Settings
    JWT_SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30

    # Database Configuration - using the exact env variable names from database.py
    DB_USER: str = os.getenv("username")
    DB_PASSWORD: str = os.getenv("password")
    DB_HOST: str = os.getenv("host")
    DB_PORT: str = os.getenv("port")
    DB_NAME: str = os.getenv("database")
    DB_SSLMODE: str = os.getenv("sslmode", "require")
    
    # Pre-calculate the full URL outside of the class if necessary for SQLAlchemy
    @property
    def DATABASE_URL(self):
        return f"postgresql://doadmin:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode={self.DB_SSLMODE}"
        # return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode={self.DB_SSLMODE}"

settings = Settings()