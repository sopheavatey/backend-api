from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from core.config import settings
from middleware.auth import auth_router, models as auth_models, database as auth_database
from routers import ocr_router, telegram_router

# Configure logging
logging.basicConfig(level=logging.INFO)

def create_application() -> FastAPI:
    """
    Creates and configures the FastAPI application instance.
    """
    app = FastAPI(title=settings.PROJECT_NAME)

    # 1. Initialize Database Tables (ensure all ORM models are registered)
    # This must be done on startup if you want FastAPI to manage it.
    auth_models.Base.metadata.create_all(bind=auth_database.engine)
    logging.info("Database tables initialized/checked.")

    # 2. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 3. Include Routers
    # Auth Router
    app.include_router(auth_router.router)
    # OCR/Upload Router
    app.include_router(ocr_router.router)

    app.include_router(auth_router.router)
    app.include_router(ocr_router.router)
    app.include_router(telegram_router.router)
    # 4. Root & Health Check Endpoints
    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {
            "message": "OCR Upload API is running",
            "status": "healthy",
            "spaces_configured": bool(settings.SPACES_NAME and settings.ACCESS_KEY),
            "ocr_mode": settings.OCR_MODE,
        }

    @app.get("/debug-config")
    async def debug_config():
        """Debug endpoint to check configuration (NEVER use in production!)"""
        return {
            "spaces_region": settings.SPACES_REGION,
            "spaces_name": settings.SPACES_NAME,
            "spaces_endpoint": settings.SPACES_ENDPOINT,
            "access_key_set": bool(settings.ACCESS_KEY),
            "secret_key_set": bool(settings.SECRET_KEY),
            "access_key_prefix": settings.ACCESS_KEY[:10] if settings.ACCESS_KEY else None
        }

    return app