import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.init_app import create_application
from services.telegram_service import ptb_application

# We need to wrap the app creation to use lifespan events for the Bot
# Instead of modifying init_app.py drastically, we can handle the lifespan here.

# 1. Define Lifespan (Startup/Shutdown logic)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🤖 Starting Telegram Bot...")
    await ptb_application.initialize()
    await ptb_application.start()
    await ptb_application.updater.start_polling()
    
    yield # Application runs here
    
    # --- SHUTDOWN ---
    print("🛑 Stopping Telegram Bot...")
    await ptb_application.updater.stop()
    await ptb_application.stop()
    await ptb_application.shutdown()

# 2. Initialize App
# We create the app first
app = create_application()
# Then we assign the lifespan we defined above
app.router.lifespan_context = lifespan

if __name__ == "__main__":
    from core.config import settings
    print(f"Starting {settings.PROJECT_NAME}...")
    uvicorn.run(app, host="0.0.0.0", port=8000)