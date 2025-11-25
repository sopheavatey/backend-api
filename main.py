import uvicorn
from core.init_app import create_application
from core.config import settings

# Create the FastAPI application using the factory function
app = create_application()

if __name__ == "__main__":
    # Note: Use settings to dynamically get the project name if needed, 
    # but typically uvicorn run arguments are used directly.
    print(f"Starting {settings.PROJECT_NAME}...")
    uvicorn.run(app, host="0.0.0.0", port=8000)