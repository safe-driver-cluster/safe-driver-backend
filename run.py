import uvicorn
from main import app  # Import directly instead of using a string

if __name__ == "__main__":
    uvicorn.run(
        app,           # Pass the app object directly, not "main:app" string
        host="0.0.0.0",
        port=8000,
        reload=False
    )