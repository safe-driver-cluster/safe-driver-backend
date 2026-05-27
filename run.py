import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        import uvicorn
        from main import app
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False
        )
    else:
        # Running in dev mode - use string import so reload works
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )