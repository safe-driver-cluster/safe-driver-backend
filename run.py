import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import sys
import io

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

if __name__ == "__main__":

    try:
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
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)