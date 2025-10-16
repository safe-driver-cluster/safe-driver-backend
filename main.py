from fastapi import FastAPI
from models import MsgPayload
import subprocess
import os
import sys
import logging
import service.model_service as model_service

import firebase_admin
from firebase_admin import credentials

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safe_driver_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# END LOGGING CONFIGURATION
# ============================================================================

# Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger.info("Initializing Firebase Admin SDK")
cred = credentials.Certificate("firebase-admin-sdk\safe-driver-system-firebase-adminsdk-fbsvc-76241499ba.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://safe-driver-system-default-rtdb.firebaseio.com/'
})
logger.info("Firebase Admin SDK initialized successfully")

app = FastAPI()

# Store the detect.py process
detect_process = None


@app.on_event("startup")
async def startup_event():
    global detect_process

    logger.info("=" * 80)
    logger.info("SafeDriver Backend Starting")
    logger.info("=" * 80)

    # Path to the same Python executable used by the current venv
    venv_python = sys.executable  
    logger.info(f"Using Python executable: {venv_python}")

    try:
        # Start detect.py using that interpreter
        detect_process = subprocess.Popen(
            [venv_python, "-c", "from model.detect import main; main()"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        logger.info(f"Started detect.py with PID: {detect_process.pid}")
    except Exception as e:
        logger.error(f"Failed to start detect.py: {e}", exc_info=True)

    try:
        device_details = model_service.check_device_registration()
        
        # register the device if not already registered
        if not device_details.get("registered"):
            result = model_service.register_device(device_details)
        else:
            result = model_service.update_device_status(
                mac=device_details.get("mac_address"), 
                status="active"
            )

    except Exception as e:
        logger.error(f"Failed to check device registration: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    global detect_process
    logger.info("Shutting down SafeDriver Backend")
    
    # Stop detect.py when the API shuts down
    if detect_process:
        try:
            detect_process.terminate()
            detect_process.wait()
            logger.info("Stopped detect.py successfully")
        except Exception as e:
            logger.error(f"Error stopping detect.py: {e}", exc_info=True)
    else:
        logger.warning("No detect.py process to stop")


@app.get("/")
def root() -> dict[str, str]:
    logger.debug("Root endpoint accessed")
    return {"message": "Hello"}


# About page route
@app.get("/about")
def about() -> dict[str, str]:
    logger.debug("About endpoint accessed")
    return {"message": "This is the about page."}


# Route to add a message
# @app.post("/messages/{msg_name}/")
# def add_msg(msg_name: str) -> dict[str, MsgPayload]:
#     # Generate an ID for the item based on the highest ID in the messages_list
#     msg_id = max(messages_list.keys()) + 1 if messages_list else 0
#     messages_list[msg_id] = MsgPayload(msg_id=msg_id, msg_name=msg_name)
#     logger.info(f"Message added: {msg_name} with ID: {msg_id}")
#     return {"message": messages_list[msg_id]}


# # Route to list all messages
# @app.get("/messages")
# def message_items() -> dict[str, dict[int, MsgPayload]]:
#     logger.debug("Messages endpoint accessed")
#     return {"messages:": messages_list}

