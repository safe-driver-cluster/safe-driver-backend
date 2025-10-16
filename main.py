from fastapi import FastAPI, HTTPException
import subprocess
import os
import sys
import logging
import asyncio
import json
import service.model_service as model_service
from database import db_helper
from beans.bean import ApiResponse, ResponseData, BehaviorResponseData

import firebase_admin
from firebase_admin import credentials, db

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

logger.info("=" * 80)
logger.info("SafeDriver Backend Starting...")
logger.info("=" * 80)

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

# Store the detect.py process and monitoring task
detect_process = None
monitor_task = None
device_mac = None

# Store latest behavior data in memory
latest_behavior_data = {
    "tag": None,
    "type": None,
    "message": None,
    "time": None,
    "data": None
}

async def read_detect_process_output():
    """Read and process output from detect.py subprocess"""
    global detect_process, latest_behavior_data, device_mac
    
    if not detect_process:
        logger.error("detect_process is not initialized")
        return
    
    logger.info("Started monitoring detect.py output")
    
    try:
        while True:
            # Check if process is still running
            if detect_process.poll() is not None:
                logger.warning(f"detect.py process terminated with code {detect_process.returncode}")
                break
            
            # Read line from stdout (already decoded in text mode)
            line = detect_process.stdout.readline()
            
            if line:
                line = line.strip()
                
                # Check if it's a behavior data message
                if line.startswith("BEHAVIOR_DATA:"):
                    try:
                        json_str = line.replace("BEHAVIOR_DATA:", "")
                        behavior_message = json.loads(json_str)
                        
                        # Update latest behavior data
                        latest_behavior_data = behavior_message
                        
                        logger.info(f"Received behavior data: tag={behavior_message.get('tag')}, "
                                  f"type={behavior_message.get('type')}, "
                                  f"message={behavior_message.get('message')}")
                        
                        # Save to Firebase
                        if device_mac:
                            db_helper.save_behavior_to_firebase(device_mac, behavior_message)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse behavior data: {e}")
                    except Exception as e:
                        logger.error(f"Error processing behavior data: {e}", exc_info=True)
                else:
                    # Regular log output from detect.py
                    logger.debug(f"detect.py: {line}")
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
    except Exception as e:
        logger.error(f"Error reading detect process output: {e}", exc_info=True)
    finally:
        logger.info("Stopped monitoring detect.py output")


@app.on_event("startup")
async def startup_event():
    global detect_process, device_mac

    # Get device MAC address
    try:
        device_details = model_service.check_device_registration()
        device_mac = device_details.get("mac_address")
        
        if not device_mac:
            logger.error("Could not retrieve device MAC address")
        else:
            logger.info(f"Device MAC: {device_mac}")
            
            # Register device if not already registered
            if not device_details.get("registered"):
                result = model_service.register_device(device_details)
                logger.info(f"Device registration result: {result}")
            else:
                model_service.update_device_status(status="active")
                
        
            # Start detect.py subprocess
            # Path to the same Python executable used by the current venv
            venv_python = sys.executable  

            # Start detect.py using that interpreter
            detect_process = subprocess.Popen(
                [venv_python, "-c", "from model.detect import main; main()"],
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            print(f"Started detect.py with PID: {detect_process.pid}")
            
    except Exception as e:
        logger.error(f"Failed to check device registration: {e}", exc_info=True)
        shutdown_event()



@app.on_event("shutdown")
async def shutdown_event():
    global detect_process, monitor_task
    logger.info("Shutting down SafeDriver Backend")
    
    # Cancel monitoring task
    if monitor_task:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled")
    
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
    return {"message": "SafeDriver API is running"}


@app.get("/behavior/latest")
async def get_latest_behavior():
    """Get the latest behavior data from detect process"""
    global latest_behavior_data, device_mac
    
    if not latest_behavior_data.get("data"):
        return {
            "success": False,
            "message": "No behavior data available yet"
        }
    
    return {
        "success": True,
        "mac_address": device_mac,
        "behavior": latest_behavior_data
    }


@app.get("/behavior/history")
async def get_behavior_history(limit: int = 10):
    """Get behavior event history from Firebase"""
    global device_mac
    
    if not device_mac:
        raise HTTPException(status_code=400, detail="Device MAC not available")
    
    try:
        ref = db.reference(f'behavior_events/{device_mac}/history')
        history_data = ref.order_by_key().limit_to_last(limit).get()
        
        if history_data:
            events = [{"id": k, **v} for k, v in history_data.items()]
            return {
                "success": True,
                "mac_address": device_mac,
                "count": len(events),
                "events": events
            }
        else:
            return {
                "success": False,
                "message": "No behavior history available"
            }
            
    except Exception as e:
        logger.error(f"Error retrieving behavior history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/process/status")
def check_process_status() -> dict:
    """Check if detect.py process is running"""
    global detect_process
    
    if detect_process is None:
        return {
            "running": False,
            "message": "Process not started"
        }
    
    return_code = detect_process.poll()
    
    return {
        "running": return_code is None,
        "pid": detect_process.pid,
        "exit_code": return_code if return_code is not None else None
    }


@app.get("/about")
def about() -> dict[str, str]:
    logger.debug("About endpoint accessed")
    return {
        "message": "SafeDriver Backend API",
        "version": "1.0.0",
        "device_mac": device_mac
    }

