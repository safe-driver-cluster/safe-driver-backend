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
import utils.utils as utils

import firebase_admin
from firebase_admin import credentials, db

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safe_driver_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create a separate logger for detect.py output (console only, no file)
detect_logger = logging.getLogger('detect_output')
detect_logger.setLevel(logging.INFO)
detect_console_handler = logging.StreamHandler(sys.stdout)
detect_console_handler.setFormatter(logging.Formatter('%(message)s'))
detect_logger.addHandler(detect_console_handler)
detect_logger.propagate = False  # Don't propagate to root logger (prevents duplicate logs)

utils.print_banner(logger)
logger.info("=" * 80)
logger.info("SafeDriver Monitoring System Starting...")
logger.info("=" * 80)

# ============================================================================
# END LOGGING CONFIGURATION
# ============================================================================

# Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger.info("Initializing Firebase Admin SDK")
try:
    # Check if Firebase app is already initialized
    firebase_admin.get_app()
    logger.info("Firebase Admin SDK already initialized")
except ValueError:
    # Initialize Firebase if not already done
    cred = credentials.Certificate("firebase-admin-sdk/safe-driver-system-firebase-adminsdk-fbsvc-76241499ba.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://safe-driver-system-default-rtdb.firebaseio.com/'
    })
    logger.info("Firebase Admin SDK initialized successfully")

# Import firestore_helper after Firebase is initialized
from database.firestore_helper import firestore_helper
logger.info("Imported firestore_helper module")

# Initialize FastAPI app
app = FastAPI()

# Store the detect.py process and monitoring tasks
detect_process = None
monitor_task = None
stderr_task = None
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
    
    logger.info("Started monitoring detect.py output (stdout)")
    
    try:
        while True:
            # Check if process is still running
            if detect_process.poll() is not None:
                logger.warning(f"detect.py process terminated with code {detect_process.returncode}")
                break
            
            # Read line from stdout
            line = await asyncio.get_event_loop().run_in_executor(
                None, detect_process.stdout.readline
            )
            
            if line:
                line = line.strip()
                
                # Check if it's a behavior data message
                if line.startswith("BEHAVIOR_DATA:"):
                    try:
                        json_str = line.replace("BEHAVIOR_DATA:", "", 1)
                        behavior_message = json.loads(json_str)
                        
                        # Update latest behavior data
                        latest_behavior_data = behavior_message
                        
                        logger.info(f"Behavior Event: {behavior_message.get('type')} - {behavior_message.get('message')}")
                        
                        # Save to Firebase
                        if device_mac:
                            db_helper.save_behavior_to_firebase(device_mac, behavior_message)
                        else:
                            logger.warning("Cannot save behavior event - device MAC not available")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse behavior data: {e}")
                        logger.error(f"Raw line: {line}")
                    except Exception as e:
                        logger.error(f"Error processing behavior data: {e}", exc_info=True)
            
            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
            
    except Exception as e:
        logger.error(f"Error reading detect process stdout: {e}", exc_info=True)
    finally:
        logger.info("Stopped monitoring detect.py output")
        model_service.update_device_status(status="inactive")
        logger.info("Device status updated to inactive")


async def read_detect_process_stderr():
    """Read and log stderr output from detect.py subprocess"""
    global detect_process
    
    if not detect_process:
        return
    
    logger.info("Started monitoring detect.py stderr")
    
    try:
        while True:
            if detect_process.poll() is not None:
                break
            
            line = await asyncio.get_event_loop().run_in_executor(
                None, detect_process.stderr.readline
            )
            
            # if line:
            #     line = line.strip()
            #     if line:
            #         # Extract just the message part after the log level
            #         # Format: 2025-10-17 11:17:30,966 - model.detect - INFO - Message
            #         parts = line.split(' - ', 3)
            #         if len(parts) >= 4:
            #             # Get just the message (last part)
            #             message = parts[3]
            #             detect_logger.info(f"{message}")
            #         else:
            #             # If format doesn't match, log as-is
            #             detect_logger.info(f"{line}")

            if line:
                line = line.strip()
                if line:
                    # Log detect.py messages with prefix
                    detect_logger.info(f"{line}")
            
            await asyncio.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Error reading detect process stderr: {e}", exc_info=True)
    finally:
        logger.info("Stopped monitoring detect.py stderr")


@app.on_event("startup")
async def startup_event():
    global detect_process, monitor_task, stderr_task, device_mac

    logger.info("=" * 80)
    logger.info("SafeDriver Backend Starting...")
    logger.info("=" * 80)

    # Get device MAC address
    try:
        device_details = model_service.check_device_registration()
        device_mac = device_details.get("mac_address")
        
        if not device_mac:
            logger.error("Could not retrieve device MAC address")
        else:
            # logger.info(f"Device MAC: {device_mac}")
            
            # Register device if not already registered
            if not device_details.get("registered"):
                result = model_service.register_device(device_details)
                logger.info(f"Device registration result: {result}")
            else:
                model_service.update_device_status(status="active")
                logger.info("Device status updated to active")
                
    except Exception as e:
        logger.error(f"Failed to check device registration: {e}", exc_info=True)

    # Path to the same Python executable used by the current venv
    venv_python = sys.executable  
    logger.info(f"Using Python executable: {venv_python}")

    try:
        # Start detect.py with stdout and stderr piped
        detect_process = subprocess.Popen(
            [venv_python, "-c", "from model.detect import main; main()"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # Text mode for automatic line handling
            bufsize=1  # Line buffered
        )
        logger.info(f"Started detect.py with PID: {detect_process.pid}")
        
        # Start monitoring tasks
        monitor_task = asyncio.create_task(read_detect_process_output())
        stderr_task = asyncio.create_task(read_detect_process_stderr())
        logger.info("Started output monitoring tasks")
        
    except Exception as e:
        logger.error(f"Failed to start detect.py: {e}", exc_info=True)
        if device_mac:
            model_service.update_device_status(status="inactive")
            logger.info("Device status updated to inactive")


@app.on_event("shutdown")
async def shutdown_event():
    global detect_process, monitor_task, stderr_task, device_mac
    
    logger.info("Shutting down SafeDriver Backend")
    
    # Cancel monitoring tasks
    if monitor_task:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            logger.info("Stdout monitoring task cancelled")
    
    if stderr_task:
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            logger.info("Stderr monitoring task cancelled")
    
    # Stop detect.py when the API shuts down
    if detect_process:
        try:
            detect_process.terminate()
            detect_process.wait(timeout=5)
            logger.info("Stopped detect.py successfully")
        except Exception as e:
            logger.error(f"Error stopping detect.py: {e}", exc_info=True)
            detect_process.kill()  # Force kill if terminate fails
    
    # Update device status to inactive
    if device_mac:
        try:
            model_service.update_device_status(status="inactive")
            logger.info("Device status updated to inactive")
        except Exception as e:
            logger.error(f"Failed to update device status: {e}")


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "SafeDriver API is running",
        "device_mac": device_mac,
        "process_running": detect_process.poll() is None if detect_process else False,
        "config_loaded": config_load_result['success'],
        "config_count": config_load_result['updated_count']
    }


@app.get("/config/status")
def get_configuration_status():
    """Get configuration loading status from startup"""
    return {
        "success": True,
        "message": "Configuration status retrieved",
        "data": config_load_result
    }


@app.get("/behavior/latest")
async def get_latest_behavior():
    """Get the latest behavior data from detect process"""
    global latest_behavior_data, device_mac
    
    if not latest_behavior_data.get("type"):
        return {
            "success": False,
            "message": "No behavior data available yet",
            "data": None
        }
    
    return {
        "success": True,
        "message": "Latest behavior data retrieved",
        "data": {
            "mac_address": device_mac,
            **latest_behavior_data
        }
    }


@app.get("/behavior/history")
async def get_behavior_history(limit: int = 10):
    """Get behavior event history from Firebase"""
    global device_mac
    
    if not device_mac:
        return {
            "success": False,
            "message": "Device MAC not available",
            "data": None
        }
    
    try:
        ref = db.reference(f'alerts/{device_mac}/history')
        history_data = ref.order_by_key().limit_to_last(limit).get()
        
        if history_data:
            events = [{"id": k, **v} for k, v in history_data.items()]
            return {
                "success": True,
                "message": f"Retrieved {len(events)} behavior events",
                "data": {
                    "mac_address": device_mac,
                    "count": len(events),
                    "events": events
                }
            }
        else:
            return {
                "success": False,
                "message": "No behavior history available",
                "data": None
            }
            
    except Exception as e:
        logger.error(f"Error retrieving behavior history: {e}", exc_info=True)
        return {
            "success": False,
            "message": str(e),
            "data": None
        }


@app.get("/process/status")
def check_process_status():
    """Check if detect.py process is running"""
    global detect_process
    
    if detect_process is None:
        return {
            "success": False,
            "message": "Process not started",
            "running": False
        }
    
    return_code = detect_process.poll()
    is_running = return_code is None
    
    return {
        "success": is_running,
        "message": "Process is running" if is_running else f"Process terminated with code {return_code}",
        "running": is_running,
        "pid": detect_process.pid,
        "exit_code": return_code
    }


@app.post("/config/save")
def save_model_configurations():
    """Save all model configurations to Firestore"""
    try:
        result = firestore_helper.save_model_configurations_to_firestore()
        return result
    except Exception as e:
        logger.error(f"Error in save configurations endpoint: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@app.get("/config/get")
def get_model_configurations():
    """Get model configurations from Firestore"""
    try:
        result = firestore_helper.get_model_configurations_from_firestore()
        if result:
            # save to local config as well
            utils.update_local_config_from_firestore(result)
            return {
                "success": True,
                "message": "Configurations retrieved successfully",
                "data": result
            }
        else:
            return {
                "success": False,
                "message": "No configurations found",
                "data": None
            }
    except Exception as e:
        logger.error(f"Error in get configurations endpoint: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": None
        }


@app.put("/config/update")
def update_specific_configuration(
    # config_category: str,
    config_name: str,
    config_value: str
):
    """Update a specific configuration value in Firestore"""
    try:
        # Try to convert the string value to appropriate type
        processed_value = config_value
        
        # Attempt type conversion based on common patterns
        if config_value.lower() in ['true', 'false']:
            processed_value = config_value.lower() == 'true'
        elif config_value.replace('.', '', 1).replace('-', '', 1).isdigit():
            processed_value = float(config_value) if '.' in config_value else int(config_value)
        elif config_value.startswith('[') and config_value.endswith(']'):
            try:
                import json
                processed_value = json.loads(config_value)
            except:
                pass  # Keep as string if JSON parsing fails
        
        # result = firestore_helper.update_specific_configuration_firestore(
        #     config_category, config_name, processed_value
        # )
        result = firestore_helper.update_specific_configuration_firestore(
            "config_category", config_name, processed_value
        )
        return result
    except Exception as e:
        logger.error(f"Error in update configuration endpoint: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@app.post("/process/restart")
async def restart_detection_process():
    """Restart the detection process to apply configuration changes"""
    global detect_process, monitor_task, stderr_task, device_mac
    
    try:
        logger.info("Restarting detection process to apply configuration changes...")
        
        # Stop current process
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                logger.info("Stdout monitoring task cancelled")
        
        if stderr_task:
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                logger.info("Stderr monitoring task cancelled")
        
        if detect_process:
            try:
                detect_process.terminate()
                detect_process.wait(timeout=5)
                logger.info("Stopped detect.py process")
            except Exception as e:
                logger.error(f"Error stopping detect.py: {e}")
                detect_process.kill()
        
        # Start new process
        venv_python = sys.executable
        detect_process = subprocess.Popen(
            [venv_python, "-c", "from model.detect import main; main()"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        logger.info(f"Restarted detect.py with PID: {detect_process.pid}")
        
        # Start monitoring tasks
        monitor_task = asyncio.create_task(read_detect_process_output())
        stderr_task = asyncio.create_task(read_detect_process_stderr())
        logger.info("Restarted output monitoring tasks")
        
        return {
            "success": True,
            "message": "Detection process restarted successfully",
            "new_pid": detect_process.pid
        }
        
    except Exception as e:
        logger.error(f"Error restarting detection process: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Failed to restart process: {str(e)}"
        }


@app.put("/config/update-and-restart")
async def update_configuration_and_restart(
    config_name: str,
    config_value: str
):
    """Update configuration and restart detection process if needed"""
    try:
        
        # First update the configuration
        processed_value = config_value
        
        # Type conversion
        if config_value.lower() in ['true', 'false']:
            processed_value = config_value.lower() == 'true'
        elif config_value.replace('.', '', 1).replace('-', '', 1).isdigit():
            processed_value = float(config_value) if '.' in config_value else int(config_value)
        elif config_value.startswith('[') and config_value.endswith(']'):
            try:
                import json
                processed_value = json.loads(config_value)
            except:
                pass
        
        # Update in Firestore
        result = firestore_helper.update_specific_configuration_firestore(
            "config_category", config_name, processed_value
        )
        
        if not result['success']:
            return result
        
        # Update local configuration
        import config.config as config
        setattr(config, config_name, processed_value)
        logger.info(f"Updated local configuration {config_name} = {processed_value}")
        
        logger.info(f"Configuration {config_name} requires process restart")
        restart_result = await restart_detection_process()
        
        return {
            "success": True,
            "message": f"Configuration updated and process restarted",
            "config_update": result,
            "process_restart": restart_result,
            "restart_required": True
        }
            
    except Exception as e:
        logger.error(f"Error in update-and-restart endpoint: {e}")
        return {
            "success": False,
            "message": str(e)
        }



