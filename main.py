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
import config.config as config
from service.model_service import (get_mac_address_alternative)
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, db

import queue
import threading
from shared import behavior_queue, stop_event
import model.detect as detect

if(config.ENABLE_FINGERPRINT and config.SYSTEM != 'windows'):
    import fingerprint.enroll as enroll
    import fingerprint.remove as remove
    from fingerprint.live import main as live_main

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=open(sys.stdout.fileno(), 
                                             mode='w', 
                                             encoding='utf-8', 
                                             closefd=False)),
        logging.FileHandler('safe_driver_debug.log', encoding='utf-8')
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

# utils.print_banner(logger)
# logger.info("=" * 80)
# logger.info("SafeDriver Monitoring System Starting...")
# logger.info("=" * 80)

# ============================================================================
# END LOGGING CONFIGURATION
# ============================================================================

# Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
load_dotenv(utils.resource_path(".env"))
admin_sdk_path = os.getenv("ADMIN_SDK_PATH", "firebase-admin-sdk/serviceAccountKey.json")
ADMIN_SDK_PATH = (
    admin_sdk_path
    if os.path.isabs(admin_sdk_path)
    else utils.resource_path(admin_sdk_path)
)

logger.info("Initializing Firebase Admin SDK")
try:
    # Check if Firebase app is already initialized
    firebase_admin.get_app()
    logger.info("Firebase Admin SDK already initialized")
except ValueError:
    # Initialize Firebase if not already done
    cred = credentials.Certificate(ADMIN_SDK_PATH) # safe-driver-system-b3da24192be1
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://safe-driver-system-default-rtdb.firebaseio.com/'
    })
    logger.info("Firebase Admin SDK initialized successfully")


# Import firestore_helper after Firebase is initialized
from database.firestore_helper import firestore_helper
# logger.info("Imported firestore_helper module")

# Initialize FastAPI app
app = FastAPI()

# Store the detect.py process and monitoring tasks
detect_process = None
monitor_task = None
stderr_task = None
device_mac = None
current_cap = None
watchdog_task = None
fingerprint_live = None
gps_thread = None
gps_stop_event = threading.Event()

# Store latest behavior data in memory
latest_behavior_data = {
    "tag": None,
    "type": None,
    "message": None,
    "time": None,
    "data": None
}

# async def read_detect_process_output():
#     """Read and process output from detect.py subprocess"""
#     global detect_process, latest_behavior_data, device_mac
    
#     if not detect_process:
#         logger.error("detect_process is not initialized")
#         return
    
#     logger.info("Started monitoring detect.py output (stdout)")
    
#     try:
#         while True:
#             # Check if process is still running
#             if detect_process.poll() is not None:
#                 logger.warning(f"detect.py process terminated with code {detect_process.returncode}")
#                 break
            
#             # Read line from stdout
#             line = await asyncio.get_event_loop().run_in_executor(
#                 None, detect_process.stdout.readline
#             )
            
#             if line:
#                 line = line.strip()
                
#                 # Check if it's a behavior data message
#                 if line.startswith("BEHAVIOR_DATA:"):
#                     try:
#                         json_str = line.replace("BEHAVIOR_DATA:", "", 1)
#                         behavior_message = json.loads(json_str)
                        
#                         # Update latest behavior data
#                         latest_behavior_data = behavior_message
                        
#                         logger.info(f"Behavior Event: {behavior_message.get('type')} - {behavior_message.get('message')}")
                        
#                         # Save to Firebase
#                         if device_mac:
#                             db_helper.save_behavior_to_firebase(device_mac, behavior_message)
#                         else:
#                             logger.warning("Cannot save behavior event - device MAC not available")
                        
#                     except json.JSONDecodeError as e:
#                         logger.error(f"Failed to parse behavior data: {e}")
#                         logger.error(f"Raw line: {line}")
#                     except Exception as e:
#                         logger.error(f"Error processing behavior data: {e}", exc_info=True)
            
#             await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
            
#     except Exception as e:
#         logger.error(f"Error reading detect process stdout: {e}", exc_info=True)
#     finally:
#         logger.info("Stopped monitoring detect.py output")
#         model_service.update_device_status(status="offline")
#         logger.info("Device status updated to offline")

async def watchdog():
    """Monitor detect thread and restart if it crashes"""
    global detect_process, monitor_task

    logger.info("Watchdog started - monitoring detect thread")

    while not stop_event.is_set():
        await asyncio.sleep(5)  # Check every 5 seconds

        if stop_event.is_set():
            break

        if detect_process is not None and not detect_process.is_alive():
            logger.warning("Detect thread crashed! Restarting...")

            # Cancel old monitor task
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            # Clear queue
            while not behavior_queue.empty():
                try:
                    behavior_queue.get_nowait()
                except queue.Empty:
                    break

            # Reset stop event in case it was set
            stop_event.clear()

            # Start new detect thread
            detect_process = threading.Thread(
                target=detect.main,
                daemon=True,
                name="detect-thread"
            )
            detect_process.start()
            logger.info(f"Detect thread restarted: {detect_process.name}")

            # Restart monitor task
            monitor_task = asyncio.create_task(read_behavior_queue())
            logger.info("Behavior queue monitor restarted")

            if device_mac:
                model_service.update_device_status(status="online")
                logger.info("Device status updated to online after watchdog restart")

    logger.info("Watchdog stopped")

async def read_behavior_queue():
    global latest_behavior_data, device_mac

    logger.info("Started monitoring behavior queue")

    try:
        while True:
            try:
                payload = behavior_queue.get_nowait()

                logger.info(f"Behavior Event: {payload.get('type')} - {payload.get('message')}")

                if device_mac:
                    evidence_jpeg = payload.pop("_evidence_jpeg", None)
                    if evidence_jpeg:
                        try:
                            from database.storage_helper import upload_alert_evidence

                            evidence = await asyncio.to_thread(
                                upload_alert_evidence,
                                evidence_jpeg,
                                device_mac,
                                payload.get("type"),
                            )
                            if evidence:
                                payload["evidence"] = evidence["url"]
                                payload["evidence_path"] = evidence["path"]
                        except Exception:
                            logger.exception(
                                "Failed to upload alert evidence for %s",
                                payload.get("type"),
                            )

                    latest_behavior_data = payload
                    db_helper.save_behavior_to_firebase(device_mac, payload)
                else:
                    payload.pop("_evidence_jpeg", None)
                    latest_behavior_data = payload
                    logger.warning("Cannot save behavior - device MAC not available")

            except queue.Empty:
                pass  # No data yet, try again next loop

            await asyncio.sleep(0.01)

    except Exception as e:
        logger.error(f"Error reading behavior queue: {e}", exc_info=True)
    finally:
        logger.info("Stopped monitoring behavior queue")
        if not stop_event.is_set():
            model_service.update_device_status(status="offline")


async def read_detect_process_stderr():
    """Read and log stderr output from detect.py subprocess"""
    global detect_process
    
    if not detect_process:
        return
    
    logger.info("Started monitoring detect.py stderr")
    
    try:
        while True:
            if detect_process.poll() is not None:
                logger.warning(f"detect.py process terminated with code {detect_process.returncode}")
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
    global detect_process, monitor_task, stderr_task, device_mac, watchdog_task, fingerprint_live, gps_thread

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
        cred = credentials.Certificate(utils.resource_path("firebase-admin-sdk/serviceAccountKey.json"))  # safe-driver-system-b3da24192be1
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://safe-driver-system-default-rtdb.firebaseio.com/'
        })
        logger.info("Firebase Admin SDK initialized successfully")

    if config.ENABLE_VOICE_ALERT_SYNC:
        try:
            from database.storage_helper import sync_voice_alerts_from_storage

            logger.info("Synchronizing voice-alert audio from Firebase Storage...")
            sync_result = await asyncio.to_thread(sync_voice_alerts_from_storage)
            logger.info(
                "Voice-alert audio synchronized: %s downloaded/replaced, %s failed",
                sync_result["downloaded"],
                sync_result["failed"],
            )
        except Exception:
            logger.exception("Voice-alert audio sync failed; continuing with local audio files")

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
                model_service.update_device_status(status="starting")
                logger.info("Device status updated to starting")

            # Check if vehicle_reg_no is set for the device
            device_reg_no_check = model_service.check_vehicle_registration(device_mac)
            if not device_reg_no_check:
                logger.warning(f"Vehicle registration number not set for device {device_mac}")

                # get vehicle number plate from firestore using device_mac
                vehicle_number_plate = db_helper.get_vehicle_number_plate_from_firestore(device_mac)
                if vehicle_number_plate:
                    logger.info(f"Retrieved vehicle registration number from Firestore: {vehicle_number_plate}")
                    # Update in realtime database as well
                    update_result = db_helper.update_vehicle_reg_no(device_mac, vehicle_number_plate)
                    logger.info(f"Updated vehicle registration number in Realtime Database: {update_result}")
                else:
                    logger.warning(f"Vehicle registration number not found in Firestore for device {device_mac}")
            else:
                logger.info(f"Vehicle registration number already set for device {device_mac}")
                logger.info(f"Vehicle registration number for device {device_mac}: {db_helper.get_vehicle_reg_no(device_mac)}")

                # get assigned driver id from realtime database using device_mac
                assigned_driver_id = db_helper.get_assigned_driver(device_mac)
                logger.info(f"Assigned driver ID for device {device_mac}: {assigned_driver_id}")

                # update assigned driver
                db_helper.update_assigned_driver(None, get_mac_address_alternative().upper())
                logger.info(f'Updated assigned driver to {None} for device {get_mac_address_alternative().upper()}')
                
    except Exception as e:
        logger.error(f"Failed to check device registration: {e}", exc_info=True)

    # Path to the same Python executable used by the current venv
    # venv_python = sys.executable 
    venv_python = "/home/safedriver/Desktop/safe-driver-backend/venv/bin/python" 
    logger.info(f"Using Python executable: {venv_python}")

    try:
        # Start detect.py with stdout and stderr piped
        # detect_process = subprocess.Popen(
        #     [venv_python, "-c", "from model.detect import main; main()"],
        #     cwd=os.path.dirname(os.path.abspath(__file__)),
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     universal_newlines=True,  # Text mode for automatic line handling
        #     bufsize=1  # Line buffered
        # )
        # logger.info(f"Started detect.py with PID: {detect_process.pid}")

        if config.ENABLE_DETECTION:
            detect_process = threading.Thread(
                target=detect.main,
                daemon=True,
                name="detect-thread"
            )
            detect_process.start()
            logger.info(f"Started detect thread: {detect_process.name}")
        
        # Start monitoring tasks
        # monitor_task = asyncio.create_task(read_detect_process_output())
        # stderr_task = asyncio.create_task(read_detect_process_stderr())

        monitor_task = asyncio.create_task(read_behavior_queue())

        logger.info("Started output monitoring tasks")

        # ── Start watchdog ──────────────────────────────────────────────
        # watchdog_task = asyncio.create_task(watchdog())
        # logger.info("Watchdog started")
        # ───────────────────────────────────────────────────────────────

        model_service.update_device_status(status="online")
        logger.info("Device status updated to online")

        # FINGERPRINT ENROLLMENT TEST
        if(config.ENABLE_FINGERPRINT and config.SYSTEM == 'linux'):
            fingerprint_live = threading.Thread(
                target=live_main,
                daemon=True,
                name="fingerprint-live-thread"
            )
            fingerprint_live.start()
            logger.info(f"Started fingerprint live thread: {fingerprint_live.name}")

        if config.ENABLE_GPS and sys.platform.startswith("linux"):
            try:
                from gps.gps import run_gps_loop

                gps_stop_event.clear()
                gps_thread = threading.Thread(
                    target=run_gps_loop,
                    kwargs={
                        "stop_event": gps_stop_event,
                        "device_mac": device_mac,
                        "port": config.GPS_SERIAL_PORT,
                        "baudrate": config.GPS_BAUDRATE,
                    },
                    daemon=True,
                    name="gps-thread",
                )
                gps_thread.start()
                logger.info("Started GPS thread: %s", gps_thread.name)
            except Exception:
                logger.exception("Failed to start GPS worker; continuing without GPS")
        
    except Exception as e:
        logger.error(f"Failed to start detect.py: {e}", exc_info=True)
        if device_mac:
            model_service.update_device_status(status="offline")
            logger.info("Device status updated to offline")


# @app.on_event("shutdown")
# async def shutdown_event():
#     global detect_process, monitor_task, stderr_task, device_mac
    
#     logger.info("Shutting down SafeDriver Backend")
    
#     # Cancel monitoring tasks
#     if monitor_task:
#         monitor_task.cancel()
#         try:
#             await monitor_task
#         except asyncio.CancelledError:
#             logger.info("Stdout monitoring task cancelled")
    
#     if stderr_task:
#         stderr_task.cancel()
#         try:
#             await stderr_task
#         except asyncio.CancelledError:
#             logger.info("Stderr monitoring task cancelled")
    
#     # Stop detect.py when the API shuts down
#     if detect_process:
#         try:
#             detect_process.terminate()
#             detect_process.wait(timeout=5)
#             logger.info("Stopped detect.py successfully")
#         except Exception as e:
#             logger.error(f"Error stopping detect.py: {e}", exc_info=True)
#             detect_process.kill()  # Force kill if terminate fails
    
#     # Update device status to inactive
#     if device_mac:
#         try:
#             model_service.update_device_status(status="offline")
#             logger.info("Device status updated to offline")
#         except Exception as e:
#             logger.error(f"Failed to update device status: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global detect_process, monitor_task, device_mac, watchdog_task, gps_thread

    logger.info("Shutting down SafeDriver Backend")

    stop_event.set()
    gps_stop_event.set()

        # Cancel watchdog first
    if watchdog_task:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            logger.info("Watchdog task cancelled")

    if monitor_task:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            logger.info("Monitor task cancelled")

    if detect_process and detect_process.is_alive():
        detect_process.join(timeout=2)
        
        if detect_process.is_alive():
            logger.warning("Detect thread still alive - force stopping camera...")
            detect.force_stop()  # ← force release camera so cap.read() unblocks
            # Object-process and MediaPipe cleanup can take several seconds.
            detect_process.join(timeout=12)
            
            if detect_process.is_alive():
                logger.warning("Detect thread did not stop - continuing shutdown")
            else:
                logger.info("Detect thread stopped after force stop")
        else:
            logger.info("Detect thread stopped successfully")

    if gps_thread and gps_thread.is_alive():
        gps_thread.join(timeout=3)
        if gps_thread.is_alive():
            logger.warning("GPS thread did not stop before shutdown timeout")
        else:
            logger.info("GPS thread stopped successfully")

    if device_mac:
        try:
            model_service.update_device_status(status="offline")
            logger.info("Device status updated to offline")
        except Exception as e:
            logger.error(f"Failed to update device status: {e}")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "SafeDriver API is running",
        "device_mac": device_mac,
        "process_running": detect_process.poll() is None if detect_process else False,
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


# @app.post("/process/restart")
# async def restart_detection_process():
#     """Restart the detection process to apply configuration changes"""
#     global detect_process, monitor_task, stderr_task, device_mac
    
#     try:
#         logger.info("Restarting detection process to apply configuration changes...")
        
#         # Stop current process
#         if monitor_task:
#             monitor_task.cancel()
#             try:
#                 await monitor_task
#             except asyncio.CancelledError:
#                 logger.info("Stdout monitoring task cancelled")
        
#         if stderr_task:
#             stderr_task.cancel()
#             try:
#                 await stderr_task
#             except asyncio.CancelledError:
#                 logger.info("Stderr monitoring task cancelled")
        
#         if detect_process:
#             try:
#                 detect_process.terminate()
#                 detect_process.wait(timeout=5)
#                 logger.info("Stopped detect.py process")
#                 model_service.update_device_status(status="offline")
#                 logger.info("Device status updated to offline")
#             except Exception as e:
#                 logger.error(f"Error stopping detect.py: {e}")
#                 detect_process.kill()
        
#         # Start new process
#         venv_python = sys.executable
#         # detect_process = subprocess.Popen(
#         #     [venv_python, "-c", "from model.detect import main; main()"],
#         #     cwd=os.path.dirname(os.path.abspath(__file__)),
#         #     stdout=subprocess.PIPE,
#         #     stderr=subprocess.PIPE,
#         #     universal_newlines=True,
#         #     bufsize=1
#         # )
#         # model_service.update_device_status(status="restarting")
#         # logger.info("Device status updated to restarting")

#         # logger.info(f"Restarted detect.py with PID: {detect_process.pid}")
        
#         # # Start monitoring tasks
#         # monitor_task = asyncio.create_task(read_behavior_queue())
#         # stderr_task = asyncio.create_task(read_detect_process_stderr())
#         # logger.info("Restarted output monitoring tasks")

#         # model_service.update_device_status(status="online")
#         # logger.info("Device status updated to online")

#         detect_process = threading.Thread(
#             target=detect_main,
#             daemon=True,
#             name="detect-thread"
#         )
#         detect_process.start()
#         logger.info(f"Started detect thread: {detect_process.name}")
        
#         # Start monitoring tasks
#         # monitor_task = asyncio.create_task(read_detect_process_output())
#         stderr_task = asyncio.create_task(read_detect_process_stderr())

#         monitor_task = asyncio.create_task(read_behavior_queue())

#         logger.info("Started output monitoring tasks")

#         model_service.update_device_status(status="online")
#         logger.info("Device status updated to online")
        
#         return {
#             "success": True,
#             "message": "Detection process restarted successfully",
#             "new_pid": detect_process.pid
#         }
        
#     except Exception as e:
#         logger.error(f"Error restarting detection process: {e}", exc_info=True)
#         return {
#             "success": False,
#             "message": f"Failed to restart process: {str(e)}"
#         }

@app.post("/process/restart")
async def restart_detection_process():
    """Restart the detection process to apply configuration changes"""
    global detect_process, monitor_task, device_mac

    try:
        logger.info("Restarting detection process...")

        # ── 1. Stop the queue monitor task ──────────────────────────────
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                logger.info("Monitor task cancelled")

        # ── 2. Signal detect thread to stop ─────────────────────────────
        stop_event.set()

        if detect_process and detect_process.is_alive():
            detect_process.join(timeout=10)  # Wait max 10 seconds
            if detect_process.is_alive():
                logger.warning("Detect thread did not stop in time")
            else:
                logger.info("Detect thread stopped")

        model_service.update_device_status(status="offline")
        logger.info("Device status updated to offline")

        # ── 3. Clear stop flag and queue for fresh start ─────────────────
        stop_event.clear()

        # Clear any leftover data in queue
        while not behavior_queue.empty():
            try:
                behavior_queue.get_nowait()
            except queue.Empty:
                break

        # ── 4. Start new detect thread ───────────────────────────────────
        model_service.update_device_status(status="restarting")
        logger.info("Device status updated to restarting")

        detect_process = threading.Thread(
            target=detect.main,
            daemon=True,
            name="detect-thread"
        )
        detect_process.start()
        logger.info(f"Started new detect thread: {detect_process.name}")

        # ── 5. Restart queue monitor ─────────────────────────────────────
        monitor_task = asyncio.create_task(read_behavior_queue())
        logger.info("Restarted behavior queue monitor")

        model_service.update_device_status(status="online")
        logger.info("Device status updated to online")

        return {
            "success": True,
            "message": "Detection process restarted successfully",
            "thread_name": detect_process.name   # ← thread name instead of pid
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

# Update Assigned Driver Endpoint
@app.put("/device/update-driver")
async def update_assigned_driver(driver_id: str):
    """Update the assigned driver for the device"""
    global device_mac
    
    if not device_mac:
        return {
            "success": False,
            "message": "Device MAC not available"
        }
    
    try:
        return db_helper.update_assigned_driver(driver_id, device_mac)
        
    except Exception as e:
        logger.error(f"Error updating assigned driver for MAC {device_mac}: {e}")
        return {
            'success': False,
            'message': str(e)
        }


# Update Driver fingerprint mapping in firestore/drivers/{driver_id}
@app.put("/driver/update-fingerprint")
async def register_driver_fingerprint(
    driver_id: str,
    fingerprint_id: str | None = None,
    scanner_id: str | None = None,
    template_position: int | None = None,
):
    """Update fingerprint mapping for a driver in Firestore."""
    try:
        result = firestore_helper.register_driver_fingerprint(
            driver_id=driver_id,
            fingerprint_data=fingerprint_id,
            scanner_id=scanner_id,
            template_position=template_position,
        )

        if not result.get('success'):
            return result

        return {
            'success': True,
            'driver_id': driver_id,
            'fingerprint_id': fingerprint_id,
            'template_id': result.get('template_id'),
            'scanner_id': scanner_id,
            'template_position': template_position,
            'message': 'Fingerprint mapping updated successfully'
        }
        
    except Exception as e:
        logger.error(f"Error updating fingerprint ID for driver {driver_id}: {e}")
        return {
            'success': False,
            'message': str(e)
        }


@app.get("/driver/by-fingerprint")
async def get_driver_by_fingerprint(
    fingerprint_id: str | None = None,
    scanner_id: str | None = None,
    template_position: int | None = None,
    template_id: str | None = None,
):
    """Resolve a driver using legacy fingerprint ID or scanner template mapping."""
    try:
        driver_id = firestore_helper.get_driver_by_fingerprint(
            fingerprint_data=fingerprint_id,
            scanner_id=scanner_id,
            template_position=template_position,
            template_id=template_id,
        )

        if driver_id is None:
            return {
                'success': False,
                'message': 'Driver not found for provided fingerprint data'
            }

        return {
            'success': True,
            'driver_id': driver_id,
        }
    except Exception as e:
        logger.error(f"Error resolving driver by fingerprint: {e}")
        return {
            'success': False,
            'message': str(e)
        }

# update vehicle_reg_no in realtime database/devices/{device_mac}/vehicle_reg_no
@app.put("/device/update-vehicle-reg")
async def update_vehicle_registration_number(vehicle_reg_no: str):
    """Update the vehicle registration number for the device"""
    global device_mac
    
    if not device_mac:
        return {
            "success": False,
            "message": "Device MAC not available"
        }
    
    try:
        return db_helper.update_vehicle_reg_no(device_mac, vehicle_reg_no)
        
    except Exception as e:
        logger.error(f"Error updating vehicle registration number for MAC {device_mac}: {e}")
        return {
            'success': False,
            'message': str(e)
        }

# Calling for enroll fingerprint when receive an external request
@app.post("/fingerprint/enroll")
async def enroll_fingerprint_for_driver(driver_id: str):
    """Enroll a fingerprint for a specific driver."""
    try:
        result = enroll.enroll_fingerprint_with_id(driver_id)
        return result
    except Exception as e:
        logger.error(f"Error enrolling fingerprint for driver {driver_id}: {e}")
        return {
            'success': False,
            'message': str(e)
        }

# Calling for remove fingerprint when receive an external request
@app.post("/fingerprint/remove")
async def enroll_fingerprint_for_driver(driver_id: str | None = None, position:int | None = None, mac:str | None = None):
    """Remove a fingerprint for a specific driver."""
    try:
        result = remove.delete_fingerprint(mac=mac, position=position, driver_id=driver_id)
        return result
    except Exception as e:
        logger.error(f"Error enrolling fingerprint for driver {driver_id}: {e}")
        return {
            'success': False,
            'message': str(e)
        }
