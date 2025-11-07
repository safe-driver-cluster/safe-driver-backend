import pytz
import cv2
from datetime import datetime
import config.config as config

def now():
    """Return current UTC timestamp in ISO format"""
    sri_lanka_tz = pytz.timezone('Asia/Colombo')
    return datetime.now(sri_lanka_tz).isoformat()


# def get_config():
#     """Return the CONFIG dictionary"""
#     return CONFIG


# def get_config_value(key: str, default=None):
#     """
#     Get a specific configuration value by key.
    
#     Args:
#         key (str): Configuration key
#         default: Default value if key not found
        
#     Returns:
#         Configuration value or default
#     """
#     return CONFIG.get(key, default)


# def update_config(key: str, value):
#     """
#     Update a configuration value.
    
#     Args:
#         key (str): Configuration key
#         value: New value
        
#     Returns:
#         bool: True if updated, False if key doesn't exist
#     """
#     if key in CONFIG:
#         CONFIG[key] = value
#         return True
#     return False


def log_config(logger):
    """
    Log important configuration values.
    
    Args:
        logger: Logger instance to use
    """
    logger.info("=" * 80)
    logger.info("SafeDriver Monitoring System - Configuration Loaded")
    logger.info("=" * 80)
    logger.info(f"Eye Closed Threshold: {config.EYE_CLOSED_THRESH}")
    logger.info(f"Microsleep Duration: {config.MICROSLEEP_SEC}s")
    logger.info(f"PERCLOS Window: {config.PERCLOS_WIN_SEC}s")
    logger.info(f"Yawn Threshold: {config.YAWN_THRESH}")
    logger.info(f"Display Modes - FPS: {config.SHOW_FPS}, Metrics: {config.SHOW_METRICS}, Warnings: {config.SHOW_WARNINGS}")
    logger.info("=" * 80)

def perform_voice_alerts(message):
    """Perform voice alerts using system TTS"""
    try:
        # For Windows
        if os.name == 'nt':
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        # For macOS
        elif os.uname().sysname == 'Darwin':
            os.system(f'say "{message}"')
        # For Linux
        else:
            os.system(f'espeak "{message}"')
    except Exception as e:
        print(f"Error performing voice alert: {e}")