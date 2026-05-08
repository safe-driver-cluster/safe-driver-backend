import pytz
import cv2
from datetime import datetime
import config.config as config
import threading
import os
from playsound import playsound
from gtts import gTTS
from database.firestore_helper import firestore_helper
import utils.utils as utils
import pygame

IS_WINDOWS = platform.system() == "Windows"

# Conditional imports
if IS_WINDOWS:
    from playsound import playsound
else:
    import pygame
    pygame.mixer.init()

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

def perform_voice_alerts(message):
    """Cross-platform voice alerts (Windows + Linux + fallback)"""
    try:
        if not config.ENABLE_VOICE_ALERTS:
            return

        def _play_sound(text_inner):
            try:
                filename = f"{text_inner}_alert.mp3"

                # Generate only once
                if not os.path.exists(filename):
                    tts = gTTS(text=text_inner, lang='en')
                    tts.save(filename)

                if IS_WINDOWS:
                    # Windows
                    playsound(filename)

                else:
                    # Linux (Primary: pygame)
                    try:
                        pygame.mixer.music.load(filename)
                        pygame.mixer.music.play()

                        while pygame.mixer.music.get_busy():
                            pass

                    except Exception as e:
                        print("[pygame failed] switching to mpg123:", e)
                        os.system(f"mpg123 '{filename}'")

            except Exception as e:
                print(f"[TTS Error] {e}")

        t = threading.Thread(target=_play_sound, args=(message,))
        t.daemon = True
        t.start()

    except Exception as e:
        print(f"Error performing voice alert: {e}")

def get_model_configurations(logger):
    """Get model configurations from Firestore"""
    try:
        result = firestore_helper.get_model_configurations_from_firestore()
        if result:
            # save to local config as well
            utils.update_local_config_from_firestore(result)
            logger.info("Configurations retrieved successfully...")
        else:
            logger.info("No configurations found!")
    except Exception as e:
        logger.error(f"Error in get configurations endpoint: {e}")
