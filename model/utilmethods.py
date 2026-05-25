import pytz
import cv2
from datetime import datetime
import config.config as config
import threading
import os
# from playsound import playsound
from gtts import gTTS
from database.firestore_helper import firestore_helper
import utils.utils as utils

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

# def perform_voice_alerts(message):
#     """Perform voice alerts using system TTS"""
#     try:
#         if(not config.ENABLE_VOICE_ALERTS):
#             return

#         # For Windows
#         def _play_sound(text_inner):
#             try:
#                 tts = gTTS(text=text_inner, lang='en')
#                 filename = message+"alert.mp3"
#                 if os.path.exists(filename):
#                     playsound(filename)
#                 else:
#                     tts.save(filename)
#                     playsound(filename)
#                 # os.remove(filename)
#             except Exception as e:
#                 print(f"[TTS Error] {e}")

#         # Run TTS in a separate thread
#         t = threading.Thread(target=_play_sound, args=(message,))
#         t.daemon = True  # ensures thread exits when main program exits
#         t.start()
#     except Exception as e:
#         print(f"Error performing voice alert: {e}")

def perform_voice_alerts(message, label="VOICE_ALERT"):
    try:
        if not config.ENABLE_VOICE_ALERTS:
            return

        # def _play_sound(text_inner):
        #     filename = None
        #     try:
        #         # unique file name
        #         filename = f"{label}.mp3"

        #         # generate TTS
        #         tts = gTTS(text=text_inner, lang='en')
        #         tts.save(filename)

        #         # play audio
        #         pygame.mixer.music.load(utils.resource_path(filename))
        #         pygame.mixer.music.play()

        #         while pygame.mixer.music.get_busy():
        #             pygame.time.Clock().tick(10)

        #     except Exception as e:
        #         print(f"[TTS Error] {e}")

        #     finally:
        #         try:
        #             if filename and os.path.exists(filename):
        #                 os.remove(utils.resource_path(filename))
        #         except:
        #             pass

        def _play_sound(text_inner):
            filename = None
            try:
                app_dir = utils.get_app_dir()
                filename = os.path.join(app_dir, f"{label}.mp3")

                tts = gTTS(text=text_inner, lang='en')
                tts.save(filename)

                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

            except Exception as e:
                print(f"[TTS Error] {e}")

            finally:
                try:
                    if filename and os.path.exists(filename):
                        os.remove(filename)
                except:
                    pass

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
