import pytz
import cv2
from datetime import datetime
import config.config as config
import threading
import os
# from playsound import playsound
from gtts import gTTS
import utils.utils as utils
import logging
logger = logging.getLogger(__name__)


import pygame
pygame.mixer.init()

_VOICE_PLAYBACK_LOCK = threading.Lock()
_VOICE_THREADS_LOCK = threading.Lock()
_VOICE_THREADS = set()
_VOICE_STOP_EVENT = threading.Event()


def get_voice_alert_file(label, language=None):
    """Return the expected local MP3 path for a voice-alert label and language."""
    selected_language = language or config.LANGUAGE
    if selected_language not in ("ENGLISH", "SINHALA", "TAMIL"):
        selected_language = "ENGLISH"

    filename = f"{label}_{selected_language}.mp3"
    return os.path.join(utils.get_audio_dir(), selected_language, filename)


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
#                 logger.info(f"[TTS Error] {e}")

#         # Run TTS in a separate thread
#         t = threading.Thread(target=_play_sound, args=(message,))
#         t.daemon = True  # ensures thread exits when main program exits
#         t.start()
#     except Exception as e:
#         logger.info(f"Error performing voice alert: {e}")

def perform_voice_alerts(message, label="VOICE_ALERT", language_dependent=True):
    try:
        if not config.ENABLE_VOICE_ALERTS or _VOICE_STOP_EVENT.is_set():
            return

        def _play_sound(text_inner):
            try:
                with _VOICE_PLAYBACK_LOCK:
                    if _VOICE_STOP_EVENT.is_set():
                        return

                    if language_dependent:
                        language = config.LANGUAGE if config.LANGUAGE in ("ENGLISH", "SINHALA", "TAMIL") else "ENGLISH"
                        lang_code = {"ENGLISH": "en", "SINHALA": "si", "TAMIL": "ta"}
                    else:
                        language = "ENGLISH"
                        lang_code = {"ENGLISH": "en"}

                    filename = get_voice_alert_file(label, language)

                    if not os.path.exists(filename):
                        logger.warning(
                            "[Sound] Voice MP3 not found for label=%s language=%s. "
                            "Generating with gTTS: %s",
                            label,
                            language,
                            filename,
                        )
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        tts = gTTS(text=text_inner, lang=lang_code[language])
                        tts.save(filename)

                    logger.info("Voice playback started: %s", label)
                    pygame.mixer.music.load(filename)
                    pygame.mixer.music.play()

                    while pygame.mixer.music.get_busy():
                        if _VOICE_STOP_EVENT.is_set():
                            pygame.mixer.music.stop()
                            break
                        pygame.time.Clock().tick(10)

                    logger.info("Voice playback completed: %s", label)

            except Exception as e:
                if not _VOICE_STOP_EVENT.is_set():
                    logger.info(f"[Sound Error] {e}")
            finally:
                with _VOICE_THREADS_LOCK:
                    _VOICE_THREADS.discard(threading.current_thread())

        t = threading.Thread(target=_play_sound, args=(message,))
        t.daemon = True
        with _VOICE_THREADS_LOCK:
            _VOICE_THREADS.add(t)
        t.start()

    except Exception as e:
        logger.info(f"Error performing voice alert: {e}")


def shutdown_voice_alerts(timeout=2.0):
    """Stop active voice playback and wait briefly for playback threads."""
    _VOICE_STOP_EVENT.set()

    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception:
        pass

    with _VOICE_THREADS_LOCK:
        threads = list(_VOICE_THREADS)

    for thread in threads:
        thread.join(timeout=timeout)


def reset_voice_alert_shutdown():
    """Allow voice playback after the detection service is restarted."""
    _VOICE_STOP_EVENT.clear()
