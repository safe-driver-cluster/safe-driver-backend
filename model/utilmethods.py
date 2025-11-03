import pytz
import cv2
from datetime import datetime

# ============================================================================
# CONFIGURATION SECTION - All configurable parameters
# ============================================================================

CONFIG = {
    # Drowsiness Detection Thresholds
    'EYE_CLOSED_THRESH': 0.60,
    'EYE_PARTIAL_THRESH': 0.40,
    'MICROSLEEP_SEC': 1.5,
    'PERCLOS_WIN_SEC': 60.0,
    'PERCLOS_DROWSY': 0.70,
    'YAWN_THRESH': 0.80,
    'YAWN_MIN_SEC': 1.0,
    'EYE_CLOSURE_FREQ_WIN': 15.0,
    'EYE_CLOSURE_FREQ_THRESH': 4,
    'MIN_CLOSURE_DURATION': 0.4,
    'BLINK_MAX_DURATION': 0.4,
    'CLOSURE_DEBOUNCE_TIME': 0.5,
    
    # UI Layout Parameters
    'WINDOW_NAME': 'SafeDriver Monitoring System',
    'ROW_SIZE': 50,
    'LEFT_MARGIN': 24,
    'LABEL_PADDING_WIDTH': 1500,
    'FPS_AVG_FRAME_COUNT': 10,
    'SCROLL_STEP': 20,
    
    # Display Control Flags
    'SHOW_BLENDSHAPES': False,
    'SHOW_FACE_MESH': True,
    'SHOW_FPS': True,
    'SHOW_METRICS': True,
    'SHOW_WARNINGS': True,
    
    # FPS Display
    'FPS_FONT': cv2.FONT_HERSHEY_DUPLEX,
    'FPS_FONT_SIZE': 0.5,
    'FPS_FONT_THICKNESS': 1,
    'FPS_COLOR': (0, 0, 0),
    'FPS_TEXT_FORMAT': 'FPS = {:.1f}',
    'FPS_Y_OFFSET': -20,
    
    # Metrics Box (Top Left)
    'METRICS_PADDING': 10,
    'METRICS_WIDTH': 175,
    'METRICS_HEIGHT': 140,
    'METRICS_Y_OFFSET': 0,
    'METRICS_CORNER_RADIUS': 10,
    'METRICS_BG_COLOR': (255, 255, 255),
    'METRICS_BG_OPACITY': 0.5,
    'METRICS_FONT': cv2.FONT_HERSHEY_SIMPLEX,
    'METRICS_FONT_SIZE': 0.5,
    'METRICS_FONT_THICKNESS': 1,
    'METRICS_TEXT_COLOR': (0, 0, 0),
    
    # Metrics Text Labels
    'LABEL_PERCLOS': 'PERCLOS: {:.2f}',
    'LABEL_BLINKS': 'Blinks/min: {:02d}',
    'LABEL_CLOSURES': 'Closures(15s): {}',
    'LABEL_YAWNS': 'Yawns: {}',
    'LABEL_MICROSLEEPS': 'Microsleeps: {}',
    'LABEL_DROWSY_EVENTS': 'Drowsy Events: {}',
    
    # Metrics Text Positions
    'PERCLOS_Y_OFFSET': 20,
    'BLINKS_Y_OFFSET': 40,
    'CLOSURES_Y_OFFSET': 60,
    'YAWNS_Y_OFFSET': 85,
    'MICROSLEEPS_Y_OFFSET': 105,
    'DROWSY_EVENTS_Y_OFFSET': 125,
    
    # Warning Display
    'WARNING_FONT': cv2.FONT_HERSHEY_DUPLEX,
    'WARNING_FONT_SIZE': 1.0,
    'WARNING_FONT_THICKNESS': 2,
    'WARNING_COLOR': (0, 0, 255),
    'WARNING_Y_POSITION': 50,
    'WARNING_RIGHT_MARGIN': 20,
    
    # Warning Text Messages
    'WARNING_MICROSLEEP': 'Microsleep Detected!',
    'WARNING_YAWNING': 'Yawning Detected!',
    'WARNING_FREQUENT_CLOSURES': 'Frequent Eye Closures!',
    'WARNING_DROWSY': 'Drowsiness Detected!',
    'WARNING_PERCLOS': 'High PERCLOS Level!',
    
    # Console Messages
    'CONSOLE_MICROSLEEP': 'Microsleep detected (Total: {})',
    'CONSOLE_YAWN': 'Yawn detected (Total: {})',
    'CONSOLE_FREQUENT_CLOSURES': 'Frequent eye closures detected',
    'CONSOLE_DROWSY': 'Drowsiness detected (Total: {})',
    'CONSOLE_PERCLOS_REACHED': 'PERCLOS threshold reached: {:.2f}',

    # Behavior data message type
    'BEHAVIOR_FREQUENT_CLOSURES': 'frequent_closures',
    'BEHAVIOR_MICROSLEEP': 'microsleep',
    'BEHAVIOR_YAWN': 'yawn',
    'BEHAVIOR_DROWSY': 'drowsy',
    'BEHAVIOR_PERCLOS_REACHED': 'perclos_threshold_reached',
    
    # Blendshapes Display
    'BLENDSHAPE_FONT': cv2.FONT_HERSHEY_SIMPLEX,
    'BLENDSHAPE_FONT_SIZE': 0.4,
    'BLENDSHAPE_FONT_THICKNESS': 1,
    'BLENDSHAPE_TEXT_COLOR': (0, 0, 0),
    'BLENDSHAPE_BAR_COLOR': (0, 255, 0),
    'BLENDSHAPE_BAR_HEIGHT': 8,
    'BLENDSHAPE_GAP_BETWEEN_BARS': 5,
    'BLENDSHAPE_TEXT_GAP': 5,
    'BLENDSHAPE_X_OFFSET': 20,
    'BLENDSHAPE_Y_START': 30,
    'BLENDSHAPE_TEXT_FORMAT': '{} ({:.2f})',
    
    # Face Mesh Drawing Colors
    'LABEL_BG_COLOR': (255, 255, 255),
    
    # Camera Error Message
    'CAMERA_ERROR_MSG': 'ERROR: Unable to read from webcam. Please verify your webcam settings.'
}

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


def now():
    """Return current UTC timestamp in ISO format"""
    sri_lanka_tz = pytz.timezone('Asia/Colombo')
    return datetime.now(sri_lanka_tz).isoformat()


def get_config():
    """Return the CONFIG dictionary"""
    return CONFIG


def get_config_value(key: str, default=None):
    """
    Get a specific configuration value by key.
    
    Args:
        key (str): Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    return CONFIG.get(key, default)


def update_config(key: str, value):
    """
    Update a configuration value.
    
    Args:
        key (str): Configuration key
        value: New value
        
    Returns:
        bool: True if updated, False if key doesn't exist
    """
    if key in CONFIG:
        CONFIG[key] = value
        return True
    return False


def log_config(logger):
    """
    Log important configuration values.
    
    Args:
        logger: Logger instance to use
    """
    logger.info("=" * 80)
    logger.info("SafeDriver Monitoring System - Configuration Loaded")
    logger.info("=" * 80)
    logger.info(f"Eye Closed Threshold: {CONFIG['EYE_CLOSED_THRESH']}")
    logger.info(f"Microsleep Duration: {CONFIG['MICROSLEEP_SEC']}s")
    logger.info(f"PERCLOS Window: {CONFIG['PERCLOS_WIN_SEC']}s")
    logger.info(f"Yawn Threshold: {CONFIG['YAWN_THRESH']}")
    logger.info(f"Display Modes - FPS: {CONFIG['SHOW_FPS']}, Metrics: {CONFIG['SHOW_METRICS']}, Warnings: {CONFIG['SHOW_WARNINGS']}")
    logger.info("=" * 80)