import argparse
import sys
import time
import logging

import numpy as np
from collections import deque

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import model.utilmethods as utils
import utils.utils as util
import config.config as config
from model.alerts import AlertManager

import firebase_admin
from firebase_admin import credentials, db

import model.frame_detector as frame_detect

from shared import stop_event

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configure logging - Log to stderr and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safe_driver_debug.log'),
        logging.StreamHandler(sys.stderr)  # Log to stderr
    ]
)
logger = logging.getLogger(__name__)

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

# State variables
EYE_CLOSED_START = None
YAWN_START = None
PERCLOS_WIN = deque()
BLINK_TIMES = deque()
EYE_CLOSURE_EVENTS = deque()
EYE_PARTIAL_CLOSURE_START = None
HEAD_TURNED_START = None  # Track when head turn started
LAST_HEAD_POSE_STATE = "CENTER"  # Track head pose state
NO_FACE_START = None  # Track when face disappeared
NO_FACE_COUNTED = False  # Track if no-face event was counted

# Event counters
YAWN_COUNT = 0
DROWSY_COUNT = 0
MICROSLEEP_COUNT = 0
HEAD_TURN_COUNT = 0
FACE_MISSING_COUNT = 0
FREQUENT_CLOSURES_COUNT = 0

# Drowsiness Behavior Detection Times Array Within 10 Minutes (EVENT_ARRAY_TIME_WINDOW_SEC)
DROWSY_EVENT_TIME_ARRAY_SEC = deque()  # Stores timestamps of drowsy events in seconds
DISTRACTION_EVENT_TIME_ARRAY_SEC = deque()  # Stores timestamps of distraction events in seconds
HEADTURN_EVENT_TIME_ARRAY_SEC = deque()  # Stores timestamps of head turn events in seconds
PERCLOSE_EVENT_TIME_ARRAY_SEC = deque()  # Stores timestamps of PERCLOS events in seconds
FREQUENTEYE_CLOSURES_EVENT_TIME_ARRAY_SEC = deque()  # Stores timestamps of frequent eye closures events in seconds
YAWN_EVENT_TIME_ARRAY_SEC = deque()  # Stores timestamps of yawn events in seconds
MICROSLEEP_EVENT_TIME_ARRAY_SEC = deque()  # Stores timestamps of microsleep events in seconds

YAWN_COUNTED = False
MICROSLEEP_COUNTED = False
DROWSY_COUNTED = False
FREQUENT_CLOSURES_COUNTED = False
HEAD_TURN_COUNTED = False  # Track head turn events
FACE_MISSING_COUNTED = False  # Track face missing events

eye_closed_score = 0.0
mouth_lower_down = 0.0

LAST_COUNTER_EVENT_TIME = {
    "yawn": None,
    "drowsy": None,
    "microsleep": None,
    "head_turn": None,
    "face_missing": None,
    "frequent_closures": None,
}

# Scroll variables
SCROLL_OFFSET = 0
MAX_SCROLL = 0

ALERT_MANAGER = AlertManager(
    logger=logger,
    now_provider=utils.now,
    output_stream=sys.stdout,
    threshold_defaults={
        config.BEHAVIOR_DROWSY: False,
        config.BEHAVIOR_YAWN: False,
        config.BEHAVIOR_MICROSLEEP: False,
        config.BEHAVIOR_FREQUENT_CLOSURES: False,
        config.BEHAVIOR_HEAD_TURN: False,
        config.BEHAVIOR_DISTRACTION: False,
    },
)


def _bs_score(blendshapes, name: str) -> float:
    """Return blendshape score by name or 0.0 if missing."""
    for c in blendshapes:
        if c.category_name == name:
            return float(c.score)
    return 0.0


def _reset_counter(counter_key: str) -> None:
    """Reset event counter and alert state for a specific event type."""
    global YAWN_COUNT, DROWSY_COUNT, MICROSLEEP_COUNT, HEAD_TURN_COUNT, FACE_MISSING_COUNT, FREQUENT_CLOSURES_COUNT

    if counter_key == "yawn":
        YAWN_COUNT = 0
    elif counter_key == "drowsy":
        DROWSY_COUNT = 0
    elif counter_key == "microsleep":
        MICROSLEEP_COUNT = 0
    elif counter_key == "head_turn":
        HEAD_TURN_COUNT = 0
    elif counter_key == "face_missing":
        FACE_MISSING_COUNT = 0
    elif counter_key == "frequent_closures":
        FREQUENT_CLOSURES_COUNT = 0

    LAST_COUNTER_EVENT_TIME[counter_key] = None
    ALERT_MANAGER.reset_event_state(counter_key)


def _increment_event_counter(counter_key: str) -> int:
    """Increment counter with stale-reset behavior based on EVENT_COUNT_RESET_SEC."""
    global YAWN_COUNT, DROWSY_COUNT, MICROSLEEP_COUNT, HEAD_TURN_COUNT, FACE_MISSING_COUNT, FREQUENT_CLOSURES_COUNT

    now_ts = time.time()
    last_ts = LAST_COUNTER_EVENT_TIME.get(counter_key)
    if last_ts is not None and (now_ts - last_ts) >= config.EVENT_COUNT_RESET_SEC:
        _reset_counter(counter_key)

    if counter_key == "yawn":
        YAWN_COUNT += 1
        new_value = YAWN_COUNT
    elif counter_key == "drowsy":
        DROWSY_COUNT += 1
        new_value = DROWSY_COUNT
    elif counter_key == "microsleep":
        MICROSLEEP_COUNT += 1
        new_value = MICROSLEEP_COUNT
    elif counter_key == "head_turn":
        HEAD_TURN_COUNT += 1
        new_value = HEAD_TURN_COUNT
    elif counter_key == "face_missing":
        FACE_MISSING_COUNT += 1
        new_value = FACE_MISSING_COUNT
    elif counter_key == "frequent_closures":
        FREQUENT_CLOSURES_COUNT += 1
        new_value = FREQUENT_CLOSURES_COUNT
    else:
        raise ValueError(f"Unknown counter key: {counter_key}")

    LAST_COUNTER_EVENT_TIME[counter_key] = now_ts
    return new_value


def calculate_head_pose(face_landmarks, image_width, image_height):
    """Calculate head pose angles (yaw, pitch, roll) from facial landmarks."""
    if not face_landmarks:
        return None
    
    # Key landmark indices for head pose estimation
    # Nose tip
    nose_tip = face_landmarks[1]
    # Chin
    chin = face_landmarks[152]
    # Left eye outer corner
    left_eye = face_landmarks[263]
    # Right eye outer corner 
    right_eye = face_landmarks[33]
    # Left mouth corner
    left_mouth = face_landmarks[61]
    # Right mouth corner
    right_mouth = face_landmarks[291]
    
    # Convert normalized coordinates to pixel coordinates
    nose_2d = (nose_tip.x * image_width, nose_tip.y * image_height)
    chin_2d = (chin.x * image_width, chin.y * image_height)
    left_eye_2d = (left_eye.x * image_width, left_eye.y * image_height)
    right_eye_2d = (right_eye.x * image_width, right_eye.y * image_height)
    left_mouth_2d = (left_mouth.x * image_width, left_mouth.y * image_height)
    right_mouth_2d = (right_mouth.x * image_width, right_mouth.y * image_height)
    
    # Calculate yaw (left-right rotation)
    # Based on the horizontal position of nose relative to eye centers
    eye_center_x = (left_eye_2d[0] + right_eye_2d[0]) / 2
    nose_x = nose_2d[0]
    eye_width = abs(right_eye_2d[0] - left_eye_2d[0])
    
    if eye_width > 0:
        # Normalize the offset (-1 to 1, where negative is left, positive is right)
        yaw_normalized = (nose_x - eye_center_x) / (eye_width / 2)
        # Convert to degrees (approximate, -45 to 45 degrees)
        yaw = yaw_normalized * 45
    else:
        yaw = 0
    
    # Calculate pitch (up-down rotation)
    # Based on vertical distance between nose and chin relative to face height
    face_height = abs(chin_2d[1] - left_eye_2d[1])
    nose_to_eye = abs(nose_2d[1] - left_eye_2d[1])
    
    if face_height > 0:
        pitch_normalized = (nose_to_eye / face_height) - 0.5
        pitch = pitch_normalized * 60  # Approximate pitch angle
    else:
        pitch = 0
    
    # Calculate roll (tilt rotation)
    # Based on the angle between the two eyes
    eye_dx = right_eye_2d[0] - left_eye_2d[0]
    eye_dy = right_eye_2d[1] - left_eye_2d[1]
    
    if eye_dx != 0:
        roll = np.degrees(np.arctan(eye_dy / eye_dx))
    else:
        roll = 0
    
    return {
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll
    }


def detect_head_turn_distraction(face_landmarks, image_width, image_height):
    """Detect if driver is looking away based on head pose."""
    global HEAD_TURNED_START, HEAD_TURN_COUNTED, LAST_HEAD_POSE_STATE, NO_FACE_START, NO_FACE_COUNTED
    global HEAD_TURN_COUNT, FACE_MISSING_COUNT
    
    if not face_landmarks:
        # No face detected - track as distraction
        HEAD_TURNED_START = None
        HEAD_TURN_COUNTED = False
        
        now = time.time()
        
        # Start tracking no-face duration
        if NO_FACE_START is None:
            NO_FACE_START = now
            LAST_HEAD_POSE_STATE = "NO_FACE"
            logger.debug("Face lost - starting no-face tracking")
        else:
            duration = now - NO_FACE_START
            
            # Send alert if face missing for too long
            if duration >= config.HEAD_TURN_DISTRACTION_SEC and not NO_FACE_COUNTED:
                NO_FACE_COUNTED = True
                face_missing_count = _increment_event_counter("face_missing")

                ######### Maintain event time queue #########
                global DISTRACTION_EVENT_TIME_ARRAY_SEC
                DISTRACTION_EVENT_TIME_ARRAY_SEC.append(now)
                timeframe_count = 0
                while (DISTRACTION_EVENT_TIME_ARRAY_SEC and (now - DISTRACTION_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                    DISTRACTION_EVENT_TIME_ARRAY_SEC.popleft()
                timeframe_count = len(DISTRACTION_EVENT_TIME_ARRAY_SEC)

                logger.warning(f"NO FACE DETECTED! Duration: {duration:.2f}s Timeframe count: {timeframe_count} Face missing count: {face_missing_count}")

                ALERT_MANAGER.check_and_send_threshold_alert(
                    tag="DISTRACTION_EVENT",
                    event_type=config.BEHAVIOR_DISTRACTION,
                    message=config.CONSOLE_FACE_LOSS.format(duration),
                    behavior_data={
                        "direction": "AWAY",
                        "duration": duration,
                        "yaw": None,
                        "pitch": None,
                        "roll": None,
                        "face_lost": True,
                        "source": "face_missing",
                    },
                    policy_key="face_missing",
                    cycle_id=int(now * 1000),
                    current_count=face_missing_count,
                    threshold=config.FACE_MISSING_COUNT_THRESH,
                    send_cloud=True,
                    trigger_voice=True,
                    voice_message=config.VOICE_ALERT_DISTRACTION,
                    trigger_buzzer=True,
                    buzzer_message=config.WARNING_DISTRACTION,
                    timeframe_count=timeframe_count,
                )
        
        return {
            'is_turned': True,
            'direction': "AWAY",
            'yaw': None,
            'pitch': None,
            'roll': None,
            'duration': now - NO_FACE_START if NO_FACE_START else 0,
            'face_lost': True
        }
    
    # Face detected - reset no-face tracking
    NO_FACE_START = None
    NO_FACE_COUNTED = False
    
    head_pose = calculate_head_pose(face_landmarks, image_width, image_height)
    
    if not head_pose:
        return {
            'is_turned': False,
            'direction': None,
            'yaw': None,
            'pitch': None,
            'roll': None,
            'duration': 0
        }
    
    yaw = head_pose['yaw']
    now = time.time()
    
    # Determine head direction based on yaw angle
    is_turned = False
    direction = None
    
    if yaw < -config.HEAD_YAW_THRESH_LEFT:
        is_turned = True
        direction = "LEFT"
    elif yaw > config.HEAD_YAW_THRESH_RIGHT:
        is_turned = True
        direction = "RIGHT"
    else:
        direction = "CENTER"
    
    # Track head turn duration
    duration = 0
    if is_turned:
        if HEAD_TURNED_START is None:
            HEAD_TURNED_START = now
            LAST_HEAD_POSE_STATE = direction
            logger.debug(f"Head turn started - Direction: {direction}, Yaw: {yaw:.1f}°")
        else:
            duration = now - HEAD_TURNED_START
            
            # Send alert if head turned for too long
            if duration >= config.HEAD_TURN_DISTRACTION_SEC and not HEAD_TURN_COUNTED:
                HEAD_TURN_COUNTED = True
                head_turn_count = _increment_event_counter("head_turn")

                ######### Maintain event time queue #########
                global HEADTURN_EVENT_TIME_ARRAY_SEC
                HEADTURN_EVENT_TIME_ARRAY_SEC.append(now)
                timeframe_count = 0
                while (HEADTURN_EVENT_TIME_ARRAY_SEC and (now - HEADTURN_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                    HEADTURN_EVENT_TIME_ARRAY_SEC.popleft()
                timeframe_count = len(HEADTURN_EVENT_TIME_ARRAY_SEC)

                logger.warning(f"HEAD TURN DISTRACTION! Direction: {direction}, Duration: {duration:.2f}s, Yaw: {yaw:.1f}° Timeframe count: {timeframe_count} Head turn count: {head_turn_count}")

                ALERT_MANAGER.check_and_send_threshold_alert(
                    tag="DISTRACTION_EVENT",
                    event_type=config.BEHAVIOR_HEAD_TURN,
                    message=config.CONSOLE_HEAD_TURN.format(direction, duration),
                    behavior_data={
                        "direction": direction,
                        "duration": duration,
                        "yaw": yaw,
                        "pitch": head_pose['pitch'],
                        "roll": head_pose['roll']
                    },
                    policy_key="head_turn",
                    cycle_id=int(now * 1000),
                    current_count=head_turn_count,
                    threshold=config.HEAD_TURN_COUNT_THRESH,
                    send_cloud=True,
                    trigger_voice=True,
                    voice_message=config.VOICE_ALERT_HEAD_TURN,
                    trigger_buzzer=True,
                    buzzer_message=config.WARNING_HEAD_TURN,
                    timeframe_count=timeframe_count,
                )
    else:
        HEAD_TURNED_START = None
        HEAD_TURN_COUNTED = False
        LAST_HEAD_POSE_STATE = "CENTER"
    
    return {
        'is_turned': is_turned,
        'direction': direction,
        'yaw': yaw,
        'pitch': head_pose['pitch'],
        'roll': head_pose['roll'],
        'duration': duration
    }


def detect_driver_behavior(face_blendshapes: np.ndarray, height, current_frame, face_landmarks=None, image_width=None, image_height=None) -> dict:
    """Detect driver drowsiness behaviors based on facial blendshapes."""
    if face_blendshapes:
        now = time.time()
        alert_cycle_id = int(now * 1000)
        bs = face_blendshapes[0]

        # --- Eye & mouth signals from blendshapes ---
        blink_l = _bs_score(bs, "eyeBlinkLeft")
        blink_r = _bs_score(bs, "eyeBlinkRight")
        global eye_closed_score
        eye_closed_score = 0.5 * (blink_l + blink_r)

        # Use mouthLowerDownRight and mouthLowerDownLeft for yawn detection
        mouth_lower_down_r = _bs_score(bs, "mouthLowerDownRight")
        mouth_lower_down_l = _bs_score(bs, "mouthLowerDownLeft")
        global mouth_lower_down
        mouth_lower_down = 0.5 * (mouth_lower_down_r + mouth_lower_down_l)

        # --- PERCLOS window maintenance ---
        is_closed = 1 if eye_closed_score > config.EYE_CLOSED_THRESH else 0
        PERCLOS_WIN.append((now, is_closed))
        while PERCLOS_WIN and (now - PERCLOS_WIN[0][0]) > config.PERCLOS_WIN_SEC:
            PERCLOS_WIN.popleft()
        perclos = (sum(v for _, v in PERCLOS_WIN) / len(PERCLOS_WIN)) if PERCLOS_WIN else 0.0
        if perclos >= config.PERCLOS_DROWSY:

            ######### Maintain event time queue #########
            global PERCLOSE_EVENT_TIME_ARRAY_SEC
            PERCLOSE_EVENT_TIME_ARRAY_SEC.append(now)
            timeframe_count = 0
            while (PERCLOSE_EVENT_TIME_ARRAY_SEC and (now - PERCLOSE_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                PERCLOSE_EVENT_TIME_ARRAY_SEC.popleft()
            timeframe_count = len(PERCLOSE_EVENT_TIME_ARRAY_SEC)
            
            logger.warning(f"PERCLOS THRESHOLD REACHED! PERCLOS: {perclos:.2f} over last {config.PERCLOS_WIN_SEC}s Timeframe count: {timeframe_count}")

            ALERT_MANAGER.check_and_send_threshold_alert(
                tag="DROWSY_EVENT",
                event_type=config.BEHAVIOR_PERCLOS_REACHED,
                message=config.CONSOLE_PERCLOS_REACHED.format(perclos),
                behavior_data={
                    "perclos": perclos,
                    "time_window": config.PERCLOS_WIN_SEC
                },
                policy_key="drowsy",
                cycle_id=alert_cycle_id,
                send_cloud=False,
                trigger_voice=True,
                voice_message=config.VOICE_ALERT_PERCLOS,
                trigger_buzzer=True,
                buzzer_message=config.WARNING_PERCLOS,
                timeframe_count=timeframe_count
            )

        # --- Eye closure frequency tracking ---
        global EYE_CLOSURE_EVENTS, EYE_PARTIAL_CLOSURE_START, FREQUENT_CLOSURES_COUNTED, FREQUENT_CLOSURES_COUNT
        
        if eye_closed_score > config.EYE_PARTIAL_THRESH:
            if EYE_PARTIAL_CLOSURE_START is None:
                EYE_PARTIAL_CLOSURE_START = now
                logger.debug(f"Eye closure started - Score: {eye_closed_score:.2f}")
        else:
            if EYE_PARTIAL_CLOSURE_START is not None:
                closure_duration = now - EYE_PARTIAL_CLOSURE_START
                
                if closure_duration >= config.MIN_CLOSURE_DURATION:
                    if not EYE_CLOSURE_EVENTS or (EYE_PARTIAL_CLOSURE_START - EYE_CLOSURE_EVENTS[-1]) > config.CLOSURE_DEBOUNCE_TIME:
                        EYE_CLOSURE_EVENTS.append(now)
                        logger.debug(f"Eye closure recorded - Duration: {closure_duration:.2f}s, Total closures: {len(EYE_CLOSURE_EVENTS)}")
                
                EYE_PARTIAL_CLOSURE_START = None
        
        while EYE_CLOSURE_EVENTS and (now - EYE_CLOSURE_EVENTS[0]) > config.EYE_CLOSURE_FREQ_WIN:
            EYE_CLOSURE_EVENTS.popleft()
        
        frequent_closures = len(EYE_CLOSURE_EVENTS) > config.EYE_CLOSURE_FREQ_THRESH
        
        if frequent_closures and not FREQUENT_CLOSURES_COUNTED:
            FREQUENT_CLOSURES_COUNTED = True
            frequent_closures_count = _increment_event_counter("frequent_closures")

            ######### Maintain event time queue #########
            global FREQUENTEYE_CLOSURES_EVENT_TIME_ARRAY_SEC
            FREQUENTEYE_CLOSURES_EVENT_TIME_ARRAY_SEC.append(now)
            timeframe_count = 0
            while (FREQUENTEYE_CLOSURES_EVENT_TIME_ARRAY_SEC and (now - FREQUENTEYE_CLOSURES_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                FREQUENTEYE_CLOSURES_EVENT_TIME_ARRAY_SEC.popleft()
            timeframe_count = len(FREQUENTEYE_CLOSURES_EVENT_TIME_ARRAY_SEC)

            logger.warning(f"FREQUENT EYE CLOSURES DETECTED! {len(EYE_CLOSURE_EVENTS)} closures in {config.EYE_CLOSURE_FREQ_WIN}s Timeframe count: {timeframe_count} Frequent closures count: {frequent_closures_count}")

            ALERT_MANAGER.check_and_send_threshold_alert(
                tag="DROWSY_EVENT",
                event_type=config.BEHAVIOR_FREQUENT_CLOSURES,
                message=config.CONSOLE_FREQUENT_CLOSURES,
                behavior_data={
                    "closure_count": len(EYE_CLOSURE_EVENTS),
                    "time_window": config.EYE_CLOSURE_FREQ_WIN
                },
                policy_key="frequent_closures",
                cycle_id=alert_cycle_id,
                current_count=frequent_closures_count,
                threshold=config.FREQUENT_CLOSURES_THRESH,
                send_cloud=True,
                trigger_voice=True,
                voice_message=config.VOICE_ALERT_DROWSY,
                trigger_buzzer=True,
                buzzer_message=config.WARNING_FREQUENT_CLOSURES,
                timeframe_count=timeframe_count

            )

        elif not frequent_closures:
            FREQUENT_CLOSURES_COUNTED = False

        # --- Microsleep & blink counting ---
        global EYE_CLOSED_START, MICROSLEEP_COUNT, MICROSLEEP_COUNTED
        if is_closed:
            if EYE_CLOSED_START is None:
                EYE_CLOSED_START = now
        else:
            if EYE_CLOSED_START is not None:
                duration = now - EYE_CLOSED_START
                if duration < config.BLINK_MAX_DURATION:
                    BLINK_TIMES.append(now)
                    logger.debug(f"Blink detected - Duration: {duration:.2f}s")
                EYE_CLOSED_START = None
                MICROSLEEP_COUNTED = False

        while BLINK_TIMES and (now - BLINK_TIMES[0]) > 60.0:
            BLINK_TIMES.popleft()
        blinks_per_min = len(BLINK_TIMES)

        microsleep = (EYE_CLOSED_START is not None) and ((now - EYE_CLOSED_START) >= config.MICROSLEEP_SEC)
        
        if microsleep and not MICROSLEEP_COUNTED:
            microsleep_count = _increment_event_counter("microsleep")
            MICROSLEEP_COUNTED = True
            duration = now - EYE_CLOSED_START

            ######### Maintain event time queue #########
            global MICROSLEEP_EVENT_TIME_ARRAY_SEC
            MICROSLEEP_EVENT_TIME_ARRAY_SEC.append(now)
            timeframe_count = 0
            while (MICROSLEEP_EVENT_TIME_ARRAY_SEC and (now - MICROSLEEP_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                MICROSLEEP_EVENT_TIME_ARRAY_SEC.popleft()
            timeframe_count = len(MICROSLEEP_EVENT_TIME_ARRAY_SEC)

            logger.warning(f"MICROSLEEP DETECTED! Duration: {duration:.2f}s, Total count: {microsleep_count} Timeframe count: {timeframe_count}")

            ALERT_MANAGER.check_and_send_threshold_alert(
                tag="DROWSY_EVENT",
                event_type=config.BEHAVIOR_MICROSLEEP,
                message=config.CONSOLE_MICROSLEEP.format(microsleep_count),
                behavior_data={
                    "duration": duration,
                    "total_count": microsleep_count
                },
                policy_key="microsleep",
                cycle_id=alert_cycle_id,
                current_count=microsleep_count,
                threshold=config.MICROSLEEP_EVENT_COUNT_THRESH,
                send_cloud=True,
                trigger_voice=True,
                voice_message=config.VOICE_ALERT_MICROSLEEP,
                trigger_buzzer=True,
                buzzer_message=config.WARNING_MICROSLEEP,
                timeframe_count=timeframe_count
            )

        # --- Yawn detection ---
        global YAWN_START, YAWN_COUNT, YAWN_COUNTED
        yawning = False
        if mouth_lower_down > config.YAWN_THRESH:
            if YAWN_START is None:
                YAWN_START = now
                logger.debug(f"Yawn started - Mouth score: {mouth_lower_down:.2f}")
            elif (now - YAWN_START) >= config.YAWN_MIN_SEC:
                yawning = True
                if not YAWN_COUNTED:
                    yawn_count = _increment_event_counter("yawn")
                    YAWN_COUNTED = True
                    duration = now - YAWN_START

                    ######### Maintain event time queue #########
                    global YAWN_EVENT_TIME_ARRAY_SEC
                    YAWN_EVENT_TIME_ARRAY_SEC.append(now)
                    timeframe_count = 0
                    while (YAWN_EVENT_TIME_ARRAY_SEC and (now - YAWN_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                        YAWN_EVENT_TIME_ARRAY_SEC.popleft()
                    timeframe_count = len(YAWN_EVENT_TIME_ARRAY_SEC)

                    logger.warning(f"YAWN DETECTED! Duration: {duration:.2f}s, Total count: {yawn_count} Timeframe count: {timeframe_count}")

                    ALERT_MANAGER.check_and_send_threshold_alert(
                        tag="DROWSY_EVENT",
                        event_type=config.BEHAVIOR_YAWN,
                        message=config.CONSOLE_YAWN.format(yawn_count),
                        behavior_data={
                            "duration": duration,
                            "total_count": yawn_count
                        },
                        policy_key="yawn",
                        cycle_id=alert_cycle_id,
                        current_count=yawn_count,
                        threshold=config.YAWN_EVENT_COUNT_THRESH,
                        send_cloud=True,
                        trigger_voice=True,
                        voice_message=config.VOICE_ALERT_YAWNING,
                        trigger_buzzer=True,
                        buzzer_message=config.WARNING_YAWNING,
                        timeframe_count=timeframe_count
                    )
        else:
            YAWN_START = None
            YAWN_COUNTED = False

        # --- Drowsiness decision ---
        global DROWSY_COUNT, DROWSY_COUNTED
        drowsy = microsleep or (perclos >= config.PERCLOS_DROWSY) or yawning or frequent_closures
        
        if drowsy and not DROWSY_COUNTED:
            drowsy_count = _increment_event_counter("drowsy")
            DROWSY_COUNTED = True

            ######### Maintain event time queue #########
            global DROWSY_EVENT_TIME_ARRAY_SEC
            DROWSY_EVENT_TIME_ARRAY_SEC.append(now)
            timeframe_count = 0
            while (DROWSY_EVENT_TIME_ARRAY_SEC and (now - DROWSY_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                DROWSY_EVENT_TIME_ARRAY_SEC.popleft()
            timeframe_count = len(DROWSY_EVENT_TIME_ARRAY_SEC)

            logger.warning(f"DROWSINESS DETECTED! Total count: {drowsy_count}, PERCLOS: {perclos:.2f} Timeframe count: {timeframe_count}")

            ALERT_MANAGER.check_and_send_threshold_alert(
                tag="DROWSY_EVENT",
                event_type=config.BEHAVIOR_DROWSY,
                message=config.CONSOLE_DROWSY.format(drowsy_count),
                behavior_data={
                    "perclos": perclos,
                    "total_count": drowsy_count,
                },
                policy_key="drowsy",
                cycle_id=alert_cycle_id,
                current_count=drowsy_count,
                threshold=config.DROWSY_EVENT_COUNT_THRESH,
                send_cloud=True,
                trigger_voice=True,
                voice_message=config.VOICE_ALERT_DROWSY,
                trigger_buzzer=True,
                buzzer_message=config.WARNING_DROWSY,
                timeframe_count=timeframe_count
            )
        elif not drowsy:
            DROWSY_COUNTED = False

        behavior_data = {
            'drowsy': drowsy,
            'yawning': yawning,
            'microsleep': microsleep,
            'perclos': perclos,
            'blinks_per_min': blinks_per_min,
            'frequent_closures': frequent_closures,
            'closure_count': len(EYE_CLOSURE_EVENTS),
            'yawn_count': YAWN_COUNT,
            'drowsy_count': DROWSY_COUNT,
            'microsleep_count': MICROSLEEP_COUNT,
            'head_pose': None,  # Will be added below
            'distracted': False,  # Will be updated below
            'face_lost': False,
            'eye_aspect_ratio': eye_closed_score,
            'mouth_aspect_ratio': mouth_lower_down
        }
        
        # Add head pose detection
        if face_landmarks and image_width and image_height:
            head_turn_data = detect_head_turn_distraction(face_landmarks, image_width, image_height)
            behavior_data['head_pose'] = head_turn_data
            behavior_data['distracted'] = head_turn_data['is_turned']
                
        return behavior_data
    else:
        # No face detected - still check for head turn/distraction
        if image_width and image_height:
            head_turn_data = detect_head_turn_distraction(None, image_width, image_height)
            
            return {
                'drowsy': False,
                'yawning': False,
                'microsleep': False,
                'perclos': 0.0,
                'blinks_per_min': 0,
                'frequent_closures': False,
                'closure_count': 0,
                'yawn_count': YAWN_COUNT,
                'drowsy_count': DROWSY_COUNT,
                'microsleep_count': MICROSLEEP_COUNT,
                'head_pose': head_turn_data,
                'distracted': True,
                'face_lost': True,
                'eye_aspect_ratio': eye_closed_score,
                'mouth_aspect_ratio': mouth_lower_down
            }
        else:

            ######### Maintain event time queue #########
            global DISTRACTION_EVENT_TIME_ARRAY_SEC
            DISTRACTION_EVENT_TIME_ARRAY_SEC.append(time.time())
            timeframe_count = 0
            while (DISTRACTION_EVENT_TIME_ARRAY_SEC and (time.time() - DISTRACTION_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                DISTRACTION_EVENT_TIME_ARRAY_SEC.popleft()
            timeframe_count = len(DISTRACTION_EVENT_TIME_ARRAY_SEC)

            logger.warning("DISTRACTION DETECTED! Missing focus on the road. Timeframe count: {timeframe_count}")

            ALERT_MANAGER.check_and_send_threshold_alert(
                tag="DISTRACTION_EVENT",
                event_type=config.BEHAVIOR_DISTRACTION,
                message=config.CONSOLE_DISTRACTION,
                policy_key="head_turn",
                cycle_id=int(time.time() * 1000),
                send_cloud=False,
                trigger_voice=True,
                voice_message=config.VOICE_ALERT_DISTRACTION,
                trigger_buzzer=True,
                buzzer_message=config.WARNING_DISTRACTION,
                timeframe_count=timeframe_count
            )
            return {
                'drowsy': False,
                'yawning': False,
                'microsleep': False,
                'perclos': 0.0,
                'blinks_per_min': 0,
                'frequent_closures': False,
                'closure_count': 0,
                'yawn_count': YAWN_COUNT,
                'drowsy_count': DROWSY_COUNT,
                'microsleep_count': MICROSLEEP_COUNT,
                'head_pose': None,
                'distracted': True,
                'face_lost': True,
                'eye_aspect_ratio': eye_closed_score,
                'mouth_aspect_ratio': mouth_lower_down
            }


def run(model: str, num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera."""
    
    logger.info("=" * 80)
    logger.info("Starting SafeDriver Monitoring System...")
    logger.info("=" * 80)
    logger.info(f"Model path: {model}")
    logger.info(f"Camera ID: {camera_id}")
    logger.info(f"Resolution: {width}x{height}")
    logger.info(f"Max faces: {num_faces}")
    logger.info(f"Detection confidence: {min_face_detection_confidence}")
    logger.info(f"Presence confidence: {min_face_presence_confidence}")
    logger.info(f"Tracking confidence: {min_tracking_confidence}")
    logger.info("=" * 80)

    # Initialize camera
    logger.info(f"Initializing camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_id}")
        sys.exit(config.CAMERA_ERROR_MSG)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Camera initialized successfully - Actual resolution: {actual_width} x {actual_height}")

    global SCROLL_OFFSET, MAX_SCROLL

    def mouse_callback(event, x, y, flags, param):
        global SCROLL_OFFSET, MAX_SCROLL
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                SCROLL_OFFSET = max(0, SCROLL_OFFSET - config.SCROLL_STEP)
            else:
                SCROLL_OFFSET = min(MAX_SCROLL, SCROLL_OFFSET + config.SCROLL_STEP)

    if config.ENABLE_WINDOW:
        cv2.namedWindow(config.WINDOW_NAME)
        cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback)
        logger.info(f"Display window '{config.WINDOW_NAME}' created")

    def save_result(result: any,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        if COUNTER % config.FPS_AVG_FRAME_COUNT == 0:
            FPS = config.FPS_AVG_FRAME_COUNT / (time.time() - START_TIME)
            START_TIME = time.time()
            logger.debug(f"FPS: {FPS:.1f}")

        DETECTION_RESULT = result
        COUNTER += 1
        
        if COUNTER == 1:
            logger.info("First frame processed successfully")

    # Load model
    logger.info(f"Loading face landmarker model from: {model}")
    try:
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            result_callback=save_result)
        detector = vision.FaceLandmarker.create_from_options(options)
        logger.info("Face landmarker model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        cap.release()
        sys.exit(1)

    logger.info("Starting video processing loop...")
    logger.info("Press ESC to exit")
    
    if config.ENABLE_OBJECT_DETECTION:
        object_detector = frame_detect.DetectorProcess()   #create once

    frame_count = 0
    detection_failures = 0

    try:
        while cap.isOpened() and not stop_event.is_set():
            success, image = cap.read()
            frame_count += 1
            
            if not success:
                detection_failures += 1
                logger.error(f"Failed to read frame {frame_count}")
                if detection_failures > 10:
                    logger.error("Too many consecutive frame read failures, exiting...")
                    sys.exit(config.CAMERA_ERROR_MSG)
                continue
            
            detection_failures = 0

            # ======================= PREPROCESS ==========================
            image = cv2.flip(image, 1)

            # Small frame for YOLO (performance boost 🚀)
            small_frame = cv2.resize(image, (416, 416))

            # ======================= OBJECT DETECTION (ASYNC) ============
            if config.ENABLE_OBJECT_DETECTION:
                # Send only every 3rd frame to reduce load
                if frame_count % 3 == 0:
                    object_detector.submit_frame(small_frame)

            # ======================= MEDIAPIPE ===========================
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            detector.detect_async(mp_image, time.time_ns() // 1_000_000)

            # ======================= DISPLAY =============================
            current_frame = image

            if config.SHOW_FPS and config.ENABLE_WINDOW:
                fps_text = config.FPS_TEXT_FORMAT.format(FPS)
                text_location = (config.LEFT_MARGIN, config.ROW_SIZE + config.FPS_Y_OFFSET)

                cv2.putText(current_frame, fps_text, text_location,
                            config.FPS_FONT,
                            config.FPS_FONT_SIZE,
                            config.FPS_COLOR,
                            config.FPS_FONT_THICKNESS,
                            cv2.LINE_AA)

            if DETECTION_RESULT:
                # Draw landmarks
                if config.SHOW_FACE_MESH and config.ENABLE_WINDOW:
                    for face_landmarks in DETECTION_RESULT.face_landmarks:
                        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        face_landmarks_proto.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                            for landmark in face_landmarks
                        ])
                        mp_drawing.draw_landmarks(
                            image=current_frame,
                            landmark_list=face_landmarks_proto,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=current_frame,
                            landmark_list=face_landmarks_proto,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=current_frame,
                            landmark_list=face_landmarks_proto,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles
                            .get_default_face_mesh_iris_connections_style())

            # Expand right side for blendshapes
            if config.SHOW_BLENDSHAPES and config.ENABLE_WINDOW:
                current_frame = cv2.copyMakeBorder(current_frame, 0, 0, 0,
                                                   config.LABEL_PADDING_WIDTH,
                                                   cv2.BORDER_CONSTANT, None,
                                                   config.LABEL_BG_COLOR)

            if DETECTION_RESULT:
                face_blendshapes = DETECTION_RESULT.face_blendshapes
                face_landmarks = DETECTION_RESULT.face_landmarks[0] if DETECTION_RESULT.face_landmarks else None

                behavior_data = detect_driver_behavior(
                    face_blendshapes, 
                    current_frame.shape[0], 
                    current_frame,
                    face_landmarks=face_landmarks,
                    image_width=current_frame.shape[1],
                    image_height=current_frame.shape[0]
                )
                    
                if behavior_data:
                    # Draw metrics background
                    if config.SHOW_METRICS and config.ENABLE_WINDOW:
                        metrics_x = config.LEFT_MARGIN - config.METRICS_PADDING
                        metrics_y = config.ROW_SIZE + config.METRICS_Y_OFFSET
                        
                        overlay = current_frame.copy()
                        corner_radius = config.METRICS_CORNER_RADIUS
                        
                        # Draw rounded rectangle
                        cv2.rectangle(overlay,
                                    (metrics_x + corner_radius, metrics_y),
                                    (metrics_x + config.METRICS_WIDTH - corner_radius, 
                                     metrics_y + config.METRICS_HEIGHT),
                                    config.METRICS_BG_COLOR, -1)
                        cv2.rectangle(overlay,
                                    (metrics_x, metrics_y + corner_radius),
                                    (metrics_x + config.METRICS_WIDTH, 
                                     metrics_y + config.METRICS_HEIGHT - corner_radius),
                                    config.METRICS_BG_COLOR, -1)
                        
                        # Corner circles
                        for dx, dy in [(corner_radius, corner_radius),
                                       (config.METRICS_WIDTH - corner_radius, corner_radius),
                                       (corner_radius, config.METRICS_HEIGHT - corner_radius),
                                       (config.METRICS_WIDTH - corner_radius, config.METRICS_HEIGHT - corner_radius)]:
                            cv2.circle(overlay, (metrics_x + dx, metrics_y + dy),
                                     corner_radius, config.METRICS_BG_COLOR, -1)
                        
                        cv2.addWeighted(overlay, config.METRICS_BG_OPACITY, 
                                      current_frame, 1 - config.METRICS_BG_OPACITY, 0, current_frame)
                        
                        # Display metrics
                        metrics_data = [
                            (config.LABEL_PERCLOS.format(behavior_data['perclos']), config.PERCLOS_Y_OFFSET),
                            (config.LABEL_BLINKS.format(behavior_data['blinks_per_min']), config.BLINKS_Y_OFFSET),
                            (config.LABEL_CLOSURES.format(behavior_data['closure_count']), config.CLOSURES_Y_OFFSET),
                            (config.LABEL_YAWNS.format(behavior_data['yawn_count']), config.YAWNS_Y_OFFSET),
                            (config.LABEL_MICROSLEEPS.format(behavior_data['microsleep_count']), config.MICROSLEEPS_Y_OFFSET),
                            (config.LABEL_DROWSY_EVENTS.format(behavior_data['drowsy_count']), config.DROWSY_EVENTS_Y_OFFSET),
                            (config.LABEL_HEAD_POSE.format( behavior_data.get('head_pose', {}).get('direction', 'N/A')), config.HEAD_POSE_Y_OFFSET),
                            ('EAR : {:.2f}'.format(behavior_data['eye_aspect_ratio']), 180),
                            ('MAR : {:.2f}'.format(behavior_data['mouth_aspect_ratio']), 210)
                        ]
                        
                        for text, y_offset in metrics_data:
                            cv2.putText(current_frame, text,
                                       (config.LEFT_MARGIN, config.ROW_SIZE + y_offset),
                                       config.METRICS_FONT, config.METRICS_FONT_SIZE,
                                       config.METRICS_TEXT_COLOR, config.METRICS_FONT_THICKNESS, cv2.LINE_AA)
                        
                        # Display warnings
                        if config.SHOW_WARNINGS and config.ENABLE_WINDOW:
                            frame_width = current_frame.shape[1] - (config.LABEL_PADDING_WIDTH if config.SHOW_BLENDSHAPES else 0)
                            
                            warning_checks = [
                                (behavior_data['microsleep'], config.WARNING_MICROSLEEP, 
                                 config.CONSOLE_MICROSLEEP.format(behavior_data['microsleep_count'])),
                                (behavior_data['yawning'], config.WARNING_YAWNING,
                                 config.CONSOLE_YAWN.format(behavior_data['yawn_count'])),
                                (behavior_data['frequent_closures'], config.WARNING_FREQUENT_CLOSURES,
                                 config.CONSOLE_FREQUENT_CLOSURES),
                                (behavior_data.get('face_lost', False), config.WARNING_DISTRACTION,
                                 "Driver face not visible"),
                                (behavior_data.get('distracted', False) and not behavior_data.get('face_lost', False), 
                                 config.WARNING_HEAD_TURN,
                                 config.CONSOLE_HEAD_TURN.format(
                                     behavior_data.get('head_pose', {}).get('direction', 'UNKNOWN'),
                                     behavior_data.get('head_pose', {}).get('duration', 0)
                                 )),
                                (behavior_data['drowsy'], config.WARNING_DROWSY,
                                 config.CONSOLE_DROWSY.format(behavior_data['drowsy_count'])),
                            ]
                            
                            for condition, warning_text, console_msg in warning_checks:
                                if condition:
                                    (text_width, _), _ = cv2.getTextSize(warning_text,
                                                                         config.WARNING_FONT,
                                                                         config.WARNING_FONT_SIZE,
                                                                         config.WARNING_FONT_THICKNESS)
                                    right_x = frame_width - text_width - config.WARNING_RIGHT_MARGIN
                                    cv2.putText(current_frame, warning_text,
                                               (right_x, config.WARNING_Y_POSITION),
                                               config.WARNING_FONT, config.WARNING_FONT_SIZE,
                                               config.WARNING_COLOR, config.WARNING_FONT_THICKNESS, cv2.LINE_AA)
                                    break
                        
                        # Display head pose information if enabled
                        if config.SHOW_HEAD_POSE_DETAILS and behavior_data.get('head_pose'):
                            head_pose = behavior_data['head_pose']
                            pose_text = f"Yaw: {head_pose['yaw']:.1f}° \nPitch: {head_pose['pitch']:.1f}° \nRoll: {head_pose['roll']:.1f}°"
                            
                            cv2.putText(current_frame, pose_text,
                                       (config.LEFT_MARGIN, config.ROW_SIZE + config.HEAD_POSE_DETAILS_Y_OFFSET),
                                       config.METRICS_FONT, config.HEAD_POSE_FONT_SIZE,
                                       config.HEAD_POSE_COLOR, config.METRICS_FONT_THICKNESS, cv2.LINE_AA)

                    # Draw blendshapes
                    if config.SHOW_BLENDSHAPES and config.ENABLE_WINDOW:
                        legend_x = current_frame.shape[1] - config.LABEL_PADDING_WIDTH + config.BLENDSHAPE_X_OFFSET
                        legend_y = config.BLENDSHAPE_Y_START - SCROLL_OFFSET
                        bar_max_width = config.LABEL_PADDING_WIDTH - 40
                        
                        # Fix: Check if face_blendshapes exists and is not empty
                        if face_blendshapes and len(face_blendshapes) > 0:
                            num_blendshapes = len(face_blendshapes[0])
                            total_height = num_blendshapes * (config.BLENDSHAPE_BAR_HEIGHT + config.BLENDSHAPE_GAP_BETWEEN_BARS)
                            MAX_SCROLL = max(0, total_height - current_frame.shape[0] + 60)
                            
                            for category in face_blendshapes[0]:
                                if legend_y + config.BLENDSHAPE_BAR_HEIGHT > 0 and legend_y < current_frame.shape[0]:
                                    text = config.BLENDSHAPE_TEXT_FORMAT.format(category.category_name, round(category.score, 2))
                                    (text_width, _), _ = cv2.getTextSize(text, config.BLENDSHAPE_FONT,
                                                                        config.BLENDSHAPE_FONT_SIZE,
                                                                        config.BLENDSHAPE_FONT_THICKNESS)

                                    cv2.putText(current_frame, text,
                                                (legend_x, legend_y + (config.BLENDSHAPE_BAR_HEIGHT // 2) + 5),
                                                config.BLENDSHAPE_FONT, config.BLENDSHAPE_FONT_SIZE,
                                                config.BLENDSHAPE_TEXT_COLOR, config.BLENDSHAPE_FONT_THICKNESS, cv2.LINE_AA)

                                    bar_width = int(bar_max_width * category.score)
                                    cv2.rectangle(current_frame,
                                                (legend_x + text_width + config.BLENDSHAPE_TEXT_GAP, legend_y),
                                                (legend_x + text_width + config.BLENDSHAPE_TEXT_GAP + bar_width,
                                                legend_y + config.BLENDSHAPE_BAR_HEIGHT),
                                                config.BLENDSHAPE_BAR_COLOR, -1)

                                legend_y += (config.BLENDSHAPE_BAR_HEIGHT + config.BLENDSHAPE_GAP_BETWEEN_BARS)
            else:
                behavior_data = detect_driver_behavior(
                    None, 
                    None, 
                    None,
                    face_landmarks=None,
                    image_width=current_frame.shape[1],
                    image_height=current_frame.shape[0]
                )
                
                if behavior_data:
                    # Display warnings
                    if config.SHOW_WARNINGS and config.ENABLE_WINDOW:
                        frame_width = current_frame.shape[1] - (config.LABEL_PADDING_WIDTH if config.SHOW_BLENDSHAPES else 0)
                        warning_text = "FACE NOT VISIBLE - TURN BACK" if behavior_data.get('face_lost', False) else config.WARNING_DISTRACTION
                        (text_width, _), _ = cv2.getTextSize(warning_text,
                                                                config.WARNING_FONT,
                                                                config.WARNING_FONT_SIZE,
                                                                config.WARNING_FONT_THICKNESS)
                        right_x = frame_width - text_width - config.WARNING_RIGHT_MARGIN
                        cv2.putText(current_frame, warning_text,
                                    (right_x, config.WARNING_Y_POSITION),
                                    config.WARNING_FONT, config.WARNING_FONT_SIZE,
                                    config.WARNING_COLOR, config.WARNING_FONT_THICKNESS, cv2.LINE_AA)

            if config.ENABLE_WINDOW:
                cv2.imshow(config.WINDOW_NAME, current_frame)

            if cv2.waitKey(1) == 27:
                logger.info("ESC key pressed - Exiting...")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received - Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Final stats - Yawns: {YAWN_COUNT}, Microsleeps: {MICROSLEEP_COUNT}, Drowsy Events: {DROWSY_COUNT}")
        
        detector.close()
        cap.release()
        if config.ENABLE_WINDOW:
            cv2.destroyAllWindows()
        
        logger.info("SafeDriver Monitoring System stopped successfully")
        logger.info("=" * 80)


def main():

    # ============================================================================
    # FIREBASE INITIALIZATION
    # ============================================================================

    logger.info("=" * 80)
    logger.info("SafeDriver Monitoring System Detector Starting...")
    logger.info("=" * 80)

    logger.info("Initializing Firebase Admin SDK in detect.py...")
    try:
        # Check if Firebase app is already initialized
        firebase_admin.get_app()
        logger.info("Firebase Admin SDK already initialized in detect.py")
    except ValueError:
        # Initialize Firebase if not already done
        cred = credentials.Certificate(util.resource_path("firebase-admin-sdk/serviceAccountKey.json")) #safe-driver-system-b3da24192be1
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://safe-driver-system-default-rtdb.firebaseio.com/'
        })
        logger.info("Firebase Admin SDK initialized successfully in detect.py")

    # Import firestore_helper after Firebase is initialized
    from database.firestore_helper import firestore_helper
    logger.info("Imported firestore_helper module successfully in detect.py")

    # ============================================================================
    # GLOBAL VARIABLES AND CONSTANTS
    # ============================================================================

    # load configurations
    utils.get_model_configurations(logger)

    # Log configuration on startup
    utils.log_config(logger)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of face landmarker model.',
        required=False,
        default=util.resource_path('model/face_landmarker.task'))
    parser.add_argument(
        '--numFaces',
        help='Max number of faces that can be detected by the landmarker.',
        required=False,
        default=1)
    parser.add_argument(
        '--minFaceDetectionConfidence',
        help='The minimum confidence score for face detection to be considered successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minFacePresenceConfidence',
        help='The minimum confidence score of face presence score in the face landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the face tracking to be considered successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=720)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=960)
    
    args, _ = parser.parse_known_args()

    run(args.model, int(args.numFaces), args.minFaceDetectionConfidence,
        args.minFacePresenceConfidence, args.minTrackingConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()