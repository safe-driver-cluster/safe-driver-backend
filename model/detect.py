import argparse
import sys
import time
import logging
import json

import numpy as np
from collections import deque

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import model.utilmethods as utils
import config.config as config

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

# ============================================================================
# END LOGGING CONFIGURATION
# ============================================================================

# Log configuration on startup
utils.log_config(logger)

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

# Event counters
YAWN_COUNT = 0
DROWSY_COUNT = 0
MICROSLEEP_COUNT = 0
YAWN_COUNTED = False
MICROSLEEP_COUNTED = False
DROWSY_COUNTED = False
FREQUENT_CLOSURES_COUNTED = False

# Scroll variables
SCROLL_OFFSET = 0
MAX_SCROLL = 0


def _bs_score(blendshapes, name: str) -> float:
    """Return blendshape score by name or 0.0 if missing."""
    for c in blendshapes:
        if c.category_name == name:
            return float(c.score)
    return 0.0


def send_behavior_to_parent(tag="BEHAVIOR_EVENT", type="behavior", message="", time=None, behavior_data={}):
    """Send behavior data to parent process via stdout"""
    try:
        # Create a structured message
        message_dict = {
            "tag": tag,
            "type": type,
            "message": message,
            "time": time or utils.now(),
            "data": behavior_data
        }
        # Write JSON to stdout with a newline delimiter
        sys.stdout.write(f"BEHAVIOR_DATA:{json.dumps(message_dict)}\n")
        sys.stdout.flush()

    except Exception as e:
        # Log errors to stderr instead of stdout
        logger.error(f"Failed to send behavior data to parent: {e}")


def detect_driver_behavior(face_blendshapes: np.ndarray, height, current_frame) -> dict:
    """Detect driver drowsiness behaviors based on facial blendshapes."""
    if face_blendshapes:
        now = time.time()
        bs = face_blendshapes[0]

        # --- Eye & mouth signals from blendshapes ---
        blink_l = _bs_score(bs, "eyeBlinkLeft")
        blink_r = _bs_score(bs, "eyeBlinkRight")
        eye_closed_score = 0.5 * (blink_l + blink_r)

        # Use mouthLowerDownRight and mouthLowerDownLeft for yawn detection
        mouth_lower_down_r = _bs_score(bs, "mouthLowerDownRight")
        mouth_lower_down_l = _bs_score(bs, "mouthLowerDownLeft")
        mouth_lower_down = 0.5 * (mouth_lower_down_r + mouth_lower_down_l)

        # --- PERCLOS window maintenance ---
        is_closed = 1 if eye_closed_score > config.EYE_CLOSED_THRESH else 0
        PERCLOS_WIN.append((now, is_closed))
        while PERCLOS_WIN and (now - PERCLOS_WIN[0][0]) > config.PERCLOS_WIN_SEC:
            PERCLOS_WIN.popleft()
        perclos = (sum(v for _, v in PERCLOS_WIN) / len(PERCLOS_WIN)) if PERCLOS_WIN else 0.0
        if perclos >= config.PERCLOS_DROWSY:
            logger.warning(f"PERCLOS THRESHOLD REACHED! PERCLOS: {perclos:.2f} over last {config.PERCLOS_WIN_SEC}s")
            send_behavior_to_parent(
                tag="DROWSY_EVENT",
                type=config.BEHAVIOR_PERCLOS_REACHED,
                message=config.CONSOLE_PERCLOS_REACHED.format(perclos),
                time=utils.now(),
                behavior_data={
                    "perclos": perclos,
                    "time_window": config.PERCLOS_WIN_SEC
                }
            )

        # --- Eye closure frequency tracking ---
        global EYE_CLOSURE_EVENTS, EYE_PARTIAL_CLOSURE_START, FREQUENT_CLOSURES_COUNTED
        
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
            logger.warning(f"FREQUENT EYE CLOSURES DETECTED! {len(EYE_CLOSURE_EVENTS)} closures in {config.EYE_CLOSURE_FREQ_WIN}s")
            send_behavior_to_parent(
                tag="DROWSY_EVENT",
                type=config.BEHAVIOR_FREQUENT_CLOSURES,
                message=config.CONSOLE_FREQUENT_CLOSURES,
                time=utils.now(),
                behavior_data={
                    "closure_count": len(EYE_CLOSURE_EVENTS),
                    "time_window": config.EYE_CLOSURE_FREQ_WIN
                }
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
            MICROSLEEP_COUNT += 1
            MICROSLEEP_COUNTED = True
            duration = now - EYE_CLOSED_START
            logger.warning(f"MICROSLEEP DETECTED! Duration: {duration:.2f}s, Total count: {MICROSLEEP_COUNT}")
            send_behavior_to_parent(
                tag="DROWSY_EVENT",
                type=config.BEHAVIOR_MICROSLEEP,
                message=config.CONSOLE_MICROSLEEP.format(MICROSLEEP_COUNT),
                time=utils.now(),
                behavior_data={
                    "duration": duration,
                    "total_count": MICROSLEEP_COUNT
                }
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
                    YAWN_COUNT += 1
                    YAWN_COUNTED = True
                    duration = now - YAWN_START
                    logger.warning(f"YAWN DETECTED! Duration: {duration:.2f}s, Total count: {YAWN_COUNT}")
                    send_behavior_to_parent(
                        tag="DROWSY_EVENT",
                        type=config.BEHAVIOR_YAWN,
                        message=config.CONSOLE_YAWN.format(YAWN_COUNT),
                        time=utils.now(),
                        behavior_data={
                            "duration": duration,
                            "total_count": YAWN_COUNT
                        }
                    )
        else:
            YAWN_START = None
            YAWN_COUNTED = False

        # --- Drowsiness decision ---
        global DROWSY_COUNT, DROWSY_COUNTED
        drowsy = microsleep or (perclos >= config.PERCLOS_DROWSY) or yawning or frequent_closures
        
        if drowsy and not DROWSY_COUNTED:
            DROWSY_COUNT += 1
            DROWSY_COUNTED = True
            logger.warning(f"DROWSINESS DETECTED! Total count: {DROWSY_COUNT}, PERCLOS: {perclos:.2f}")
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
            'microsleep_count': MICROSLEEP_COUNT
        }
                
        return behavior_data

    return None

def run(model: str, num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera."""
    
    logger.info("=" * 80)
    logger.info("Starting SafeDriver Monitoring System")
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

    cv2.namedWindow(config.WINDOW_NAME)
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback)
    logger.info(f"Display window '{config.WINDOW_NAME}' created")

    def save_result(result: vision.FaceLandmarkerResult,
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
    
    frame_count = 0
    detection_failures = 0

    try:
        while cap.isOpened():
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

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            detector.detect_async(mp_image, time.time_ns() // 1_000_000)

            # Show FPS
            if config.SHOW_FPS:
                fps_text = config.FPS_TEXT_FORMAT.format(FPS)
                text_location = (config.LEFT_MARGIN, config.ROW_SIZE + config.FPS_Y_OFFSET)
                current_frame = image
                cv2.putText(current_frame, fps_text, text_location,
                            config.FPS_FONT, config.FPS_FONT_SIZE, 
                            config.FPS_COLOR, config.FPS_FONT_THICKNESS, cv2.LINE_AA)
            else:
                current_frame = image

            if DETECTION_RESULT:
                # Draw landmarks
                if config.SHOW_FACE_MESH:
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
            if config.SHOW_BLENDSHAPES:
                current_frame = cv2.copyMakeBorder(current_frame, 0, 0, 0,
                                                   config.LABEL_PADDING_WIDTH,
                                                   cv2.BORDER_CONSTANT, None,
                                                   config.LABEL_BG_COLOR)

            if DETECTION_RESULT:
                face_blendshapes = DETECTION_RESULT.face_blendshapes

                if face_blendshapes:
                    behavior_data = detect_driver_behavior(face_blendshapes, current_frame.shape[0], current_frame)
                    
                    if behavior_data:
                        # Draw metrics background
                        if config.SHOW_METRICS:
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
                            ]
                            
                            for text, y_offset in metrics_data:
                                cv2.putText(current_frame, text,
                                           (config.LEFT_MARGIN, config.ROW_SIZE + y_offset),
                                           config.METRICS_FONT, config.METRICS_FONT_SIZE,
                                           config.METRICS_TEXT_COLOR, config.METRICS_FONT_THICKNESS, cv2.LINE_AA)
                        
                        # Display warnings
                        if config.SHOW_WARNINGS:
                            frame_width = current_frame.shape[1] - (config.LABEL_PADDING_WIDTH if config.SHOW_BLENDSHAPES else 0)
                            
                            warning_checks = [
                                (behavior_data['microsleep'], config.WARNING_MICROSLEEP, 
                                 config.CONSOLE_MICROSLEEP.format(behavior_data['microsleep_count'])),
                                (behavior_data['yawning'], config.WARNING_YAWNING,
                                 config.CONSOLE_YAWN.format(behavior_data['yawn_count'])),
                                (behavior_data['frequent_closures'], config.WARNING_FREQUENT_CLOSURES,
                                 config.CONSOLE_FREQUENT_CLOSURES),
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
                    
                    # Draw blendshapes
                    if config.SHOW_BLENDSHAPES:
                        legend_x = current_frame.shape[1] - config.LABEL_PADDING_WIDTH + config.BLENDSHAPE_X_OFFSET
                        legend_y = config.BLENDSHAPE_Y_START - SCROLL_OFFSET
                        bar_max_width = config.LABEL_PADDING_WIDTH - 40
                        
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
        cv2.destroyAllWindows()
        
        logger.info("SafeDriver Monitoring System stopped successfully")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of face landmarker model.',
        required=False,
        default='model/face_landmarker.task')
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
    
    args = parser.parse_args()

    run(args.model, int(args.numFaces), args.minFaceDetectionConfidence,
        args.minFacePresenceConfidence, args.minTrackingConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()