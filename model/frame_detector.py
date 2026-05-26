from collections import deque
import multiprocessing as mp
import cv2
import logging
import sys
import time

from polars import duration
import config.config as config
from model.alerts import AlertManager
import model.utilmethods as utils
import utils.utils as util

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

DETECT_PHONE = False
DETECT_BOTTLE = False
DETECT_CIGARETTE = False
DETECT_GLASSES = False

DETECT_PHONE_COUNT = 0
DETECT_BOTTLE_COUNT = 0
DETECT_CIGARETTE_COUNT = 0
DETECT_GLASSES_COUNT = 0

TIME_START = utils.now()

LAST_COUNTER_EVENT_TIME = {
    config.BEHAVIOR_MOBILE_USE: None,
    config.BEHAVIOR_SMOKING: None,
    config.BEHAVIOR_DRINKING: None,
}

PHONE_EVENT_TIME_ARRAY_SEC = deque()
BOTTLE_EVENT_TIME_ARRAY_SEC = deque()
CIGARETTE_EVENT_TIME_ARRAY_SEC = deque()

frame_count = 0

# ============================================================================
# ALERT MANAGER
# ============================================================================
ALERT_MANAGER = AlertManager(
    logger=logger,
    now_provider=utils.now,
    output_stream=sys.stdout,
    threshold_defaults={
        config.BEHAVIOR_MOBILE_USE: False,
        config.BEHAVIOR_SMOKING: False,
        config.BEHAVIOR_DRINKING: False,
    },
)
# ============================================================================

def _reset_detect_counts(counter_key):
    global DETECT_PHONE_COUNT, DETECT_BOTTLE_COUNT, DETECT_CIGARETTE_COUNT, DETECT_GLASSES_COUNT
    DETECT_PHONE_COUNT = 0
    DETECT_BOTTLE_COUNT = 0
    DETECT_CIGARETTE_COUNT = 0
    DETECT_GLASSES_COUNT = 0

    LAST_COUNTER_EVENT_TIME[counter_key] = None
    ALERT_MANAGER.reset_event_state(counter_key)

def _increment_detect_count(counter_key):
    global DETECT_PHONE_COUNT, DETECT_BOTTLE_COUNT, DETECT_CIGARETTE_COUNT, DETECT_GLASSES_COUNT

    now_ts = time.time()
    last_ts = LAST_COUNTER_EVENT_TIME.get(counter_key)
    if last_ts is not None and (now_ts - last_ts) >= config.EVENT_COUNT_RESET_SEC:
        _reset_detect_counts(counter_key)

    if counter_key == config.BEHAVIOR_MOBILE_USE:
        DETECT_PHONE_COUNT += 1
        new_value = DETECT_PHONE_COUNT
    elif counter_key == config.BEHAVIOR_SMOKING:
        DETECT_CIGARETTE_COUNT += 1
        new_value = DETECT_CIGARETTE_COUNT
    elif counter_key == config.BEHAVIOR_DRINKING:
        DETECT_BOTTLE_COUNT += 1
        new_value = DETECT_BOTTLE_COUNT
    else:
        raise ValueError(f"Unknown counter key: {counter_key}")
    
    LAST_COUNTER_EVENT_TIME[counter_key] = now_ts
    return new_value
    
def detector_worker(frame_queue):
    from ultralytics import YOLO

    # Load models INSIDE process
    detect_model = YOLO(util.resource_path("model/yolov8n.pt"))  # safe-driver-system-b3da24192be1
    cigarette_model = YOLO(util.resource_path("model/cigarette_model.pt"))
    glasses_model = YOLO(util.resource_path("model/glasses_model.pt"))

    logger.info("Object Detection process started.")

    global DETECT_PHONE, DETECT_BOTTLE, DETECT_CIGARETTE, DETECT_GLASSES, DETECT_PHONE_COUNT, DETECT_BOTTLE_COUNT, DETECT_CIGARETTE_COUNT, DETECT_GLASSES_COUNT
    global frame_count

    while True:
        frame = frame_queue.get()
        now = time.time()

        if frame is None:
            logger.info("Detection process stopping...")
            break

        frame_count += 1

        try:
            # -------------------------------------------------------------------------------------
            # 1. OBJECT DETECTION (phone, bottle)
            # -------------------------------------------------------------------------------------
            if config.ENABLE_PHONE_BOTTLE_PERSON_DETECTION and frame_count % config.DETECT_PHONE_BOTTLE_PERSON_FRAME == 0:
                detect_results = detect_model(frame, conf=config.YOLO_MODEL_PHONE_BOTTLE_PERSON_CONFIDENCE_THRESHOLD, verbose=False)

                for r in detect_results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = detect_model.names[cls]
                        conf = float(box.conf[0])

                        if label in ["cell phone", "bottle"] and conf >= config.YOLO_MODEL_PHONE_BOTTLE_PERSON_CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            if config.ENABLE_CV2_WINDOW:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                                cv2.putText(frame, f"{label} {conf:.2f}",
                                            (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0,255,0), 2)

                            
                            
                            if label == "cell phone":
                                phone_detect_count = _increment_detect_count("mobile_use")
                                ######### Maintain event time queue #########
                                global PHONE_EVENT_TIME_ARRAY_SEC
                                PHONE_EVENT_TIME_ARRAY_SEC.append(now)
                                timeframe_count = 0
                                while (PHONE_EVENT_TIME_ARRAY_SEC and (now - PHONE_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                                    PHONE_EVENT_TIME_ARRAY_SEC.popleft()
                                timeframe_count = len(PHONE_EVENT_TIME_ARRAY_SEC)

                                if config.ENABLE_LOGGING:
                                    logger.info(f"{label} detected: {conf:.2f} timeframe_count: {timeframe_count}")

                                ALERT_MANAGER.check_and_send_threshold_alert(
                                    tag="DETECTION_EVENT",
                                    event_type=config.BEHAVIOR_MOBILE_USE,
                                    message=config.CONSOLE_MOBILE_USE,
                                    policy_key="mobile_use",
                                    cycle_id=int(now * 1000),
                                    current_count=phone_detect_count,
                                    threshold=config.THRESHOLD_PHONE_ALERT_TO_CLOUD,
                                    send_cloud=True,
                                    trigger_voice=True,
                                    voice_message=config.VOICE_ALERT_PHONE,
                                    trigger_buzzer=True,
                                    buzzer_message=config.WARNING_MOBILE_USE,
                                    timeframe_count=timeframe_count,
                                )
                            elif label == "bottle":
                                bottle_detect_count = _increment_detect_count("drinking")
                                ######### Maintain event time queue #########
                                global BOTTLE_EVENT_TIME_ARRAY_SEC
                                BOTTLE_EVENT_TIME_ARRAY_SEC.append(now)
                                timeframe_count = 0
                                while (BOTTLE_EVENT_TIME_ARRAY_SEC and (now - BOTTLE_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                                    BOTTLE_EVENT_TIME_ARRAY_SEC.popleft()
                                timeframe_count = len(BOTTLE_EVENT_TIME_ARRAY_SEC)

                                if config.ENABLE_LOGGING:
                                    logger.info(f"{label} detected: {conf:.2f} timeframe_count: {timeframe_count}")

                                ALERT_MANAGER.check_and_send_threshold_alert(
                                    tag="DETECTION_EVENT",
                                    event_type=config.BEHAVIOR_DRINKING,
                                    message=config.CONSOLE_DRINKING,
                                    policy_key="drinking",
                                    cycle_id=int(now * 1000),
                                    current_count=bottle_detect_count,
                                    threshold=config.THRESHOLD_BOTTLE_ALERT_TO_CLOUD,
                                    send_cloud=True,
                                    trigger_voice=True,
                                    voice_message=config.VOICE_ALERT_DRINKING,
                                    trigger_buzzer=True,
                                    buzzer_message=config.WARNING_DRINKING,
                                    timeframe_count=timeframe_count,
                                )

            # -------------------------------------------------------------------------------------
            # 2. CIGARETTE DETECTION
            # -------------------------------------------------------------------------------------

            if config.ENABLE_CIGARETTE_DETECTION and frame_count % config.DETECT_CIGARETTE_FRAME == 0:
                results = cigarette_model(frame, conf=config.YOLO_MODEL_CIGARETTE_CONFIDENCE_THRESHOLD, verbose=False)

                for r in results:
                    for box in r.boxes:
                        label = cigarette_model.names[int(box.cls[0])]
                        conf = float(box.conf[0])

                        if label == "cigarette" and conf >= config.YOLO_MODEL_CIGARETTE_CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            if config.ENABLE_CV2_WINDOW:
                                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                                cv2.putText(frame, f"Cigarette {conf:.2f}",
                                            (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0,0,255), 2)

                            cigarette_detect_count = _increment_detect_count("smoking")
                            ########## Maintain event time queue #########
                            global CIGARETTE_EVENT_TIME_ARRAY_SEC
                            CIGARETTE_EVENT_TIME_ARRAY_SEC.append(now)
                            timeframe_count = 0
                            while (CIGARETTE_EVENT_TIME_ARRAY_SEC and (now - CIGARETTE_EVENT_TIME_ARRAY_SEC[0]) > config.EVENT_ARRAY_TIME_WINDOW_SEC):
                                CIGARETTE_EVENT_TIME_ARRAY_SEC.popleft()
                            timeframe_count = len(CIGARETTE_EVENT_TIME_ARRAY_SEC)

                            if config.ENABLE_LOGGING:
                                logger.info(f"{label} detected: {conf:.2f} timeframe_count: {timeframe_count}")

                            ALERT_MANAGER.check_and_send_threshold_alert(
                                tag="DETECTION_EVENT",
                                event_type=config.BEHAVIOR_SMOKING,
                                message=config.CONSOLE_SMOKING,
                                policy_key="smoking",
                                cycle_id=int(now * 1000),
                                current_count=cigarette_detect_count,
                                threshold=config.THRESHOLD_CIGARETTE_ALERT_TO_CLOUD,
                                send_cloud=True,
                                trigger_voice=True,
                                voice_message=config.VOICE_ALERT_SMOKING,
                                trigger_buzzer=True,
                                buzzer_message=config.WARNING_SMOKING,
                                timeframe_count=timeframe_count,
                            )

            # -------------------------------
            # 3. GLASSES DETECTION
            # -------------------------------
            if config.ENABLE_GLASSES_DETECTION and frame_count % config.DETECT_GLASSES_FRAME == 0:
                # -------------------------------
                # 4. GLASSES DETECTION (IMPROVED)
                # -------------------------------

                def center_crop(frame, zoom=1.8):
                    h, w, _ = frame.shape
                    new_w = int(w / zoom)
                    new_h = int(h / zoom)

                    x1 = (w - new_w) // 2
                    y1 = (h - new_h) // 2
                    x2 = x1 + new_w
                    y2 = y1 + new_h

                    return frame[y1:y2, x1:x2], (x1, y1)


                detected = False

                # 👉 Try normal detection first (lower threshold for better recall)
                glass_results = glasses_model(frame, conf=config.YOLO_MODEL_GLASSES_CONFIDENCE_THRESHOLD, verbose=False)

                for r in glass_results:
                    for box in r.boxes:
                        label = glasses_model.names[int(box.cls[0])]
                        conf = float(box.conf[0])

                        if (label == "glasses" or label == "sunglasses") and conf >= config.YOLO_MODEL_GLASSES_CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            if config.ENABLE_CV2_WINDOW:
                                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                                cv2.putText(frame, f"{label} {conf:.2f}",
                                            (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255,0,0), 2)

                            print("Glasses detected!")
                            detected = True

                # 👉 If not detected → use center zoom (your idea 🔥)
                if not detected:
                    crop, (ox, oy) = center_crop(frame, zoom=1.8)

                    resized = cv2.resize(crop, (416, 416))
                    zoom_results = glasses_model(resized, conf=config.YOLO_MODEL_GLASSES_CONFIDENCE_THRESHOLD, verbose=False)

                    scale_x = crop.shape[1] / 416
                    scale_y = crop.shape[0] / 416

                    for r in zoom_results:
                        for box in r.boxes:
                            label = glasses_model.names[int(box.cls[0])]
                            conf = float(box.conf[0])

                            if (label == "glasses" or label == "sunglasses") and conf >= config.YOLO_MODEL_GLASSES_CONFIDENCE_THRESHOLD:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])

                                # map back to original frame
                                x1 = int(x1 * scale_x) + ox
                                x2 = int(x2 * scale_x) + ox
                                y1 = int(y1 * scale_y) + oy
                                y2 = int(y2 * scale_y) + oy

                                if config.ENABLE_CV2_WINDOW:
                                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                                    cv2.putText(frame, f"{label} {conf:.2f}",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, (255,0,0), 2)

                                print("Glasses detected (zoom)!")

            # -------------------------------------------------------------------------------------
            # SHOW FRAME
            # -------------------------------------------------------------------------------------
            if config.ENABLE_CV2_WINDOW:
                cv2.imshow("Safe Driver System", frame)

        except Exception as e:
            logger.info("Detection error:", e)


class DetectorProcess:
    def __init__(self):
        self.frame_queue = mp.Queue(maxsize=1)

        self.process = mp.Process(
            target=detector_worker,
            args=(self.frame_queue,),
            daemon=True
        )
        self.process.start()

    def submit_frame(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()  # drop old frame
            except:
                pass

        self.frame_queue.put(frame)

    def stop(self):
        self.frame_queue.put(None)
        self.process.join()