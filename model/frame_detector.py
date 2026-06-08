from collections import deque
import cv2
import logging
import platform
import sys
import time

import config.config as config
from model.alerts import AlertManager
import model.utilmethods as utils
import utils.utils as util

import queue
import threading

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
    try:
        if platform.system().lower() == "linux":
            # PyTorch/OpenCV otherwise create several native worker threads.
            # On a Raspberry Pi this competes heavily with MediaPipe and can
            # trigger an OOM/native abort that produces no Python traceback.
            import torch

            torch.set_num_threads(config.OBJECT_DETECTION_TORCH_THREADS_LINUX)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            cv2.setNumThreads(1)

        from ultralytics import YOLO

        # Only load enabled models. Loading every model at once creates a large
        # memory spike on Raspberry Pi even when a detector is disabled.
        detect_model = (
            YOLO(util.resource_path("model/yolov8n.pt"))
            if config.ENABLE_PHONE_BOTTLE_PERSON_DETECTION
            else None
        )
        cigarette_model = (
            YOLO(util.resource_path("model/cigarette_model.pt"))
            if config.ENABLE_CIGARETTE_DETECTION
            else None
        )
        glasses_model = (
            YOLO(util.resource_path("model/glasses_model.pt"))
            if config.ENABLE_GLASSES_DETECTION
            else None
        )

        inference_size = (
            config.OBJECT_DETECTION_IMGSZ_LINUX
            if platform.system().lower() == "linux"
            else config.OBJECT_DETECTION_IMGSZ
        )

        logger.info(
            "Object Detection worker started (imgsz=%s, phone/bottle=%s, cigarette=%s, glasses=%s).",
            inference_size,
            detect_model is not None,
            cigarette_model is not None,
            glasses_model is not None,
        )

        global DETECT_PHONE, DETECT_BOTTLE, DETECT_CIGARETTE, DETECT_GLASSES, DETECT_PHONE_COUNT, DETECT_BOTTLE_COUNT, DETECT_CIGARETTE_COUNT, DETECT_GLASSES_COUNT
        global frame_count

        while True:
            try:
                frame = frame_queue.get(timeout=5)
            except queue.Empty:
                logger.warning("Detector: no frame received for 5s, still waiting...")
                continue
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
                    detect_results = detect_model(
                        frame,
                        conf=config.YOLO_MODEL_PHONE_BOTTLE_PERSON_CONFIDENCE_THRESHOLD,
                        imgsz=inference_size,
                        verbose=False,
                    )

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
                    results = cigarette_model(
                        frame,
                        conf=config.YOLO_MODEL_CIGARETTE_CONFIDENCE_THRESHOLD,
                        imgsz=inference_size,
                        verbose=False,
                    )

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
                    glass_results = glasses_model(
                        frame,
                        conf=config.YOLO_MODEL_GLASSES_CONFIDENCE_THRESHOLD,
                        imgsz=inference_size,
                        verbose=False,
                    )

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

                                logger.info("Glasses detected!")
                                detected = True

                    # 👉 If not detected → use center zoom (your idea 🔥)
                    if not detected:
                        crop, (ox, oy) = center_crop(frame, zoom=1.8)

                        resized = cv2.resize(crop, (416, 416))
                        zoom_results = glasses_model(
                            resized,
                            conf=config.YOLO_MODEL_GLASSES_CONFIDENCE_THRESHOLD,
                            imgsz=inference_size,
                            verbose=False,
                        )

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

                                    logger.info("Glasses detected (zoom)!")

                # -------------------------------------------------------------------------------------
                # SHOW FRAME
                # -------------------------------------------------------------------------------------
                if config.ENABLE_CV2_WINDOW:
                    cv2.imshow("Safe Driver System", frame)

            except Exception:
                logger.exception("Object detection inference failed")

    except KeyboardInterrupt:         # ← catch the interrupt cleanly
        logger.info("Object detection process interrupted - shutting down cleanly")
    except Exception as e:
        logger.info(f"Detection process error: {e}", exc_info=True)
    finally:
        logger.info("Object detection process stopped")


class DetectorProcess:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)  # ← regular queue
        self._stop = False
        
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="detector-thread"
        )
        self.thread.start()

    def _run(self):
        detector_worker(self.frame_queue)

    def is_alive(self):
        return self.thread.is_alive()

    def submit_frame(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        try:
            self.frame_queue.put_nowait(frame)
        except:
            pass

    def stop(self):
        # Force None into queue by clearing it first
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        try:
            self.frame_queue.put(None, timeout=2)  # blocking put, guarantees delivery
        except:
            pass
        
        self.thread.join(timeout=5)
        logger.info("Object detector thread stopped")
