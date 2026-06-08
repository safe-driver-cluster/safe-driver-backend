from collections import deque
import cv2
import logging
import multiprocessing
import os
import platform
import sys
import time

import config.config as config
from model.alerts import AlertManager
import model.utilmethods as utils
import utils.utils as util

import queue
import threading
from shared import behavior_queue

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


def _cloud_status(timeframe_count, threshold):
    if timeframe_count < threshold:
        return "below_threshold"
    if timeframe_count <= config.CLOUD_ALERT_TIMEFRAME_COUNT_LIMIT:
        return "cloud_allowed"
    return "cloud_limit_reached"


def _log_object_detection_event(
    *,
    label,
    event_type,
    confidence,
    event_count,
    timeframe_count,
    threshold,
    bbox,
    frame_count,
    inference_size,
):
    if not config.ENABLE_LOGGING:
        return

    logger.info(
        (
            "Object detection event: label=%s event_type=%s confidence=%.2f "
            "event_count=%s timeframe_count=%s/%s cloud_threshold=%s "
            "cloud_status=%s"
        ),
        label,
        event_type,
        confidence,
        event_count,
        timeframe_count,
        config.CLOUD_ALERT_TIMEFRAME_COUNT_LIMIT,
        threshold,
        _cloud_status(timeframe_count, threshold),
    )
    
def _forward_behavior_events(output_queue):
    """Forward object alerts from a subprocess-local queue to its parent."""
    if output_queue is None:
        return

    while True:
        try:
            output_queue.put_nowait(behavior_queue.get_nowait())
        except queue.Empty:
            break
        except queue.Full:
            logger.warning("Object behavior output queue is full; dropping alert")
            break


def _resolve_model_path(default_path, linux_path):
    """Select an ARM-friendly model backend on Linux."""
    if platform.system().lower() != "linux":
        return util.resource_path(default_path)

    resolved_linux_path = util.resource_path(linux_path)
    if os.path.exists(resolved_linux_path):
        return resolved_linux_path

    if config.OBJECT_DETECTION_ALLOW_PYTORCH_FALLBACK_LINUX:
        logger.warning(
            "Linux model %s was not found; falling back to PyTorch model %s",
            resolved_linux_path,
            default_path,
        )
        return util.resource_path(default_path)

    logger.error(
        "Linux object model not found: %s. Export the .pt model to NCNN; "
        "PyTorch fallback is disabled because it previously exited with SIGILL.",
        resolved_linux_path,
    )
    return None


def detector_worker(frame_queue, output_queue=None):
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
        detect_model_path = (
            _resolve_model_path(
                config.YOLO_MODEL_PHONE_BOTTLE_PERSON,
                config.YOLO_MODEL_PHONE_BOTTLE_PERSON_LINUX,
            )
            if config.ENABLE_PHONE_BOTTLE_PERSON_DETECTION
            else None
        )
        cigarette_model_path = (
            _resolve_model_path(
                config.YOLO_MODEL_CIGARETTE,
                config.YOLO_MODEL_CIGARETTE_LINUX,
            )
            if config.ENABLE_CIGARETTE_DETECTION
            else None
        )
        glasses_model_path = (
            _resolve_model_path(
                config.YOLO_MODEL_GLASSES,
                config.YOLO_MODEL_GLASSES_LINUX,
            )
            if config.ENABLE_GLASSES_DETECTION
            else None
        )

        detect_model = (
            YOLO(detect_model_path, task="detect")
            if config.ENABLE_PHONE_BOTTLE_PERSON_DETECTION and detect_model_path
            else None
        )
        cigarette_model = (
            YOLO(cigarette_model_path, task="detect")
            if config.ENABLE_CIGARETTE_DETECTION and cigarette_model_path
            else None
        )
        glasses_model = (
            YOLO(glasses_model_path, task="detect")
            if config.ENABLE_GLASSES_DETECTION and glasses_model_path
            else None
        )

        if detect_model is None and cigarette_model is None and glasses_model is None:
            logger.error("No compatible object detection models are available; object worker is stopping")
            return

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
                if detect_model is not None and frame_count % config.DETECT_PHONE_BOTTLE_PERSON_FRAME == 0:
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

                                    _log_object_detection_event(
                                        label=label,
                                        event_type=config.BEHAVIOR_MOBILE_USE,
                                        confidence=conf,
                                        event_count=phone_detect_count,
                                        timeframe_count=timeframe_count,
                                        threshold=config.THRESHOLD_PHONE_ALERT_TO_CLOUD,
                                        bbox=(x1, y1, x2, y2),
                                        frame_count=frame_count,
                                        inference_size=inference_size,
                                    )

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

                                    _log_object_detection_event(
                                        label=label,
                                        event_type=config.BEHAVIOR_DRINKING,
                                        confidence=conf,
                                        event_count=bottle_detect_count,
                                        timeframe_count=timeframe_count,
                                        threshold=config.THRESHOLD_BOTTLE_ALERT_TO_CLOUD,
                                        bbox=(x1, y1, x2, y2),
                                        frame_count=frame_count,
                                        inference_size=inference_size,
                                    )

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

                if cigarette_model is not None and frame_count % config.DETECT_CIGARETTE_FRAME == 0:
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

                                _log_object_detection_event(
                                    label=label,
                                    event_type=config.BEHAVIOR_SMOKING,
                                    confidence=conf,
                                    event_count=cigarette_detect_count,
                                    timeframe_count=timeframe_count,
                                    threshold=config.THRESHOLD_CIGARETTE_ALERT_TO_CLOUD,
                                    bbox=(x1, y1, x2, y2),
                                    frame_count=frame_count,
                                    inference_size=inference_size,
                                )

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
                if glasses_model is not None and frame_count % config.DETECT_GLASSES_FRAME == 0:
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

                                if config.ENABLE_LOGGING:
                                    logger.info(
                                        (
                                            "Object detection event: label=%s event_type=glasses "
                                            "confidence=%.2f frame_count=%s imgsz=%s bbox=%s cloud_status=disabled"
                                        ),
                                        label,
                                        conf,
                                        frame_count,
                                        inference_size,
                                        (x1, y1, x2, y2),
                                    )
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

                                    if config.ENABLE_LOGGING:
                                        logger.info(
                                            (
                                                "Object detection event: label=%s event_type=glasses "
                                                "confidence=%.2f frame_count=%s imgsz=%s bbox=%s "
                                                "source=center_zoom cloud_status=disabled"
                                            ),
                                            label,
                                            conf,
                                            frame_count,
                                            inference_size,
                                            (x1, y1, x2, y2),
                                        )

                # -------------------------------------------------------------------------------------
                # SHOW FRAME
                # -------------------------------------------------------------------------------------
                if config.ENABLE_CV2_WINDOW:
                    cv2.imshow("Safe Driver System", frame)

            except Exception:
                logger.exception("Object detection inference failed")
            finally:
                _forward_behavior_events(output_queue)

    except KeyboardInterrupt:         # ← catch the interrupt cleanly
        logger.info("Object detection process interrupted - shutting down cleanly")
    except Exception as e:
        logger.info(f"Detection process error: {e}", exc_info=True)
    finally:
        logger.info("Object detection process stopped")


class DetectorProcess:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)  # ← regular queue
        self._use_process = platform.system().lower() == "linux"
        self.thread = None
        self.process = None
        self.output_queue = None

        if self._use_process:
            context = multiprocessing.get_context("spawn")
            self.frame_queue = context.Queue(maxsize=1)
            self.output_queue = context.Queue(maxsize=20)
            self.process = context.Process(
                target=detector_worker,
                args=(self.frame_queue, self.output_queue),
                daemon=True,
                name="object-detector-process",
            )
            self.process.start()
            logger.info("Object detector subprocess started with PID %s", self.process.pid)
        else:
            self.thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="detector-thread"
            )
            self.thread.start()

    def _run(self):
        detector_worker(self.frame_queue)

    def is_alive(self):
        if self._use_process:
            return self.process is not None and self.process.is_alive()
        return self.thread is not None and self.thread.is_alive()

    def exit_code(self):
        if self._use_process and self.process is not None:
            return self.process.exitcode
        return None

    def drain_events(self):
        if self.output_queue is None:
            return

        while True:
            try:
                behavior_queue.put(self.output_queue.get_nowait())
            except queue.Empty:
                break

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
        
        if self._use_process:
            self.process.join(timeout=5)
            if self.process.is_alive():
                logger.warning("Object detector subprocess did not stop; terminating it")
                self.process.terminate()
                self.process.join(timeout=2)
            logger.info("Object detector subprocess stopped with exit code %s", self.process.exitcode)
            self.frame_queue.close()
            self.output_queue.close()
        else:
            self.thread.join(timeout=5)
            logger.info("Object detector thread stopped")
