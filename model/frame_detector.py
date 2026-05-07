import multiprocessing as mp
import cv2
import logging
import sys
import time
import config.config as config
from model.alerts import AlertManager
import model.utilmethods as utils

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

def detector_worker(frame_queue):
    from ultralytics import YOLO

    # Load models INSIDE process
    detect_model = YOLO("model/yolov8n.pt")
    cigarette_model = YOLO("model/cigarette_model.pt")
    glasses_model = YOLO("model/glasses_model.pt")

    logger.info("Detection process started.")

    while True:
        frame = frame_queue.get()

        if frame is None:
            logger.info("Detection process stopping...")
            break

        try:
            # -------------------------------------------------------------------------------------
            # 1. OBJECT DETECTION (phone, bottle)
            # -------------------------------------------------------------------------------------
            if config.ENABLE_PHONE_BOTTLE_PERSON_DETECTION:
                detect_results = detect_model(frame)

                for r in detect_results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = detect_model.names[cls]
                        conf = float(box.conf[0])

                        if label in ["cell phone", "bottle", "person"]:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}",
                                        (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,255,0), 2)

                            if config.ENABLE_LOGGING:
                                logger.info(f"{label} detected: {conf:.2f}")
                            
                            if label == "cell phone":
                                DETECT_PHONE_COUNT += 1
                            elif label == "bottle":
                                DETECT_BOTTLE_COUNT += 1

            # -------------------------------------------------------------------------------------
            # 2. CIGARETTE DETECTION
            # -------------------------------------------------------------------------------------

            if config.ENABLE_CIGARETTE_DETECTION:
                results = cigarette_model(frame, conf=0.3)

                for r in results:
                    for box in r.boxes:
                        label = cigarette_model.names[int(box.cls[0])]
                        conf = float(box.conf[0])

                        if label == "cigarette" and conf >= 0.25:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            if config.ENABLE_CV2_WINDOW:
                                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                                cv2.putText(frame, f"Cigarette {conf:.2f}",
                                            (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0,0,255), 2)

                            if config.ENABLE_LOGGING:
                                logger.info("Cigarette detected!")

                            DETECT_CIGARETTE_COUNT += 1

            # -------------------------------------------------------------------------------------
            # 3. GLASSES DETECTION
            # -------------------------------------------------------------------------------------
            if config.ENABLE_GLASSES_DETECTION:
                glass_results = glasses_model(frame, conf=0.3)

                for r in glass_results:
                    for box in r.boxes:
                        label = glasses_model.names[int(box.cls[0])]
                        conf = float(box.conf[0])

                        if label == "glasses":
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            if config.ENABLE_CV2_WINDOW:
                                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                                cv2.putText(frame, f"Glasses {conf:.2f}",
                                            (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255,0,0), 2)

                            if config.ENABLE_LOGGING:
                                logger.info("Glasses detected!")

                            DETECT_GLASSES_COUNT += 1
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