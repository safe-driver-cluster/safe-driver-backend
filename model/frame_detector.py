import multiprocessing as mp
import cv2
import logging
import sys

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

def detector_worker(frame_queue):
    from ultralytics import YOLO

    # Load models INSIDE process
    detect_model = YOLO("model/yolov8n.pt")
    cigarette_model = YOLO("model/cigarette_model.pt")
    glasses_model = YOLO("model/glasses_model.pt")

    logger.info("Detection process started")

    while True:
        frame = frame_queue.get()

        if frame is None:
            logger.info("Detection process stopping...")
            break

        try:
            # -------------------------------
            # 1. OBJECT DETECTION (phone, bottle)
            # -------------------------------
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

                        logger.info(f"{label} detected: {conf:.2f}")

            # -------------------------------
            # 2. SMOKING CLASSIFICATION
            # -------------------------------
            # class_results = class_model(frame)

            # for r in class_results:
            #     probs = r.probs.data.tolist()
            #     class_id = probs.index(max(probs))
            #     confidence = max(probs)

            #     label = class_model.names[class_id]

            #     # Show classification result
            #     text = f"{label} ({confidence:.2f})"
            #     cv2.putText(frame, text, (20, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 1, (255,255,0), 2)

            #     # Smoking alert
            #     if label == "smoking" and confidence > 0.7:
            #         cv2.putText(frame, "🚬 SMOKING DETECTED!",
            #                     (20, 100),
            #                     cv2.FONT_HERSHEY_SIMPLEX,
            #                     1, (0,0,255), 3)

            #         print("ALERT: Driver is smoking!")

            # -------------------------------
            # 3. CIGARETTE DETECTION
            # -------------------------------

            results = cigarette_model(frame, conf=0.3)

            for r in results:
                for box in r.boxes:
                    label = cigarette_model.names[int(box.cls[0])]
                    conf = float(box.conf[0])

                    if label == "cigarette" and conf >= 0.25:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                        cv2.putText(frame, f"Cigarette {conf:.2f}",
                                    (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0,0,255), 2)

                        logger.info("Cigarette detected!")

            # -------------------------------
            # 4. GLASSES DETECTION
            # -------------------------------

            glass_results = glasses_model(frame, conf=0.3)

            for r in glass_results:
                for box in r.boxes:
                    label = glasses_model.names[int(box.cls[0])]
                    conf = float(box.conf[0])

                    if label == "glasses":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                        cv2.putText(frame, f"Glasses {conf:.2f}",
                                    (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255,0,0), 2)

                        logger.info("Glasses detected!")
            # -------------------------------
            # SHOW FRAME
            # -------------------------------
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