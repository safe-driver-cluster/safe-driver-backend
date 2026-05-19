from ultralytics import YOLO
import cv2
import time

prev_time = 0
frame_count = 0

ENABLE_PHONE_BOTTLE_PERSON_DETECTION = True
ENABLE_CIGARETTE_DETECTION = True
ENABLE_GLASSES_DETECTION = False

DETECT_PHONE_BOTTLE_PERSON = 1
DETECT_CIGARATE = 1
DETECT_GLASSES = 7

# Load models
detect_model = YOLO("model/yolov8n.pt")          # object detection
class_model = YOLO("model/smoking_model.pt")     # smoking classification
cigarette_model = YOLO("model/cigarette_model.pt") # cigarette detection
glasses_model = YOLO("model/glasses_model.pt")     # glasses detection

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # -------------------------------
    # 1. OBJECT DETECTION (phone, bottle)
    # -------------------------------
    if ENABLE_PHONE_BOTTLE_PERSON_DETECTION and frame_count % DETECT_PHONE_BOTTLE_PERSON == 0:
        detect_results = detect_model(frame, conf=0.5)

        for r in detect_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = detect_model.names[cls]
                conf = float(box.conf[0])

                if label in ["cell phone", "bottle", "person"] and conf >= 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 2)

                    print(f"{label} detected: {conf:.2f}")

    # -------------------------------
    # 2. CIGARETTE DETECTION
    # -------------------------------
    if ENABLE_CIGARETTE_DETECTION and frame_count % DETECT_CIGARATE == 0:

        results = cigarette_model(frame, conf=0.7)

        for r in results:
            for box in r.boxes:
                label = cigarette_model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                if label == "cigarette" and conf >= 0.7:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(frame, f"Cigarette {conf:.2f}",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,0,255), 2)

                    print("Cigarette detected!")

    # -------------------------------
    # 3. GLASSES DETECTION
    # -------------------------------
    if ENABLE_GLASSES_DETECTION and frame_count % DETECT_GLASSES == 0:
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
        glass_results = glasses_model(frame, conf=0.7)

        for r in glass_results:
            for box in r.boxes:
                label = glasses_model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                if (label == "glasses" or label == "sunglasses") and conf >= 0.7:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

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
            zoom_results = glasses_model(resized, conf=0.7)

            scale_x = crop.shape[1] / 416
            scale_y = crop.shape[0] / 416

            for r in zoom_results:
                for box in r.boxes:
                    label = glasses_model.names[int(box.cls[0])]
                    conf = float(box.conf[0])

                    if (label == "glasses" or label == "sunglasses") and conf >= 0.7:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # map back to original frame
                        x1 = int(x1 * scale_x) + ox
                        x2 = int(x2 * scale_x) + ox
                        y1 = int(y1 * scale_y) + oy
                        y2 = int(y2 * scale_y) + oy

                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}",
                                    (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255,0,0), 2)

                        print("Glasses detected (zoom)!")

    # -------------------------------
    # FPS CALCULATION
    # -------------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    # -------------------------------
    # SHOW FRAME
    # -------------------------------
    cv2.imshow("Safe Driver System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()