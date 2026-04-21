from ultralytics import YOLO
import cv2

# Load models
detect_model = YOLO("model/yolov8n.pt")          # object detection
class_model = YOLO("model/smoking_model.pt")     # smoking classification
cigarette_model = YOLO("model/cigarette_model.pt") # cigarette detection

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # 1. OBJECT DETECTION (phone, bottle)
    # -------------------------------
    detect_results = detect_model(frame)

    for r in detect_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = detect_model.names[cls]
            conf = float(box.conf[0])

            if label in ["cell phone", "bottle"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

                print(f"{label} detected: {conf:.2f}")

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

            if label == "cigarette":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, f"Cigarette {conf:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,0,255), 2)

                print("🚬 Cigarette detected!")

    # -------------------------------
    # SHOW FRAME
    # -------------------------------
    cv2.imshow("Safe Driver System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()