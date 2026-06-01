from ultralytics import YOLO

model = YOLO("model/yolov8n.pt")

logger.info(model.names)