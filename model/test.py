from ultralytics import YOLO

model = YOLO("model/yolov8n.pt")

print(model.names)