"""Export SafeDriver YOLO .pt models to Raspberry Pi-friendly NCNN models."""

from pathlib import Path

from ultralytics import YOLO


MODEL_DIR = Path(__file__).resolve().parent
MODELS = (
    "yolov8n.pt",
    "cigarette_model.pt",
    "glasses_model.pt",
)


def main():
    for model_name in MODELS:
        model_path = MODEL_DIR / model_name
        print(f"Exporting {model_path} to NCNN...")
        YOLO(str(model_path)).export(format="ncnn", imgsz=320)

    print("NCNN exports completed.")


if __name__ == "__main__":
    main()
