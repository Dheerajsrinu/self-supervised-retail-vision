# train_yolo.py
import argparse
import torch
from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"C:/Users/satyasrp/personal/projects/aiml/image/shelves-5/data.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=-1)  # auto batch size
    parser.add_argument("--model", default="yolov8s.pt", help="backbone checkpoint to start from")
    args = parser.parse_args()

    # Select device
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="shelf_detector_v1",
        pretrained=True,
        patience=25,
        optimizer="AdamW",
        lr0=0.0007,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.7,
        device=device,
        save=True,
    )

    # Correct folder name
    print("Training finished!")
    print("Best weights saved at: runs/train/shelf_detector_v1/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    main()
