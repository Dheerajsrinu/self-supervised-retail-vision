import argparse
import torch
from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"C:/Users/satyasrp/personal/projects/aiml/image/product_recognition/data.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument(
        "--model",
        default=r"C:/Users/satyasrp/personal/projects/aiml/image/models/yolo11m.pt"
    )
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="product_recognition_yolo11",
        pretrained=True,

        # === SPEED FIXES ===
        workers=8,
        mosaic=0,
        copy_paste=0,
        mixup=0,
        augment=False,

        # === OPTIMIZED TRAINING ===
        patience=30,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,

        device=device,
        save=True,
    )

    print("Training finished!")
    print("Best weights saved at: runs/train/product_recognition_yolo11/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    main()
