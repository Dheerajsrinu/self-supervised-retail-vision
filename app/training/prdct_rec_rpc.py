import argparse
import torch
from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()

    # === YOUR PATHS (Update as needed) ===
    parser.add_argument(
        "--data",
        default=r"C:/Users/satyasrp/personal/projects/aiml/image/Groceries-7/data.yaml"
    )
    parser.add_argument(
        "--model",
        default=r"C:/Users/satyasrp/personal/projects/aiml/image/models/yolo11m_4dh.yaml"
    )

    # === TRAINING PARAMS ===
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()

    # Auto device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n device is : {device}\n")

    # Load YOLO model (YAML custom model)
    model = YOLO(args.model)

    print("Starting training...\n")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="rpc_yolov8_4dh",     # experiment name
        pretrained=True,

        # === MATCHING CLI ===
        lr0=0.001,
        optimizer="AdamW",

        # === OPTIONAL SPEED-UP SETTINGS ===
        workers=8,
        mosaic=0,
        copy_paste=0,
        mixup=0,
        augment=False,
        
        device=device,
        save=True,
        verbose=True,
    )

    print("\nTraining Completed Successfully!")
    print("Best weights saved at: runs/detect/rpc_yolov8_4dh/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    main()
