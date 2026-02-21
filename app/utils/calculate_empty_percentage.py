from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def intersection_area(boxA, boxB):
    """Compute intersection area between 2 bounding boxes"""
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    if x2 < x1 or y2 < y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def compute_empty_space(
    shelf_model_path: str,
    product_model_path: str,
    image_path: str,
    save_path: str = "output.jpg",
    conf: float = 0.25
):

    # Load models
    shelf_model = YOLO(shelf_model_path)
    product_model = YOLO(product_model_path)
    print("loaded models")

    # Load image
    img = Image.open(image_path)
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    print("loaded image")

    # Inference
    shelf_results = shelf_model.predict(image_path, conf=conf)[0]
    product_results = product_model.predict(image_path, conf=conf)[0]
    print("Inference")
    print(product_results)
    # Extract predictions
    shelves = [b.xyxy[0].tolist() for b in shelf_results.boxes]
    products = [b.xyxy[0].tolist() for b in product_results.boxes]
    print("Extract shelf")

    output = []

    # Draw product boxes in green
    for box in products:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 10), "", fill="green", font=font)

    # Process shelves
    for shelf_box in shelves:
        x1, y1, x2, y2 = shelf_box
        shelf_area = (x2 - x1) * (y2 - y1)

        used_area = sum(intersection_area(shelf_box, p) for p in products)

        empty_area = shelf_area - used_area
        empty_percentage = empty_area / shelf_area * 100

        # Save output metrics
        output.append({
            "shelf_box": shelf_box,
            "shelf_area": shelf_area,
            "product_area": used_area,
            "empty_area": empty_area,
            "empty_percentage": round(empty_percentage, 2)
        })

        # Draw shelf box in red
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)

        # Label shelf empty %
        text = f"Empty: {round(empty_percentage, 2)}%"
        draw.text((x1, y1 - 15), text, fill="blue", font=font)

    # Save annotated image
    img.save(save_path)
    print(f"Saved annotated image to {save_path}")

    return output


# Example usage
if __name__ == "__main__":
    # image_path="C:/Users/satyasrp/personal/projects/aiml/image/retail-shelf-availability-2/test/images/DSC05223_jpg.rf.2dc23870321b27b6fef89e45055ecbd7.jpg"
    image_path = "C:/Users/satyasrp/personal/projects/aiml/image/product_recognition/train/images/test_623_jpg.rf.52020998167f5a8e551ca90ce4077d1c.jpg"
    result = compute_empty_space(
        shelf_model_path="C:/Users/satyasrp/personal/projects/aiml/runs/detect/shelf_detector_v14/weights/best.pt",
        product_model_path="C:/Users/satyasrp/personal/projects/aiml/runs/detect/product_recognition_yolo11/weights/best.pt",
        image_path=image_path,
        save_path="annotated_shelf_2.jpg"
    )
    print(result)
