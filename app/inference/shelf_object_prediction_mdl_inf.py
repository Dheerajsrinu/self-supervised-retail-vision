import os
from ultralytics import YOLO
from PIL import Image
import uuid
from app import model_store

class ShelfObjectInference():
    def run_inference(
        self,
        image_path: str,
        save_image_path: str = "shelves_predicted_image.jpg",
        conf: float = 0.25
    ):
        """
        Runs YOLO inference, returns predictions JSON, and saves:
        - predictions.json
        - image with bounding boxes

        :param image_path: Path to input image
        :param save_image_path: If provided, saves output image with bounding boxes
        :param conf: Confidence threshold
        :return: dict with predictions
        """
        # Run prediction
        results = model_store.shelf_detector.predict(image_path, conf=conf, imgsz=832)[0]

        if save_image_path:
            img_with_boxes = results.plot()  # returns a numpy BGR image
            from cv2 import imwrite
            out_dir = "results/shelves/"
            os.makedirs(out_dir, exist_ok=True)
            image_path = out_dir+str(uuid.uuid4())+save_image_path
            imwrite(image_path, img_with_boxes)
            print("Saved image:", image_path)

        return results
