from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import uuid
from app import model_store
import os

class ProductAsObjectInference():

    def run_inference(  
        self,
        image_path: str,
        conf: float = 0.25,
        save_image_path: str = "product_as_object_image.jpg",
    ):

        # Inference
        product_results = model_store.product_object_model.predict(image_path, conf=conf)[0]
        if save_image_path:
            img_with_boxes = product_results.plot()  # returns a numpy BGR image
            from cv2 import imwrite
            out_dir = "results/product_objects/"
            os.makedirs(out_dir, exist_ok=True)
            image_path = out_dir+str(uuid.uuid4())+save_image_path
            imwrite(image_path, img_with_boxes)
            print("Saved image:", save_image_path)

        return product_results
