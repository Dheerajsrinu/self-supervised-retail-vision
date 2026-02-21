import os
import uuid
from app.inference.shelf_object_prediction_mdl_inf import ShelfObjectInference
from app.inference.prd_as_obj_det_inf import ProductAsObjectInference
from PIL import Image, ImageDraw, ImageFont

class EmptyShelfPercentageDetails():
    def __init__(self):
        self.shelf_object_prediction_mdl_inf = ShelfObjectInference()
        self.product_object_mdl_inf = ProductAsObjectInference()


    def intersection_area(shelf, boxA, boxB):
        """Compute intersection area between 2 bounding boxes"""
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])
        if x2 < x1 or y2 < y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def run_inference(
            self,
            image_path: str,
            save_path: str = "empty_percentage.jpg",
            conf: float = 0.25
        ):

        # Load image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        print("loaded image")

        # Inference
        shelf_results = self.shelf_object_prediction_mdl_inf.run_inference(image_path=image_path)
        product_results = self.product_object_mdl_inf.run_inference(image_path=image_path)

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

            used_area = sum(self.intersection_area(shelf_box, p) for p in products)

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
        out_dir = "results/empty_percentage/"
        os.makedirs(out_dir, exist_ok=True)
        save_path = out_dir+str(uuid.uuid4())+save_path
        img.save(save_path)
        print(f"Saved annotated image to {save_path}")

        return output
    def execute(self, event):
        image_path = event["file_path"]
        execute_response = self.run_inference(image_path=image_path)
        return execute_response

