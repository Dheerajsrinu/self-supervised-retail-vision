from app.inference.shelf_object_prediction_mdl_inf import ShelfObjectInference

import uuid

class FetchShelfDetails():
    def __init__(self):
        self.soi_obj = ShelfObjectInference()

    def run_inference(
            self,
            image_path: str
        ):
        inference_result = self.soi_obj.run_inference(image_path=image_path)
        predictions = []
        for b in inference_result.boxes:
            cls_id = int(b.cls[0].item())
            confidence = float(b.conf[0].item())

            # xyxy pixel coords
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            xc = x1 + width / 2
            yc = y1 + height / 2

            predictions.append({
                "x": round(xc, 3),
                "y": round(yc, 3),
                "width": round(width, 3),
                "height": round(height, 3),
                "confidence": round(confidence, 3),
                "class": "shelves",
                "class_id": cls_id,
                "detection_id": str(uuid.uuid4())
            })

        return predictions

    def execute(self, event):
        image_path = event["file_path"]
        execute_response = self.run_inference(image_path=image_path)
        return execute_response
