from app.inference.prd_as_obj_det_inf import ProductAsObjectInference

class FetchProductAsObjectDetails():
    
    def run_inference(
            self,
            image_path: str
        ):
        paoi_obj = ProductAsObjectInference()
        inference_result = paoi_obj.run_inference(image_path=image_path)
        return inference_result
    def execute(self, event):
        image_path = event["file_path"]
        execute_response = self.run_inference(image_path=image_path)
        inference_speed = execute_response.speed
        number_of_products = len(execute_response.boxes)
        response_object = {
            "number_of_products": number_of_products,
            "inference_speed": inference_speed
        }
        return response_object
