from app.inference.cropped_products_inf import CroppedProductInference
from app.inference.simclr_mlp_inf import ProductRecognitionInference
class ProductDetails():
    
    def run_crop_inference(
            self,
            image_paths_list: list[str],
            request_id: str
        ):
        cpi_obj = CroppedProductInference()
        inference_result = cpi_obj.run_inference(request_id=request_id, image_paths_list=image_paths_list)
        return inference_result

    def run_prodct_inference(self, product_list, cropped_images_object):
        pri_obj = ProductRecognitionInference()
        inference_result = pri_obj.run_inference(product_list=product_list, cropped_images_object= cropped_images_object)
        return inference_result
    
    def execute(self, event):
        print("event -> ", event)
        image_paths_list = event["file_paths_list"]
        request_id = event["request_id"]
        print("image_paths_list -> ",image_paths_list)
        crop_inference_response = self.run_crop_inference(image_paths_list, request_id)
        # print("crop_inference_response-> ",crop_inference_response)
        product_list=[]
        cropped_images_object=[]
        # product_list = crop_inference_response[0]["cropped_images_path_list"]
        # cropped_images_object = crop_inference_response[0]["cropped_images_object"]
        for item in crop_inference_response:
            product_list.extend(item.get("cropped_images_path_list", []))
            cropped_images_object.extend(item.get("cropped_images_object", []))
        product_inference_response = self.run_prodct_inference(product_list, cropped_images_object)

        response = {
            "products_count": product_inference_response
            # "product_details": crop_inference_response
        }

        return response
    