
import csv
from ultralytics import YOLO
import cv2, os, glob
from app import model_store
from collections import Counter
import uuid

class CroppedProductInference():

    def process_image(self, img_path: str, out_dir: str, results):
        print(f"Processing image: {img_path}")
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        image_name = os.path.basename(img_path)
        boxes = result.boxes
        img = cv2.imread(img_path)
        cropped_images_list = []
        cropped_images_object = []
        try:
            for i, box in enumerate(boxes.xyxy):
                class_id = int(boxes.cls[i].item())
                class_label = result.names[class_id]
                confidence = float(boxes.conf[i].item())
                class_folder = os.path.join(out_dir, class_label.lower())
                # os.makedirs(class_folder, exist_ok=True)
                os.makedirs(out_dir, exist_ok=True)
                # Save the crop in the respective class folder
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                crop_filename = f"{image_name.split('.')[0]}_crop_{i}.jpg"
                # cv2.imwrite(os.path.join(out_dir, crop_filename), crop)
                if crop is None or crop.size == 0:
                    print("Empty crop, skipping")
                    continue
                save_path = os.path.normpath(os.path.join(out_dir, crop_filename))
                success = cv2.imwrite(save_path, crop)
                print("success -> ",success)
                if not success:
                    print("Failed to save:", save_path)
                else:
                    print("Saved:", save_path)
                
                print(f"image_name: {image_name}, class_label: {class_label}, confidence: {confidence:.4f}, saved to folder: {class_folder}")

                updated_image_name = f"{out_dir}/{crop_filename}"
                cropped_images_object.append({
                    updated_image_name: {
                        "image_name": image_name,
                        "class_label":class_label,
                        "confidence": confidence
                    }
                })

                print("updated_image_name -> ",updated_image_name)
                cropped_images_list.append(updated_image_name)
        except Exception as e:
            print("exception is -> ",e)

        return cropped_images_list, cropped_images_object

    def run_inference(  
        self,
        request_id: str,
        image_paths_list: list[str],
        conf: float = 0.25,
        save_image_path: str = "anotated.jpg",
    ):
        # Check if model is loaded
        if model_store.product_rec_model is None:
            raise RuntimeError(
                "Product recognition model is not loaded. "
                "Please ensure the model file exists at 'models/rpc_yolov11_4dh3/weights/best.pt' "
                "and restart the application."
            )

        # Folder to store crops
        out_dir = 'results/products/crops_fine_grained_with_classes/yolov11_4dh/'+request_id
        os.makedirs(out_dir, exist_ok=True)
        print("len of list_of_checkout_images is -> ", len(image_paths_list))
        response=[]
        cnt = 0
        for img_path in image_paths_list:
            cnt += 1
            results = model_store.product_rec_model(img_path, conf=0.25)
            boxes = results[0].boxes

            cls_ids = boxes.cls.cpu().numpy().astype(int)
            class_counts = Counter(cls_ids)

            print("\nImage:", os.path.basename(img_path))
            name_counts = {model_store.product_rec_model.names[c]: n for c, n in class_counts.items()}
            print("Class counts predicted by YOLO (names):", name_counts)
            cropped_images_path_list, cropped_images_object = self.process_image(img_path=img_path, out_dir=out_dir, results=results)

            annotated_img = results[0].plot() 
            annotated_folder = os.path.join(out_dir, "anotated_images")
            os.makedirs(annotated_folder, exist_ok=True)
            image_name=str(uuid.uuid4())+save_image_path
            annot_path = os.path.join(annotated_folder, image_name)
            print("annot_path -> ",annot_path)
            cv2.imwrite(annot_path, annotated_img)
            print(f"image_path: {img_path}, annotacted image saved to folder: {annot_path}")
            response.append({
                "image": img_path,
                # "yolo_v11_4dh_class_labels": name_counts,
                "anotate_image_path": annot_path,
                "cropped_images_path_list": cropped_images_path_list,
                "cropped_images_object": cropped_images_object
            })
        print("Count of total images processed -> ", cnt)
        return response
