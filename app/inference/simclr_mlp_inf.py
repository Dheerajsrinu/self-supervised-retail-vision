import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2 as T
from PIL import Image
import numpy as np
from collections import Counter

class MLP4(nn.Module):
    def __init__(self, input_dim=2048, num_classes=17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ProductRecognitionInference():

    def load_encoder(self,ckpt_path, device):
        encoder = models.resnet50(weights=None)
        encoder.fc = nn.Identity()  # output is 2048-D
        ckpt = torch.load(ckpt_path, map_location=device)

        # load encoder weights
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        else:
            raise ValueError("Checkpoint missing encoder weights!")

        encoder.to(device)
        encoder.eval()
        return encoder

    def load_mlp4(self,mlp_path, num_classes, device):
        mlp = MLP4(input_dim=2048, num_classes=num_classes)
        state = torch.load(mlp_path, map_location=device)
        mlp.load_state_dict(state)
        mlp.to(device)
        mlp.eval()
        return mlp

    def predict(self,image_path, encoder, mlp, class_names, device):

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            h = encoder(img)
            logits = mlp(h)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = probs.topk(5)

        results = []
        for p, idx in zip(top_prob[0], top_idx[0]):
            cls = class_names[idx]
            prob = p.item() * 100
            # print(f"{cls} → {prob:.2f}%")
            results.append((cls, prob))

        top1_label = class_names[top_idx[0][0].item()]
        top1_prob = top_prob[0][0].item() * 100


        classifier_obj_response={
            image_path: {
                "class_label": top1_label,
                "confidence": top1_prob
            }
        }
        return {
            "top1": top1_label,
            "top5": results,
            "classifier_obj_response": classifier_obj_response
        }

    def normalize(self, conf):
        return conf * 100 if conf <= 1 else conf

    def array_to_map(self, arr):
        result = {}
        for item in arr:
            image_path = next(iter(item))
            data = item[image_path]
            result[image_path] = {
                "class_label": data["class_label"],
                "confidence": self.normalize(data["confidence"])
            }
        return result

    def compare_results(self, arr1, arr2):

        map1 = self.array_to_map(arr1)
        map2 = self.array_to_map(arr2)

        output = []

        for image_path in map1:
            if image_path in map2:
                a = map1[image_path]
                b = map2[image_path]

                # NEW RULE:
                # If arr1 confidence > 25, always choose arr1
                if a["confidence"] > 25:
                    winner = a
                else:
                    # Otherwise, choose the one with highest confidence
                    winner = a if a["confidence"] >= b["confidence"] else b

                output.append({
                    "image_path": image_path,
                    "class_label": winner["class_label"],
                    "confidence": winner["confidence"]
                })

        return output


    def run_inference(
        self,
        product_list: list[str],
        cropped_images_object,
        save_image_path: str = "product_recognition_image.jpg",
        conf: float = 0.25
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_names = [
            "alcohol", "candy", "canned food", "chocolate", "dessert",
            "dried food", "dried fruit", "drink", "gum", "instant drink",
            "instant noodles", "milk", "personal hygiene",
            "puffed food", "seasoner", "stationery", "tissue"
        ]

        # Load models
        encoder = self.load_encoder("models/simclr_checkpoints_finetuned/simclr_epoch_100.pth", device)
        mlp = self.load_mlp4("models/mlp_4_eval_out/mlp4_classifier.pt", num_classes=len(class_names), device=device)
        data=[]
        classifier_object=[]
        for test_image in product_list:
            response = self.predict(test_image, encoder, mlp, class_names, device)
            data.append(response["top1"])
            classifier_object.append(response["classifier_obj_response"])
        result = self.compare_results(cropped_images_object, classifier_object)
        print(result)
        updated_data = [i["class_label"] for i in result]
        normalized_data = [item.capitalize() for item in updated_data]
        counted_data = Counter(normalized_data)
        return dict(counted_data)

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("[INFO] Using device:", device)

#     class_names = [
#         "alcohol", "candy", "canned food", "chocolate", "dessert",
#         "dried food", "dried fruit", "drink", "gum", "instant drink",
#         "instant noodles", "milk", "personal hygiene",
#         "puffed food", "seasoner", "stationery", "tissue"
#     ]

#     # Load models
#     encoder = load_encoder("models/simclr_checkpoints_finetuned/simclr_epoch_100.pth", device)
#     mlp = load_mlp4("models/mlp_4_eval_out/mlp4_classifier.pt", num_classes=len(class_names), device=device)

#     # Test image
#     # test_image = "results/products/crops_fine_grained_with_classes/yolov11_4dh/d39c8b6a-7362-42a1-85cd-789d36912993/191eb93e-fd3c-4e76-8790-5a55114c4b03_20180824-14-11-13-16_crop_0.jpg"
#     # test_image = "C:/Users/satyasrp/personal/projects/aiml/image/crops/single_image/20180913-16-00-48-636_crop_4.jpg"
#     # test_image = "C:/Users/satyasrp/personal/projects/aiml/image/crops_fine_grained_with_classes/yolov11_4dh/dessert/20180925-13-53-17-1610_jpg_crop_0.jpg"
#     # predict(test_image, encoder, mlp, class_names, device)
#     lst = [
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_0.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_1.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_2.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_3.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_4.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_5.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_6.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_7.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_8.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_9.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_10.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_11.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_12.jpg",
#       "results/products/crops_fine_grained_with_classes/yolov11_4dh/886bd633-34e0-4777-a93a-a22b125c0e79/cee9580b-c6ee-4f55-a4f0-450567f14f7b_20180910-10-43-35-783_jpg_crop_13.jpg"
#     ]

#     for test_image in lst:
#         response = predict(test_image, encoder, mlp, class_names, device)
#         print(response["top1"])
