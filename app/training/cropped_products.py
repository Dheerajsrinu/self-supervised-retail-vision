# import csv
# from ultralytics import YOLO
# import cv2, os, glob

# # Load best YOLOv8-4DH model
# model = YOLO('C:/Users/satyasrp/personal/projects/aiml/runs/detect/rpc_yolov8_4dh3/weights/best.pt')

# # Folder to store crops
# out_dir = 'crops_fine_grained_with_classes/yolov11_4dh'
# os.makedirs(out_dir, exist_ok=True)

# # Load all checkout images (test or full dataset)
# list_of_checkout_images = glob.glob("C:/Users/satyasrp/personal/projects/aiml/image/Groceries-7/test/images/*.jpg")
# # list_of_checkout_images = glob.glob("C:/Users/satyasrp/personal/projects/aiml/image/retail_product_checkout/test2019/*.jpg")
# print("len of list_of_checkout_images is -> ", len(list_of_checkout_images))

# # Create and open the CSV file
# csv_file = open('fine_tuned_class_labels.csv', 'w', newline='', encoding='utf-8')
# csv_writer = csv.writer(csv_file)
# # Write header
# csv_writer.writerow(['image_file_name', 'label_name'])

# cnt = 0
# for img_path in list_of_checkout_images:
#     cnt += 1
#     results = model(img_path, conf=0.25)  # Run inference on image
#     print(f"Processing image: {img_path}")

#     # Check if results is a list or a single object
#     if isinstance(results, list):
#         result = results[0]  # Get the first result if it's a list
#     else:
#         result = results  # Single result, directly use it

#     # Extract the image name and class label(s)
#     image_name = os.path.basename(img_path)

#     # Get class labels from results
#     boxes = result.boxes
#     img = cv2.imread(img_path)

#     for i, box in enumerate(boxes.xyxy):
#         class_id = int(boxes.cls[i].item())  # Class index
#         class_label = result.names[class_id]  # Get class label from `names`

#         # Save the crop
#         x1, y1, x2, y2 = map(int, box)
#         crop = img[y1:y2, x1:x2]
#         cv2.imwrite(f"{out_dir}/{image_name.split('.')[0]}_crop_{i}.jpg", crop)
#         print("image_name, class_label are -> ",image_name, class_label)
#         updated_image_name = f"{image_name.split('.')[0]}_crop_{i}.jpg"
#         # Write image name and class label to the CSV
#         csv_writer.writerow([updated_image_name, class_label])

# print("Count of total images processed -> ", cnt)

# # Close the CSV file
# csv_file.close()
# print("CSV file with image names and class labels has been created.")


import csv
from ultralytics import YOLO
import cv2, os, glob

# Load best YOLOv8-4DH model
model = YOLO('C:/Users/satyasrp/personal/projects/aiml/runs/detect/rpc_yolov8_4dh3/weights/best.pt')

# Folder to store crops
out_dir = 'crops_fine_grained_with_classes/yolov11_4dh'
os.makedirs(out_dir, exist_ok=True)

# Load all checkout images (test or full dataset)
list_of_checkout_images = glob.glob("C:/Users/satyasrp/personal/projects/aiml/image/Groceries-7/test/images/*.jpg")
# list_of_checkout_images = glob.glob("C:/Users/satyasrp/personal/projects/aiml/image/retail_product_checkout/test2019/*.jpg")
print("len of list_of_checkout_images is -> ", len(list_of_checkout_images))

# Create and open the CSV file
csv_file = open('fine_tuned_class_labels_with_classes.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
# Write header
csv_writer.writerow(['image_file_name', 'label_name'])

cnt = 0
for img_path in list_of_checkout_images:
    cnt += 1
    results = model(img_path, conf=0.25)  # Run inference on image
    print(f"Processing image: {img_path}")

    # Check if results is a list or a single object
    if isinstance(results, list):
        result = results[0]  # Get the first result if it's a list
    else:
        result = results  # Single result, directly use it

    # Extract the image name and class label(s)
    image_name = os.path.basename(img_path)

    # Get class labels from results
    boxes = result.boxes
    img = cv2.imread(img_path)

    for i, box in enumerate(boxes.xyxy):
        class_id = int(boxes.cls[i].item())  # Class index
        class_label = result.names[class_id]  # Get class label from `names`

        # Create folder for the class label if it doesn't exist
        class_folder = os.path.join(out_dir, class_label.lower())  # Make class label folder name lowercase
        os.makedirs(class_folder, exist_ok=True)

        # Save the crop in the respective class folder
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        crop_filename = f"{image_name.split('.')[0]}_crop_{i}.jpg"
        cv2.imwrite(os.path.join(class_folder, crop_filename), crop)
        
        print(f"image_name: {image_name}, class_label: {class_label}, saved to folder: {class_folder}")

        updated_image_name = f"{class_label.lower()}/{crop_filename}"
        # Write image name and class label to the CSV
        csv_writer.writerow([updated_image_name, class_label])

print("Count of total images processed -> ", cnt)

# Close the CSV file
csv_file.close()
print("CSV file with image names and class labels has been created.")
