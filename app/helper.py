import uuid
import os
from typing import List
from fastapi import UploadFile

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_images(images: List[bytes], filenames: List[str]) -> List[str]:
    file_paths = []
    for img_bytes, name in zip(images, filenames):
        image_id = str(uuid.uuid4())
        path = f"{UPLOAD_DIR}/{image_id}_{name}"
        with open(path, "wb") as f:
            f.write(img_bytes)
        file_paths.append(path)
    return file_paths
