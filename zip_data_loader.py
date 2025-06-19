import os
import zipfile
from PIL import Image
import numpy as np
from io import BytesIO

def load_images_as_list_of_dicts(zip_folder_path):
    """
    Return a list where each item is a dict for each ZIP:
    {
      "zip_name": <zip filename>,
      "images": [np.array, np.array, ...]  # list of numpy images in that zip
    }
    """
    result = []

    for zip_name in os.listdir(zip_folder_path):
        if zip_name.endswith(".zip"):
            zip_path = os.path.join(zip_folder_path, zip_name)
            images_in_zip = []
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.lower().endswith(".jpg"):
                        with zip_ref.open(file_name) as file:
                            image = Image.open(BytesIO(file.read())).convert("RGB")
                            image_np = np.array(image)
                            images_in_zip.append(image_np)
            result.append({
                "zip_name": zip_name,
                "images": images_in_zip
            })

    return result

def test_format():
    zip_folder = "pictures"
    zip_images_list = load_images_as_list_of_dicts(zip_folder)

    for zip_dict in zip_images_list:
        print(f"ZIP: {zip_dict['zip_name']} has {len(zip_dict['images'])} images")
        for i, img_np in enumerate(zip_dict['images']):
            print(f"  Image {i+1} shape: {img_np.shape}")

test_format()