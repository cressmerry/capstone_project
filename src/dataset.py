import os
import urllib.request
import pandas as pd
import json

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

# Open Images v6/v7 Target Class Machine IDs (MIDs)
TARGET_CLASSES = {
    "/m/01g317": "person",
    "/m/03b443": "laptop",
    "/m/01mzpv": "chair",
    "/m/01dx8v": "book",
    "/m/01y9k5": "table"
}

# GCS URLs for Open Images validation split
VAL_BOX_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
VAL_METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {url} to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print(f"File {filepath} already exists.")

def setup_subset(limit_images=50):
    """
    Downloads validation annotations, filters for target classes, 
    and downloads a subset of images from the GCS bucket.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # 1. Download Open Images annotation and image metadata CSVs
    val_box_csv = os.path.join(DATASET_DIR, "validation-annotations-bbox.csv")
    val_meta_csv = os.path.join(DATASET_DIR, "validation-images-with-rotation.csv")
    
    download_file(VAL_BOX_URL, val_box_csv)
    download_file(VAL_METADATA_URL, val_meta_csv)
    
    # 2. Read annotations
    print("Parsing annotations...")
    df_box = pd.read_csv(val_box_csv)
    df_meta = pd.read_csv(val_meta_csv)
    
    # Filter for target classes
    df_filtered = df_box[df_box["LabelName"].isin(TARGET_CLASSES.keys())].copy()
    df_filtered["ClassName"] = df_filtered["LabelName"].map(TARGET_CLASSES)
    
    # Find unique images that contain target classes
    unique_image_ids = df_filtered["ImageID"].unique()
    print(f"Found {len(unique_image_ids)} validation images containing target classes.")
    
    subset_ids = unique_image_ids[:limit_images]
    df_meta_subset = df_meta[df_meta["ImageID"].isin(subset_ids)]
    
    annotations_data = {}
    
    print(f"Downloading {limit_images} images and parsing annotations...")
    for idx, row in df_meta_subset.iterrows():
        img_id = row["ImageID"]
        img_url = row["OriginalURL"]
        
        # Download image
        img_filename = f"{img_id}.jpg"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        
        try:
            download_file(img_url, img_path)
            
            # Find annotations for this image
            img_anns = df_filtered[df_filtered["ImageID"] == img_id]
            
            bboxes = []
            for _, ann in img_anns.iterrows():
                # Open Images bbox coordinates are normalized [0, 1]
                # Format: [ymin, xmin, ymax, xmax]
                bboxes.append({
                    "class": ann["ClassName"],
                    "mid": ann["LabelName"],
                    "box": [ann["YMin"], ann["XMin"], ann["YMax"], ann["XMax"]]
                })
                
            from PIL import Image
            with Image.open(img_path) as pil_img:
                width_val, height_val = pil_img.size
                
            annotations_data[img_filename] = {
                "url": img_url,
                "width": width_val,
                "height": height_val,
                "detections": bboxes
            }
        except Exception as e:
            print(f"Skipping image {img_id} due to download error: {e}")
            
    # Save annotations file
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_data, f, indent=4)
        
    print(f"Setup complete! Downloaded {len(annotations_data)} images.")
    print(f"Annotations saved to {ANNOTATIONS_FILE}")

if __name__ == "__main__":
    setup_subset(limit_images=10) # Download a very small subset of 10 images for testing
