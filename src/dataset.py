import os
import urllib.request
import pandas as pd
import json
import concurrent.futures
from PIL import Image
from tqdm import tqdm

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

# Open Images v6/v7 Target Class Machine IDs (MIDs)
TARGET_CLASSES = {
    "/m/01g317": "person",
    "/m/03b443": "laptop",
    "/m/01mzpv": "chair",
    "/m/01dx8v": "book",
    "/m/01y9k5": "table",
    "/m/018xm": "cell phone",
    "/m/0cmf2": "cup",
    "/m/01m2v": "keyboard",
    "/m/020lf": "mouse"
}

# GCS URLs for Open Images validation split
VAL_BOX_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
VAL_METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"

def download_file(url, filepath, verbose=True):
    if not os.path.exists(filepath):
        if verbose:
            print(f"Downloading {url} to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        if verbose:
            print(f"File {filepath} already exists.")

def setup_subset(limit_images=None):
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
    print("Parsing annotations CSV...")
    df_box = pd.read_csv(val_box_csv)
    df_meta = pd.read_csv(val_meta_csv)
    
    # Filter for target classes
    df_filtered = df_box[df_box["LabelName"].isin(TARGET_CLASSES.keys())].copy()
    df_filtered["ClassName"] = df_filtered["LabelName"].map(TARGET_CLASSES)
    
    # Find unique images that contain target classes
    unique_image_ids = df_filtered["ImageID"].unique()
    print(f"Found {len(unique_image_ids)} validation images containing target classes.")
    
    if limit_images is not None:
        subset_ids = unique_image_ids[:limit_images]
    else:
        subset_ids = unique_image_ids
        
    df_meta_subset = df_meta[df_meta["ImageID"].isin(subset_ids)]
    total_images = len(df_meta_subset)
    
    # Pre-group annotations by ImageID to avoid slicing DataFrame inside threads
    print("Grouping annotations by ImageID...")
    image_annotations = {}
    grouped = df_filtered.groupby("ImageID")
    
    for img_id, group in tqdm(grouped, desc="Grouping annotations"):
        image_annotations[img_id] = []
        for _, ann in group.iterrows():
            image_annotations[img_id].append({
                "class": ann["ClassName"],
                "mid": ann["LabelName"],
                "box": [ann["YMin"], ann["XMin"], ann["YMax"], ann["XMax"]]
            })
            
    annotations_data = {}
    
    def download_and_parse_image(row):
        img_id = row["ImageID"]
        img_url = row["OriginalURL"]
        img_filename = f"{img_id}.jpg"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        
        result = None
        try:
            download_file(img_url, img_path, verbose=False)
            
            with Image.open(img_path) as pil_img:
                width_val, height_val = pil_img.size
                
            bboxes = image_annotations.get(img_id, [])
            result = (img_filename, {
                "url": img_url,
                "width": width_val,
                "height": height_val,
                "detections": bboxes
            })
        except Exception:
            pass
                
        return result

    print(f"Downloading up to {total_images} images and parsing annotations in parallel...")
    rows = [row for _, row in df_meta_subset.iterrows()]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_and_parse_image, row): row for row in rows}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading images"):
            res = future.result()
            if res is not None:
                img_filename, data = res
                annotations_data[img_filename] = data
                
    # Save annotations file
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_data, f, indent=4)
        
    print(f"Setup complete! Successfully downloaded and annotated {len(annotations_data)} images.")
    print(f"Annotations saved to {ANNOTATIONS_FILE}")

if __name__ == "__main__":
    setup_subset()
