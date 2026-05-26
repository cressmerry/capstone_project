import os
import tarfile
import urllib.request
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SSD_DIR = os.path.join(MODEL_DIR, "ssd_mobilenet_v2_coco_2018_03_29")
TAR_URL = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
TAR_FILE = os.path.join(MODEL_DIR, "ssd_mobilenet_v2_coco_2018_03_29.tar.gz")
PBTXT_URL = "https://raw.githubusercontent.com/Qengineering/MobileNet_SSD_OpenCV_TensorFlow/master/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
PBTXT_FILE = os.path.join(SSD_DIR, "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Download and extract model weights
    pb_file_path = os.path.join(SSD_DIR, "frozen_inference_graph.pb")
    if not os.path.exists(pb_file_path):
        print(f"Downloading SSD MobileNet V2 weights from {TAR_URL}...")
        try:
            urllib.request.urlretrieve(TAR_URL, TAR_FILE)
            print("Download completed. Extracting files...")
            with tarfile.open(TAR_FILE, "r:gz") as tar:
                # Only extract frozen_inference_graph.pb
                for member in tar.getmembers():
                    if "frozen_inference_graph.pb" in member.name:
                        member.name = os.path.basename(member.name)
                        tar.extract(member, path=SSD_DIR)
                        print(f"Extracted {member.name} to {SSD_DIR}")
            if os.path.exists(TAR_FILE):
                os.remove(TAR_FILE)
        except Exception as e:
            print(f"Error downloading/extracting weights: {e}")
            return
    else:
        print("SSD MobileNet V2 weights already exist.")

    # 2. Download OpenCV pbtxt config
    if not os.path.exists(PBTXT_FILE):
        print(f"Downloading pbtxt configuration from {PBTXT_URL}...")
        try:
            urllib.request.urlretrieve(PBTXT_URL, PBTXT_FILE)
            print(f"Config saved to {PBTXT_FILE}")
        except Exception as e:
            print(f"Error downloading config: {e}")
            return
    else:
        print("Config pbtxt already exists.")

    # 3. Save labels file
    labels_file = os.path.join(SSD_DIR, "coco_labels.json")
    with open(labels_file, "w") as f:
        json.dump(COCO_LABELS, f, indent=4)
    print(f"Labels saved to {labels_file}")
    print("Model download completed successfully!")

if __name__ == "__main__":
    main()
