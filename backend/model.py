import os
import json
import cv2
import numpy as np

class SSDObjectDetector:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(__file__), "models", "ssd_mobilenet_v2_coco_2018_03_29")
        self.pb_path = os.path.join(self.model_dir, "frozen_inference_graph.pb")
        self.pbtxt_path = os.path.join(self.model_dir, "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
        self.labels_path = os.path.join(self.model_dir, "coco_labels.json")
        
        # Load Labels
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                # Keys in JSON are strings, convert to int for mapping
                self.labels = {int(k): v for k, v in json.load(f).items()}
        else:
            raise FileNotFoundError(f"Labels file not found at {self.labels_path}. Run download_models.py first.")
            
        # Load Network
        if os.path.exists(self.pb_path) and os.path.exists(self.pbtxt_path):
            self.net = cv2.dnn.readNetFromTensorflow(self.pb_path, self.pbtxt_path)
            # Try to use CUDA if available, fallback to CPU
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA backend for OpenCV DNN.")
            except Exception:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Falling back to CPU backend for OpenCV DNN.")
        else:
            raise FileNotFoundError(
                f"Model files not found at {self.model_dir}. Run download_models.py first."
            )

    def detect(self, image_np, conf_threshold=0.5, nms_threshold=0.4, allowed_classes=None):
        """
        Run object detection on a numpy image (BGR format).
        
        Parameters:
        - image_np: numpy array in BGR format.
        - conf_threshold: minimum confidence score to retain a detection.
        - nms_threshold: IOU threshold for Non-Maximum Suppression.
        - allowed_classes: set of class labels (strings) to filter by. If None, returns all.
        
        Returns:
        - List of dicts, each representing a detection:
          {
            "class_id": int,
            "label": str,
            "confidence": float,
            "bbox": [x, y, width, height]  # pixels
          }
        """
        h, w = image_np.shape[:2]
        
        # SSD MobileNet V2 input size is 300x300
        # swapRB=True converts BGR to RGB
        blob = cv2.dnn.blobFromImage(image_np, size=(300, 300), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Shape of detections is (1, 1, N, 7)
        num_detections = detections.shape[2]
        
        for i in range(num_detections):
            confidence = float(detections[0, 0, i, 2])
            if confidence >= conf_threshold:
                class_id = int(detections[0, 0, i, 1])
                label = self.labels.get(class_id, "unknown")
                
                # Check class filter
                if allowed_classes is not None and label not in allowed_classes:
                    continue
                
                # Bounding box coordinates are normalized to [0, 1]
                x_min = int(detections[0, 0, i, 3] * w)
                y_min = int(detections[0, 0, i, 4] * h)
                x_max = int(detections[0, 0, i, 5] * w)
                y_max = int(detections[0, 0, i, 6] * h)
                
                # Clamp coordinates to image boundaries
                x_min = max(0, min(x_min, w - 1))
                y_min = max(0, min(y_min, h - 1))
                x_max = max(0, min(x_max, w - 1))
                y_max = max(0, min(y_max, h - 1))
                
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                if box_w > 0 and box_h > 0:
                    boxes.append([x_min, y_min, box_w, box_h])
                    confidences.append(confidence)
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        results = []
        # Support different return types of NMSBoxes depending on OpenCV version (list or flat numpy array)
        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else [idx[0] if isinstance(idx, (list, np.ndarray)) else idx for idx in indices]
            for idx in indices:
                class_id = class_ids[idx]
                results.append({
                    "class_id": class_id,
                    "label": self.labels.get(class_id, "unknown"),
                    "confidence": round(confidences[idx], 4),
                    "bbox": boxes[idx]  # [x, y, w, h]
                })
                
        return results
