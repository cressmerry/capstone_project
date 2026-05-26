import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

# Parameters
BATCH_SIZE = 4
EPOCHS = 5
IMG_SIZE = 300
NUM_CLASSES = 5  # person, laptop, chair, book, table
CLASS_MAP = {"person": 0, "laptop": 1, "chair": 2, "book": 3, "table": 4}

def load_data():
    """
    Loads images and bboxes from annotations.json, prepares them for training.
    """
    if not os.path.exists(ANNOTATIONS_FILE):
        raise FileNotFoundError(
            f"Annotations file not found at {ANNOTATIONS_FILE}. Run dataset.py first."
        )
        
    with open(ANNOTATIONS_FILE, "r") as f:
        annotations = json.load(f)
        
    x_train = []
    y_class = []
    y_bbox = []
    
    for filename, data in annotations.items():
        img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            continue
            
        # Load and resize image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        x_train.append(img / 255.0)  # Normalize
        
        # Take the first detection for simplification in this educational script
        # In a full SSD, we match detections with anchor boxes.
        # For this demonstration, we focus on single-object regression & classification.
        if len(data["detections"]) > 0:
            det = data["detections"][0]
            cls_name = det["class"]
            cls_id = CLASS_MAP.get(cls_name, 0)
            
            # Box format: [ymin, xmin, ymax, xmax]
            bbox = det["box"]
            
            y_class.append(cls_id)
            y_bbox.append(bbox)
        else:
            y_class.append(0)  # Background/Person
            y_bbox.append([0.0, 0.0, 0.0, 0.0])
            
    return np.array(x_train, dtype=np.float32), np.array(y_class, dtype=np.int32), np.array(y_bbox, dtype=np.float32)

def build_transfer_ssd_model():
    """
    Loads MobileNetV2, freezes base layers, and appends classification 
    and regression output heads representing the SSD output layer.
    """
    # Load backbone MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    
    # Freeze the base backbone layers
    base_model.trainable = False
    
    # Feature extractor output
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    
    # Classification Head (Output logits for classes)
    class_output = layers.Dense(NUM_CLASSES, activation="softmax", name="class_head")(x)
    
    # Bounding Box Regression Head (Outputs 4 coordinates: [ymin, xmin, ymax, xmax])
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_head")(x)
    
    # Build Multi-Output Model
    model = Model(inputs=base_model.input, outputs=[class_output, bbox_output])
    
    return model

def custom_multibox_loss(y_true_class, y_pred_class, y_true_bbox, y_pred_bbox):
    """
    Educational loss combining categorical cross-entropy and smooth L1 loss.
    """
    # Classification loss
    class_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_class, y_pred_class)
    class_loss = tf.reduce_mean(class_loss)
    
    # Bounding box regression loss (Smooth L1 Loss)
    # We only compute box loss for objects (where true box != [0,0,0,0])
    mask = tf.cast(tf.reduce_sum(y_true_bbox, axis=-1) > 0.0, tf.float32)
    
    diff = tf.abs(y_true_bbox - y_pred_bbox)
    smooth_l1 = tf.where(diff < 1.0, 0.5 * tf.square(diff), diff - 0.5)
    bbox_loss = tf.reduce_sum(smooth_l1, axis=-1) * mask
    bbox_loss = tf.reduce_mean(bbox_loss)
    
    # Total combined loss
    total_loss = class_loss + 2.0 * bbox_loss
    return total_loss, class_loss, bbox_loss

def main():
    print("TensorFlow Version:", tf.__version__)
    
    # 1. Load custom training subset
    try:
        x_train, y_class, y_bbox = load_data()
        print(f"Loaded training data: Images shape={x_train.shape}, Labels={y_class.shape}, Bboxes={y_bbox.shape}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Creating mock dataset for demonstration...")
        x_train = np.random.rand(10, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        y_class = np.random.randint(0, NUM_CLASSES, size=(10,)).astype(np.int32)
        y_bbox = np.random.rand(10, 4).astype(np.float32)

    # 2. Build model
    model = build_transfer_ssd_model()
    model.summary()
    
    # 3. Define Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # 4. Custom Training Loop
    print("\nStarting Training of SSD Output Head on Open Images subset...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_class_loss = 0.0
        epoch_bbox_loss = 0.0
        
        # Simple batching
        num_batches = int(np.ceil(len(x_train) / BATCH_SIZE))
        for step in range(num_batches):
            start_idx = step * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(x_train))
            
            x_batch = x_train[start_idx:end_idx]
            y_class_batch = y_class[start_idx:end_idx]
            y_bbox_batch = y_bbox[start_idx:end_idx]
            
            with tf.GradientTape() as tape:
                pred_class, pred_bbox = model(x_batch, training=True)
                total_loss, c_loss, b_loss = custom_multibox_loss(
                    y_class_batch, pred_class, y_bbox_batch, pred_bbox
                )
                
            # Compute gradients and apply weights updates
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += total_loss.numpy()
            epoch_class_loss += c_loss.numpy()
            epoch_bbox_loss += b_loss.numpy()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Total Loss: {epoch_loss/num_batches:.4f} "
              f"(Class Loss: {epoch_class_loss/num_batches:.4f}, Bbox Loss: {epoch_bbox_loss/num_batches:.4f})")
        
    print("\nTraining completed successfully! Output head weights are updated.")
    
    # Save training weights
    weights_path = os.path.join(DATASET_DIR, "ssd_fine_tuned_head.weights.h5")
    model.save_weights(weights_path)
    print(f"Fine-tuned head weights saved to {weights_path}")

if __name__ == "__main__":
    main()
