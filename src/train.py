import os
import json
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
NUM_CLASSES = 9  # person, laptop, chair, book, table, cell phone, cup, keyboard, mouse
CLASS_MAP = {
    "person": 0,
    "laptop": 1,
    "chair": 2,
    "book": 3,
    "table": 4,
    "cell phone": 5,
    "cup": 6,
    "keyboard": 7,
    "mouse": 8
}

def load_data():
    """
    Parses annotations.json and returns lists of image paths, class targets, and bounding box targets.
    """
    if not os.path.exists(ANNOTATIONS_FILE):
        raise FileNotFoundError(
            f"Annotations file not found at {ANNOTATIONS_FILE}. Run dataset.py first."
        )
        
    with open(ANNOTATIONS_FILE, "r") as f:
        annotations = json.load(f)
        
    image_paths = []
    y_class = []
    y_bbox = []
    
    for filename, data in annotations.items():
        img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            continue
            
        # Take the first detection for simplification in this educational script
        # In a full SSD, we match detections with anchor boxes.
        if len(data["detections"]) > 0:
            det = data["detections"][0]
            cls_name = det["class"]
            cls_id = CLASS_MAP.get(cls_name, 0)
            bbox = det["box"]
        else:
            cls_id = 0
            bbox = [0.0, 0.0, 0.0, 0.0]
            
        image_paths.append(img_path)
        y_class.append(cls_id)
        y_bbox.append(bbox)
        
    return image_paths, y_class, y_bbox

def preprocess_image_and_targets(image_path, label, bbox):
    """
    Reads image file, decodes, resizes, and normalizes it.
    This runs on CPU using tf.data AUTOTUNE to avoid bottlenecks.
    """
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0  # Normalize to [0, 1]
    return img, label, bbox

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
    
    # 1. Load custom training subset via tf.data.Dataset pipeline
    try:
        image_paths, y_class, y_bbox = load_data()
        print(f"Parsed annotations: Found {len(image_paths)} valid images.")
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, y_class, y_bbox))
        dataset = dataset.shuffle(buffer_size=max(len(image_paths), 100), reshuffle_each_iteration=True)
        dataset = dataset.map(preprocess_image_and_targets, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Creating mock tf.data.Dataset for demonstration...")
        mock_imgs = tf.random.uniform((10, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
        mock_cls = tf.random.uniform((10,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
        mock_box = tf.random.uniform((10, 4), dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((mock_imgs, mock_cls, mock_box))
        dataset = dataset.batch(BATCH_SIZE)

    # 2. Build model
    model = build_transfer_ssd_model()
    model.summary()
    
    # 3. Define Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # 4. Custom Training Loop with tf.data.Dataset iteration
    print("\nStarting Training of SSD Output Head on Open Images subset...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_class_loss = 0.0
        epoch_bbox_loss = 0.0
        step_count = 0
        
        for x_batch, y_class_batch, y_bbox_batch in dataset:
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
            step_count += 1
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Total Loss: {epoch_loss/step_count:.4f} "
              f"(Class Loss: {epoch_class_loss/step_count:.4f}, Bbox Loss: {epoch_bbox_loss/step_count:.4f})")
        
    print("\nTraining completed successfully! Output head weights are updated.")
    
    # Save training weights
    weights_path = os.path.join(DATASET_DIR, "ssd_fine_tuned_head.weights.h5")
    model.save_weights(weights_path)
    print(f"Fine-tuned head weights saved to {weights_path}")

if __name__ == "__main__":
    main()
