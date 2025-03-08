import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load CSV with Class Labels
csv_file = r".\dataset\Indian-Traffic Sign-Dataset\traffic_sign.csv"  
df = pd.read_csv(csv_file)
class_labels = {row["ClassId"]: row["Name"] for _, row in df.iterrows()}

# Dataset Path (Images are in separate folders named by ClassId)
DATASET_PATH = r".\dataset\Indian-Traffic Sign-Dataset\Images"
IMG_SIZE = (32, 32)
NUM_CLASSES = len(class_labels)

# Load Images and Labels
images, labels = [], []
for class_id, name in class_labels.items():
    class_path = os.path.join(DATASET_PATH, str(class_id))
    if not os.path.exists(class_path):
        continue  # Skip if folder does not exist
    
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip corrupted or unreadable images
        
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(class_id)

# Convert to NumPy Arrays and Normalize
images = np.array(images) / 255.0
labels = to_categorical(labels, NUM_CLASSES)

# Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax"),
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save(r".\model\traffic_sign_model.h5")
print("Model Training Completed!")