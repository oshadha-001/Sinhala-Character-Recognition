# Part 1: Data Preparation
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
from imblearn.over_sampling import SMOTE  # For class balancing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data_path = r'D:\Dataset\train'
categories = os.listdir(data_path)
print(f"Categories: {len(categories)}")

data = []
target = []

# Load and preprocess images
for category in categories:
    imgs_path = os.path.join(data_path, category)
    img_names = os.listdir(imgs_path)
    print(f"Processing {imgs_path}: {len(img_names)} images")

    for img_name in img_names:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        img = cv2.resize(img, (16, 16))  # Increase to 16x16
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Binarize
        img = 255 - img  # Invert (white characters on black background)
        data.append(img)
        target.append(category)

print(f"Total images: {len(data)}, Targets: {len(target)}")

# Display an example image
plt.imshow(data[10], cmap='gray')
plt.title(f"Sample: {target[10]}")
plt.show()

# Convert to NumPy arrays
data = np.array(data)
print('Before reshape:', data.shape)

# Encode target labels
le = LabelEncoder()
target = le.fit_transform(target)
print('Encoded target:', target[:10])

# Reshape and normalize data
data = data.reshape(data.shape[0], -1)  # Flatten to (N, 16*16)
data = data / 255.0  # Normalize to [0, 1]
print('After reshape:', data.shape)

# Check class distribution
unique, counts = np.unique(target, return_counts=True)
print("Class distribution:", dict(zip(le.classes_, counts)))

# Balance classes using SMOTE (optional, if imbalance detected)
smote = SMOTE(random_state=42)
data, target = smote.fit_resample(data, target)
print('After SMOTE:', data.shape, target.shape)

# Save data and target
np.save('data', data)
np.save('target', target)
joblib.dump(le, 'label_encoder.sav')

# Split data
train_data, test_data, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42, stratify=target
)
print(f"Train: {train_data.shape}, {train_target.shape}")
print(f"Test: {test_data.shape}, {test_target.shape}")

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_data, train_target)

# Predict and evaluate
predicted_target = model.predict(test_data)
acc = accuracy_score(test_target, predicted_target)
print('Accuracy:', acc)
classi_report = classification_report(test_target, predicted_target, target_names=le.classes_)
print('Classification Report:\n', classi_report)

# Save model
joblib.dump(model, 'sinhala-character-rf.sav')