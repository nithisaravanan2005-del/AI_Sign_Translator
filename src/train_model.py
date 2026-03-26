import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

# Load dataset
data = pd.read_csv("data/dataset.csv", header=None)

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save label mapping
import json
label_map = dict(zip(encoder.classes_, range(len(encoder.classes_))))
with open("models/label_map.json", "w") as f:
    json.dump(label_map, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(126,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("models/sign_model.h5")

print("Model trained and saved successfully.")