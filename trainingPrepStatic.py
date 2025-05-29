import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Step 1: Load and Prepare Data
DATA_PATH = 'C:/Users/Edson Longares/OneDrive/Documents/EmbedProject/LSTM_training/data'  
labels = sorted(os.listdir(DATA_PATH))
label_map = {label: idx for idx, label in enumerate(labels)}

data, data_labels = [], []

for label in labels:
    label_dir = os.path.join(DATA_PATH, label)
    for file_name in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file_name)
        sample = np.load(file_path)  # Load a single frame or feature vector
        data.append(sample)
        data_labels.append(label_map[label])

X = np.array(data)

# Normalize input data
X = X / np.max(X)  # Scale data to range [0, 1]

# Convert labels to one-hot encoding
y = to_categorical(data_labels, num_classes=len(labels)).astype(int)

# Verify shapes
print(f"X shape: {X.shape}")  # Should be (num_samples, num_features)
print(f"y shape: {y.shape}")  # Should be (num_samples, num_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build and Train the Dense Model
model = Sequential([
    Input(shape=(X.shape[1],)),  # Input shape is now (num_features)
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Step 3: Visualize Training and Validation Metrics
plt.figure(figsize=(12, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Step 4: Save the Trained Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
model_save_path = os.path.join(BASE_DIR, 'asl_dense_model_optimizedv3.keras')  # Save in the same directory
model.save(model_save_path)

print(f"Model saved at: {model_save_path}")