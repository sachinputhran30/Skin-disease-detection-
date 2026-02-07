import pickle
import matplotlib.pyplot as plt

with open("history.pkl", "rb") as f:
    history = pickle.load(f)

epochs = range(1, len(history['accuracy']) + 1)

plt.figure(figsize=(8,5))

plt.scatter(
    epochs, history['accuracy'],
    label='Training Accuracy',
    marker='o',
    s=70
)

plt.scatter(
    epochs, history['val_accuracy'],
    label='Validation Accuracy',
    marker='^',
    s=70
)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Scatter Plot of Training vs Validation Accuracy')
plt.ylim(0.75, 0.95)   # zoom for clarity
plt.legend()
plt.grid(alpha=0.3)
plt.show()


1

import pickle
import matplotlib.pyplot as plt

with open("history.pkl", "rb") as f:
    history = pickle.load(f)

epochs = range(1, len(history['accuracy']) + 1)

plt.plot(epochs, history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(epochs, history['val_accuracy'], marker='o', label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#2

plt.plot(epochs, history['loss'], marker='o', label='Training Loss')
plt.plot(epochs, history['val_loss'], marker='o', label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


3
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

MODEL_PATH = "models/disease_model.h5"
TEST_DIR = r"C:\Users\sachi\finalyearproject1\Dataset\Disease"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

model = load_model(MODEL_PATH)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET = r"C:\Users\sachi\finalyearproject1\Dataset\Disease"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2   # 🔥 MUST MATCH TRAINING
)

test_ds = datagen.flow_from_directory(
    DATASET,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',   # 🔥 THIS IS THE KEY FIX
    shuffle=False
)

preds = model.predict(test_ds)
y_pred = np.argmax(preds, axis=1)
y_true = test_ds.classes
class_names = list(test_ds.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()





