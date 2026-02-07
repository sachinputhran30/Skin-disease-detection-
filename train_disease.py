import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import pickle

# ==========================
# CONFIG
# ==========================
DATASET = r"C:\Users\sachi\finalyearproject1\Dataset\Disease"
IMAGE_SIZE = (160, 160)   # 🔥 Reduced size (FASTER)
BATCH_SIZE = 16
EPOCHS = 15               # 🔥 Limited epochs

# ==========================
# DATA GENERATOR
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

train_ds = train_datagen.flow_from_directory(
    DATASET,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_ds = train_datagen.flow_from_directory(
    DATASET,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_ds.num_classes
print("Classes:", train_ds.class_indices)

# ==========================
# MOBILENETV2 MODEL
# ==========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160,160,3)
)

base_model.trainable = False  # 🔥 Freeze base layers (FAST)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ==========================
# COMPILE
# ==========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# CALLBACKS
# ==========================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.3
    )
]

# ==========================
# TRAIN (ONLY ONCE)
# ==========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================
# SAVE HISTORY (FOR GRAPHS)
# ==========================
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# ==========================
# SAVE MODEL
# ==========================
os.makedirs("models", exist_ok=True)
model.save("models/disease_model.h5")

print("✅ FAST disease model trained successfully!")
