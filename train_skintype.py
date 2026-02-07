import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# =========================
# CONFIG
# =========================
DATASET = r"C:\Users\sachi\finalyearproject1\Dataset\SkinType"
IMAGE_SIZE = (160, 160)     # 🔥 Reduced size (FASTER)
BATCH_SIZE = 16
EPOCHS = 15                 # 🔥 Reduced epochs

# =========================
# DATA GENERATORS
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_ds = datagen.flow_from_directory(
    DATASET,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_ds = datagen.flow_from_directory(
    DATASET,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# =========================
# MOBILENETV2 MODEL
# =========================
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # 🔥 Freeze base (FAST)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(train_ds.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
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

# =========================
# TRAIN (ONCE)
# =========================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)
model.save("models/skintype_model.h5")

print("✅ FAST MobileNet Skin Type model saved successfully!")
