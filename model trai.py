# --------------------------------------------------
# MobileNetV2 – Insect Species Classification (CPU-Friendly)
# --------------------------------------------------

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from PIL import ImageFile
import warnings

# --------------------------------------------------
# Safety
# --------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# --------------------------------------------------
# Paths & Hyperparameters
# --------------------------------------------------
TRAIN_DIR = r"e:/Isect pest/train"
VAL_DIR   = r"e:/Isect pest/val"

IMG_SIZE = 190
BATCH_SIZE = 16  # smaller batch for CPU
EPOCHS_HEAD = 25
EPOCHS_FINE = 10
LEARNING_RATE_HEAD = 1e-4
LEARNING_RATE_FINE = 5e-5

# --------------------------------------------------
# Data Generators
# --------------------------------------------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
print("🧬 Number of classes:", NUM_CLASSES)

# --------------------------------------------------
# Class Weights
# --------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# --------------------------------------------------
# Base Model
# --------------------------------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze backbone

# --------------------------------------------------
# Classification Head
# --------------------------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Dense Block 1
x = Dense(1024, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Dense Block 2
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# --------------------------------------------------
# Compile (Head Training)
# --------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("mobilenetv2_insect_best.keras", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

# --------------------------------------------------
# Train Head Only
# --------------------------------------------------
print("🚀 Training classifier head...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks
)

# --------------------------------------------------
# Fine-Tuning Last Layers (Optional, CPU-Friendly)
# --------------------------------------------------
print("🔧 Fine-tuning backbone...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_FINE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=callbacks
)

# --------------------------------------------------
# Save & Convert
# --------------------------------------------------
model.save("mobilenetv2_insect.keras")
model.save("mobilenetv2_insect.h5")
model.save("mobilenetv2_insect_savedmodel")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open("mobilenetv2_insect.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Training & TFLite Export Completed Successfully")
