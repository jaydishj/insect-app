# ==================================================
# INSECT CLASSIFICATION - MobileNetV2 (FULL PIPELINE)
# ==================================================

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# ======================
# 1. PATH CONFIGURATION
# ======================

TRAIN_DIR = r"c:/Users/Admin/Downloads/classification/train"
VAL_DIR   = r"c:/Users/Admin/Downloads/classification/val"
TEST_DIR  = r"c:/Users/Admin/Downloads/classification/test"

IMG_SIZE = 190
BATCH_SIZE = 8
EPOCHS_1 = 5
EPOCHS_2 = 5

# ======================
# 2. LOAD DATASETS
# ======================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=123
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_ds.class_names
num_classes = len(class_names)

print("✅ Classes:", class_names)
print("✅ Number of classes:", num_classes)

# ======================
# 3. DATA OPTIMIZATION
# ======================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ======================
# 4. BUILD MODEL
# ======================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Phase 1 freeze

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),   # ✅ FIX
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# 5. TRAIN PHASE 1
# ======================
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_1
)

# ======================
# 6. FINE-TUNING PHASE 2
# ======================
base_model.trainable = True

fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_2
)

# ======================
# 7. EVALUATION
# ======================
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ======================
# 8. SAVE MODEL (ALL FORMATS)
# ======================
model.save("mobilenetv2_insect.keras")
model.save("mobilenetv2_insect.h5")
model.save("mobilenetv2_insect_savedmodel")

with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("💾 Model & class names saved")

# ======================
# 9. TFLITE CONVERSION
# ======================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("mobilenetv2_insect.tflite", "wb") as f:
    f.write(tflite_model)

print("📱 TFLite model saved (float16 optimized)")

# ======================
# 10. PLOTS
# ======================
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(8,5))
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.title("Loss")
plt.show()

print("🎉 TRAINING + EXPORT COMPLETED SUCCESSFULLY")
