import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import mixed_precision
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

mixed_precision.set_global_policy('mixed_float16')

APTOS_CSV_PATH = "data/APTOS_2019/train.csv"
APTOS_IMAGES_PATH = "data/APTOS_2019/train_graham"
FOUNDATIONAL_MODEL = "expert_checkpoints/new_eyepacs_foundation_model.keras"
FINETUNED_MODEL = "expert_checkpoints/new_eyepacs_aptos_finetuned_model.keras"

IMAGE_SIZE = 380
BATCH_SIZE = 16
EPOCHS = 15

# Ordinal encoding
def ordinal_encoding(label):
    # Converts 0-4 into a 4-node progressive array
    mask = tf.sequence_mask(label, maxlen=4)

    return tf.cast(mask, tf.float32)

def decoding_and_formatting(file_path, label):
    img_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet.preprocess_input(img)

    ordinal_label = ordinal_encoding(label)

    return img, ordinal_label

print("Loading Preprocessed APTOS Dataset...")

df = pd.read_csv(APTOS_CSV_PATH)
df["file_path"] = df["id_code"].apply(
    lambda x: os.path.join(APTOS_IMAGES_PATH, f"{x}.png")
)

df = df[df["file_path"].apply(os.path.exists)]

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["diagnosis"],
    random_state=42
)

print(f"Fine-tuning on {len(train_df)} images. Validating on {len(val_df)} images.")

def build_dataset(data_frame, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices(
        (data_frame["file_path"].values, data_frame["diagnosis"].values)
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.map(decoding_and_formatting, num_parallel_calls=2)
    dataset = dataset.batch(BATCH_SIZE)

    if is_training:
        augmentation_layer = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal_and_vertical"),
                keras.layers.RandomRotation(0.1),
            ]
        )
        dataset = dataset.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=2
        )
    
    return dataset.prefetch(buffer_size=2)

train_ds = build_dataset(train_df, is_training=True)
val_ds = build_dataset(val_df, is_training=False)

print(f"\nLoading Foundational Model: {FOUNDATIONAL_MODEL}...")

model = keras.models.load_model(FOUNDATIONAL_MODEL)

# Unfreezing the entire NN so the deep layers can adapt to the new threshold of APTOS dataset
model.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["binary_accuracy", keras.metrics.AUC(name="auc", multi_label=True)]
)

print("\nINITIATING EXPERT CALIBRATION OF THE FOUNDATION MODEL ON APTOS DATASET...")

callbacks = [
    ModelCheckpoint(FINETUNED_MODEL, save_best_only=True, monitor="val_auc", mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\nSUCCESS: Expert Calibration Complete! Fine-Tuned Model Saved to {FINETUNED_MODEL}")
