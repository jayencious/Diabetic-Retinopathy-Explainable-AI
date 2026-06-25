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

CSV_PATH = "data/EyePACS/trainLabels.csv"
IMAGES_PATH = "data/EyePACS/train_graham"
MODEL_SAVE_PATH = "expert_checkpoints/final_eyepacs_foundation_model.keras"

os.makedirs('expert_checkpoints', exist_ok=True)

IMAGE_SIZE = 380
BATCH_SIZE = 16
EPOCHS = 20

# Ordinal encoding
def ordinal_encoding(label):
    """
    Converts an image label (0-4) into a 4-element binary array.
    Example: Label 3 -> [1.0, 1.0, 1.0, 0.0]
    """

    # Creating a boolean mask and casting to float32
    mask = tf.sequence_mask(label, maxlen=4)

    return tf.cast(mask, tf.float32)

def decoding_and_formatting(file_path, label):
    # Native C++ decoding for speed
    img_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet.preprocess_input(img)

    # applying ordinal encoding
    ordinal_label = ordinal_encoding(label)

    return img, ordinal_label

print("Loading EyePACS Dataset...")

df = pd.read_csv(CSV_PATH)
df["file_path"] = df["image"].apply(
    lambda x: os.path.join(IMAGES_PATH, f"{x}.png")
)

# Filtering out any img that fails preprocessing
df = df[df["file_path"].apply(os.path.exists)]

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["level"],
    random_state=42
)

print(f"Training on {len(train_df)} images. Validating on {len(val_df)} images.")

def build_dataset(dataframe, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["file_path"].values, dataframe["level"].values)
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.map(
        decoding_and_formatting,
        num_parallel_calls=2
    )
    dataset = dataset.batch(BATCH_SIZE)

    if is_training:
        augmentation_layer = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal_and_vertical"),
                keras.layers.RandomRotation(0.15),
                keras.layers.RandomZoom(0.1)
            ]
        )

        dataset = dataset.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=2
        )
    
    return dataset.prefetch(buffer_size=2)

train_ds = build_dataset(train_df, is_training=True)
val_ds = build_dataset(val_df, is_training=False)

# Clinical ordinal architecture
def build_ordinal_model():
    base_model = keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # Creating custom ordinal head
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)

    outputs = keras.layers.Dense(
        4,
        activation="sigmoid",
        dtype="float32"
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["binary_accuracy", keras.metrics.AUC(name="auc", multi_label=True)]
    )

    return model

print("\nINITIATING FOUNDATIONAL MODEL TRAINING...")

model = build_ordinal_model()

callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_auc", mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\nFoundational Model successfully saved to {MODEL_SAVE_PATH}")
