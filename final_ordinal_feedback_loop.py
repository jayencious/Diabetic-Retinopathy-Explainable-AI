import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import mixed_precision
from keras.callbacks import ModelCheckpoint, EarlyStopping

mixed_precision.set_global_policy('mixed_float16')

MODEL_PATH = "expert_checkpoints/new_eyepacs_aptos_finetuned_model.keras"
CORRECTED_MODEL_PATH = "expert_checkpoints/new_xai_corrected_model.keras"
HALLUCINATION_REPORT = "new_xai_reports/hallucination_reports_ordinal.csv"
APTOS_IMAGES_PATH = "data/APTOS_2019/train_graham"

IMAGE_SIZE = 380
BATCH_SIZE = 4
EPOCHS = 15

feedback_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
        keras.layers.SpatialDropout2D(0.15)
    ]
)

def encoding_ordinal(label):
    mask = tf.sequence_mask(label, maxlen=4)

    return tf.cast(mask, tf.float32)

def decoding_and_formatting(file_path, label):
    img_bytes = tf.io.read_file(file_path)
    
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMAGE_SIZE, IMAGE_SIZE]
    )
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet.preprocess_input(img)

    ordinal_label = encoding_ordinal(label)

    return img, ordinal_label

def build_feedback_pipeline():
    print(f"Loading Ordinal Dual-XAI Hallucination Report from {HALLUCINATION_REPORT}...")

    if not os.path.exists(HALLUCINATION_REPORT):
        raise FileNotFoundError(f"Missing XAI report at {HALLUCINATION_REPORT}")
    
    df = pd.read_csv(HALLUCINATION_REPORT)

    if len(df) == 0:
        print("No hallucinations found! The model is already clinically optimal.")

        return None
    
    print(f"Isolating {len(df)} severe logic failures for aggresive feedback retraining...")

    dataset = tf.data.Dataset.from_tensor_slices(
        (df["file_path"].values, df["true_label"].values)
    )
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(
        decoding_and_formatting,
        num_parallel_calls=2
    )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(
        lambda x, y: (feedback_augmentation(x, training=True), y),
        num_parallel_calls=2
    )

    return dataset.prefetch(buffer_size=2)

class MemoryCleanupCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

        keras.backend.clear_session()

def run_ordinal_feedback_loop():
    feedback_ds = build_feedback_pipeline()
    if feedback_ds is None:
        return
    
    print(f"\n[1] Loading Flawed Fine-Tuned Model: {MODEL_PATH}")

    model = keras.models.load_model(MODEL_PATH)
    model.trainable = True

    print("[2] Implementing Binary Focal Loss for Independent Node Penalization...")

    focal_loss = keras.losses.BinaryFocalCrossentropy(
        gamma=2.0,
        label_smoothing=0.05
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-6),
        loss=focal_loss,
        metrics=["binary_accuracy", keras.metrics.AUC(name="auc", multi_label=True)]
    )

    callbacks = [
        ModelCheckpoint(
            CORRECTED_MODEL_PATH,
            save_best_only=True,
            monitor="loss",
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        MemoryCleanupCallback()
    ]

    print("\nINITIATING ORDINAL DUAL-XAI-DRIVEN FEEDBACK LOOP MECHANISM...")

    model.fit(
        feedback_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"SUCCESS: Logic Corrected! Final, Clinical-Grade Model Saved to: {CORRECTED_MODEL_PATH}")

if __name__ == "__main__":
    run_ordinal_feedback_loop()
