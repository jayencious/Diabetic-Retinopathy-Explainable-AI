import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import mixed_precision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

mixed_precision.set_global_policy('mixed_float16')

MODEL_PATH = "expert_checkpoints/new_xai_corrected_model.keras"

IDRID_CSV_PATH = "data/IDRID-dataset/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
IDRID_IMAGES_PATH = "data/IDRID-dataset/train_graham"

BLIND_EVAL_APTOS_RESULTS_PATH = "new_evaluation_results"

os.makedirs(BLIND_EVAL_APTOS_RESULTS_PATH, exist_ok=True)

IMAGE_SIZE = 380
BATCH_SIZE = 16

print("Loading Graham-Processed IDRiD Dataset...")

if not os.path.exists(IDRID_CSV_PATH):
    raise FileNotFoundError(f"Could not find IDRiD CSV at {IDRID_CSV_PATH}")

df = pd.read_csv(IDRID_CSV_PATH)

IMAGE_COLUMN = "Image name"
LABEL_COLUMN = "Retinopathy grade"

df["file_path"] = df[IMAGE_COLUMN].apply(
    lambda x: os.path.join(IDRID_IMAGES_PATH, f"{x}.png")
)

df = df[df["file_path"].apply(os.path.exists)]

print(f"Found {len(df)} blind test images from IDRiD.")

if len(df) == 0:
    raise ValueError("No images found! Check IDRID_IMAGES_PATH and make sure that Graham preprocessing was done right.")

def loading_and_preprocessing_img(file_path, label):
    img_bytes = tf.io.read_file(file_path)

    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMAGE_SIZE, IMAGE_SIZE]
    )
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet.preprocess_input(img)

    return img, label

test_ds = tf.data.Dataset.from_tensor_slices(
    (df["file_path"].values, df[LABEL_COLUMN].values)
)
test_ds = test_ds.map(
    loading_and_preprocessing_img,
    num_parallel_calls=2
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(2)

print(f"\nLoading Clinical-Grade Model: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)

print("\nINITIATING CROSS-DOMAIN INFERENCE ON IDRiD Dataset...")

raw_sigmoids = model.predict(test_ds)

predicted_classes = np.sum(raw_sigmoids > 0.5, axis=1)
true_labels = df[LABEL_COLUMN].values

print("\n\n" + "="*50)
print("FINAL IDRiD CROSS-DOMAIN RESULTS")
print("="*50)

exact_accuracy = accuracy_score(true_labels, predicted_classes)
baseline_accuracy = 0.4175
improvement = (exact_accuracy - baseline_accuracy) * 100

print(f"\nExact-Match Accuracy: {exact_accuracy * 100:.2f}%")
print(f"Previous Baseline:      {baseline_accuracy * 100:.2f}%")
print(f"Net Improvement:       +{improvement:.2f}%")

print("\nDetailed Clinical Report:")
print(classification_report(
    true_labels,
    predicted_classes,
    target_names=[
        "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"
    ]
    )
)

cm = confusion_matrix(true_labels, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
    yticklabels=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
)
plt.title(f"IDRiD Dataset Blind Test - Accuracy: {exact_accuracy*100:.2f}%\n(Active XAI + Ordinal Framework)")
plt.ylabel("True Clinical Grade")
plt.xlabel("AI Predicted Grade")
plt.savefig(
    os.path.join(BLIND_EVAL_APTOS_RESULTS_PATH, "new_idrid_final_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSUCCESS: Confusion Matrix saved to: {BLIND_EVAL_APTOS_RESULTS_PATH}/new_idrid_final_confusion_matrix.png")
print("="*50)
