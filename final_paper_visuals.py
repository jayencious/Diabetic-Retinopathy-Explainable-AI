import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import mixed_precision
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

mixed_precision.set_global_policy('mixed_float16')

MODEL_PATH = "expert_checkpoints/new_xai_corrected_model.keras"
IDRID_CSV_PATH = "data/IDRID-dataset/B. Disease Grading/2. Groundtruths/a. IDRID_Disease Grading_Training Labels.csv"
IDRID_IMAGES_PATH = "data/IDRID-dataset/train_graham"

PAPER_FIGURES_PATH = "master_research_results/research_paper_figures"

os.makedirs(PAPER_FIGURES_PATH, exist_ok=True)

IMAGE_SIZE = 380
BATCH_SIZE = 16

print("Generating High-Resolution Research Paper Figures...")

df = pd.read_csv(IDRID_CSV_PATH)
df["file_path"] = df["Image name"].apply(
    lambda x: os.path.join(IDRID_IMAGES_PATH, f"{x}.png")
)
df = df[df["file_path"].apply(os.path.exists)]

def loading_and_preprocessing_img(file_path, label):
    img_bytes = tf.io.read_file(file_path)

    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet.preprocess_input(img)

    return img, label

dataset = tf.data.Dataset.from_tensor_slices(
    (df["file_path"].values, df["Retinopathy grade"].values)
)
dataset = dataset.map(loading_and_preprocessing_img, num_parallel_calls=2)
dataset = dataset.batch(BATCH_SIZE).prefetch(2)

model = keras.models.load_model(MODEL_PATH)

raw_sigmoids = model.predict(dataset, verbose=1)

true_labels = df["Retinopathy grade"].values
predicted_classes = np.sum(raw_sigmoids > 0.5, axis=1)

p0 = 1.0 - raw_sigmoids[:, 0]
p1 = raw_sigmoids[:, 0] - raw_sigmoids[:, 1]
p2 = raw_sigmoids[:, 1] - raw_sigmoids[:, 2]
p3 = raw_sigmoids[:, 2] - raw_sigmoids[:, 3]
p4 = raw_sigmoids[:, 3]

class_probs = np.column_stack(
    (p0, p1, p2, p3, p4)
)
class_probs = np.clip(
    class_probs, 0.0, 1.0
)
class_probs = class_probs / np.sum(class_probs, axis=1, keepdims=True)

true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3, 4])
class_names = ["0 - Healthy", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"]
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

plt.figure(figsize=(10, 8))
for idx, colour in zip(range(5), colours):
    if np.sum(true_labels_bin[:, idx]) > 0:
        fpr, tpr, _ = roc_curve(true_labels_bin[:, idx], class_probs[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colour, lw=2.5, label=f"ROC {class_names[idx]} (AUC - {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold")
plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold")
plt.title("Receiver Operating Characteristic (ROC) - IDRiD Blind Test", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

roc_path = os.path.join(PAPER_FIGURES_PATH, "Fig1_ROC_Curve.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"SUCCESS: Saved ROC Curve to: {roc_path}")

precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, labels=[0,1,2,3,4], zero_division=0)

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, precision * 100, width, label="Precision", color="#4C72B0")
rects2 = ax.bar(x, recall * 100, width, label="Recall (Sensitivity)", color="#55A868")
rects3 = ax.bar(x + width, f1_score * 100, width, label="F1-Score", color="#C44E52")

ax.set_ylabel("Percentage (%)", fontsize=14, fontweight="bold")
ax.set_title("Class-Wise Diagnostic Performance - IDRiD Dataset", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=12)
ax.legend(fontsize=12, loc="lower right")
ax.set_ylim([0, 110])
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

bar_path = os.path.join(PAPER_FIGURES_PATH, "Fig2_Class_Metrics_Bar.png")
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"SUCCESS: Saved Performance Bar Chart to: {bar_path}")

print("\n" + "="*80)
print("Model A (Baseline): 41.75% Accuracy for IDRiD Dataset")
print(f"Model B (Our Framework): {np.mean(true_labels == predicted_classes)*100:.2f}% Accuracy")
print("="*80)
