import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import mixed_precision
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

mixed_precision.set_global_policy('mixed_float16')

MODEL_PATH = "expert_checkpoints/final_xai_corrected_model.keras"
RESULTS_PATH = "master_research_results"

os.makedirs(RESULTS_PATH, exist_ok=True)

IMAGE_SIZE = 380
BATCH_SIZE = 16

DATASETS = {
    "APTOS_2019": {
        "csv_path": "data/APTOS_2019/train.csv",
        "img_dir": "data/APTOS_2019/train_graham",
        "img_col": "id_code",
        "label_col": "diagnosis"
    },
    "IDRiD": {
        "csv_path": "data/IDRID-dataset/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        "img_dir": "data/IDRID-dataset/train_graham",
        "img_col": "Image name",
        "label_col": "Retinopathy grade"
    },
    "EyePACS": {
        "csv_path": "data/EyePACS/trainLabels.csv",
        "img_dir": "data/EyePACS/train_graham",
        "img_col": "image",
        "label_col": "level"
    }
}

def loading_and_preprocessing_img(file_path, label):
    img_bytes = tf.io.read_file(file_path)

    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMAGE_SIZE, IMAGE_SIZE]
    )
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet.preprocess_input(img)

    return img, label

def translation_ordinal_to_multiclass(sigmoids):
    """
    Translates 4 ordinal sigmoids into 5 mutually exclusive class probabilities for AUC.
    """

    p0 = 1.0 - sigmoids[:, 0]
    p1 = sigmoids[:, 0] - sigmoids[:, 1]
    p2 = sigmoids[:, 1] - sigmoids[:, 2]
    p3 = sigmoids[:, 2] - sigmoids[:, 3]
    p4 = sigmoids[:, 3]

    probs = np.column_stack(
        (p0, p1, p2, p3, p4)
    )
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    return probs

def dataset_evaluation(name, config, model):
    print(f"\n{'='*60}")
    print(f"EVALUATING DATASET: {name}...")
    print(f"{'='*60}")

    if not os.path.exists(config["csv_path"]):
        print(f"FAILURE: CSV not found at {config["csv_path"]}. Skipping...")

        return
    
    df = pd.read_csv(config["csv_path"])
    df["file_path"] = df[config["img_col"]].apply(
        lambda x: os.path.join(config["img_dir"], f"{x}.png")
    )
    df = df[df["file_path"].apply(os.path.exists)]

    print(f"Loaded {len(df)} images.")
    if len(df) == 0:
        return
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["file_path"].values, df[config["label_col"]].values)
    )
    dataset = dataset.map(loading_and_preprocessing_img, num_parallel_calls=2)
    dataset = dataset.batch(BATCH_SIZE).prefetch(2)

    print("Running Inference...")

    raw_sigmoids = model.predict(dataset)

    true_labels = df[config["label_col"]].values
    predicted_class = np.sum(raw_sigmoids > 0.5, axis=1)
    class_probs = translation_ordinal_to_multiclass(raw_sigmoids)

    # Metrics Calculation
    exact_accuracy = accuracy_score(true_labels, predicted_class)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_class, labels=[0, 1, 2, 3, 4], zero_division=0)

    true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3, 4])

    try:
        auc_scores = roc_auc_score(true_labels_bin, class_probs, multi_class="ovr", average=None)
    except ValueError:
        auc_scores = [0, 0, 0, 0, 0]
    
    print(f"\n{name} CLINICAL METRICS")
    print(f"Overall Exact-Match Accuracy: {exact_accuracy * 100:.2f}%")
    print("-"*80)
    print(f"{'Class':<20} | {'AUC':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-"*80)

    class_names = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"]
    for i in range(5):
        print(f"{class_names[i]:<20} | {auc_scores[i]:<10.4f} | {f1_score[i]:<10.4} | {precision[i]:<10.4f} | {recall[i]:<10.4f}")
    
    print("-"*80)
    print(f"{'Macro Average':<20} | {np.mean(auc_scores):<10.4f} | {np.mean(f1_score):<10.4f} | {np.mean(precision):<10.4f} | {np.mean(recall):<10.4f}")

    cm = confusion_matrix(true_labels, predicted_class, labels=[0,1,2,3,4])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        yticklabels=["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    )
    plt.title(f"{name} Evaluation - Accuracy: {exact_accuracy*100:.2f}%\n(Active XAI + Ordinal Framework)")
    plt.ylabel("True Clinical Grade")
    plt.xlabel("AI Predicted Grade")

    cm_path = os.path.join(RESULTS_PATH, f"{name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion Matrix Saved to: {cm_path}")

if __name__ == "__main__":
    print(f"Loading Clinical-Grade Model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    master_model = keras.models.load_model(MODEL_PATH)

    for dataset_name, dataset_config in DATASETS.items():
        dataset_evaluation(
            dataset_name,
            dataset_config,
            master_model
        )
    
    print("\n" + "="*60)
    print("MASTER EVALUATION COMPLETE! ALL THE RESULTS ARE SAVED.")
    print("="*60)
