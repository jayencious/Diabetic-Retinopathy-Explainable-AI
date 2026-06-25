import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import keras
from keras import mixed_precision
from sklearn.model_selection import train_test_split

mixed_precision.set_global_policy('mixed_float16')

MODEL_PATH = "expert_checkpoints/final_eyepacs_aptos_finetuned_model.keras"
APTOS_CSV_PATH = "data/APTOS_2019/train.csv"
APTOS_IMAGES_PATH = "data/APTOS_2019/train_graham"
NEW_XAI_REPORTS_PATH = "final_xai_reports"
NEW_XAI_VISUALS_PATH = os.path.join(NEW_XAI_REPORTS_PATH, "final_visualization")

os.makedirs(NEW_XAI_VISUALS_PATH, exist_ok=True)

IMAGE_SIZE = 380

print("Loading Graham-Processed APTOS 2019 Dataset...")

df = pd.read_csv(APTOS_CSV_PATH)
df["file_path"] = df["id_code"].apply(
    lambda x: os.path.join(APTOS_IMAGES_PATH, f"{x}.png")
)
df = df[df["file_path"].apply(os.path.exists)]

_, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["diagnosis"],
    random_state=42
)

xai_report_df = val_df.head(100)

print(f"Loading Fine-Tuned Model: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)

# Ordinal math translation
def predict_ordinal_class(sigmoid_outputs):
    """
    Converts the 4 sigmoid probabilities into a single 0-4 integer class.
    """

    return np.sum(sigmoid_outputs > 0.5)

def lime_prediction_func(images):
    """
    LIME requires a standard 5-class probability distribution summing to 1.
    We need to mathematically translate our 4 independent sigmoids back into 5 classes.
    """

    processed_images = keras.applications.efficientnet.preprocess_input(
        images.astype(np.float32)
    )
    sigmoids = model.predict(processed_images, verbose=0)

    # Translating Ordinal to multi-class probabilities
    p0 = 1.0 - sigmoids[:, 0]
    p1 = sigmoids[:, 0] - sigmoids[:, 1]
    p2 = sigmoids[:, 1] - sigmoids[:, 2]
    p3 = sigmoids[:, 2] - sigmoids[:, 3]
    p4 = sigmoids[: ,3]

    # Stacking and clipping to prevent the negative floating point errors
    probs = np.column_stack(
        (p0, p1, p2, p3, p4)
    )
    probs = np.clip(probs, 0.0, 1.0)

    # Normalization so that they sum up exactly to 1.0 for LIME
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    return probs

# Grad-CAM Implementation
def generate_gradcam(img_array):
    img_tensor = tf.expand_dims(img_array, axis=0)
    img_tensor = keras.applications.efficientnet.preprocess_input(
        tf.cast(img_tensor, tf.float32)
    )

    base_model = model.layers[1]
    last_convolution_layer_name = "top_activation"

    inner_model = keras.models.Model(
        inputs=base_model.inputs,
        outputs=[
            base_model.get_layer(last_convolution_layer_name).output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        convolution_outputs, x = inner_model(img_tensor)
        tape.watch(convolution_outputs)

        for layer in model.layers[2:]:
            x = layer(x)
        
        loss = tf.reduce_sum(x)
    
    grads = tape.gradient(loss, convolution_outputs)
    if grads is None:
        return np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.float32
        )
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    convolution_outputs = convolution_outputs[0]

    heatmap = convolution_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(heatmap.shape, dtype=np.float32)
    
    heatmap = heatmap /max_val
    heatmap = tf.cast(heatmap, tf.float32)

    return heatmap.numpy()

# Execution of Dual-XAI Audit Reports
def run_ordinal_dual_xai_audit():
    explainer = lime_image.LimeImageExplainer()
    hallucination_log = []

    print("\nINITIATING ORDINAL DUAL-XAI LOGIC AUDITING...")

    for idx, row in xai_report_df.iterrows():
        file_path = row["file_path"]
        true_label = int(row["diagnosis"])
        file_name = row["id_code"]

        raw_img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        img_tensor = keras.applications.efficientnet.preprocess_input(
            np.expand_dims(raw_img.astype(np.float32), axis=0)
        )

        sigmoids = model.predict(img_tensor, verbose=0)[0]
        predicted_class = predict_ordinal_class(sigmoids)

        if predicted_class == true_label:
            continue

        print(f"\n[FLAGGED] Image: {file_name} | Original Class: {true_label} | Predicted Class: {predicted_class} | Sigmoids: {np.round(sigmoids, 2)}")
        print("Generating Grad-CAM and LIME Explanations...")

        heatmap = generate_gradcam(raw_img)
        heatmap_resized = cv2.resize(
            heatmap,
            (IMAGE_SIZE, IMAGE_SIZE)
        )
        heatmap_resized = np.uint8(255 * heatmap_resized)

        colourmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        gradcam_overlay = cv2.addWeighted(raw_img, 0.6, colourmap, 0.4, 0)

        explanation = explainer.explain_instance(
            raw_img.astype('double'),
            lime_prediction_func,
            top_labels=5,
            hide_color=0,
            num_samples=500
        )

        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        lime_overlay = mark_boundaries(temp / 255.0, mask)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Original Image (Class: {true_label})")
        axes[1].imshow(gradcam_overlay)
        axes[1].set_title(f"Grad-CAM Image (Predicted: {predicted_class})")
        axes[2].imshow(lime_overlay)
        axes[2].set_title("Lime Superpixels")

        for ax in axes:
            ax.axis("off")
        
        save_path = os.path.join(NEW_XAI_VISUALS_PATH, f"{file_name}_audit.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        # Logs for Feedback Loop Mechanism
        hallucination_log.append(
            {
                "id_code": file_name,
                "file_path": file_path,
                "true_label": true_label,
                "predicted_label": predicted_class,
                "sigmoids": str(np.round(sigmoids, 3).tolist())
            }
        )
    
    final_xai_report_df = pd.DataFrame(hallucination_log)
    final_xai_report_df.to_csv(os.path.join(NEW_XAI_REPORTS_PATH, "hallucination_reports_ordinal.csv"), index=False)

    print(f"\nSUCCESS: XAI Audit Reports Complete! Found {len(hallucination_log)} logic failures.")

if __name__ == "__main__":
    run_ordinal_dual_xai_audit()
