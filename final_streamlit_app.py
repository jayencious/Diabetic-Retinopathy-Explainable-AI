import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import os

# Streamlit Page Config
st.set_page_config(
    page_title="Active XAI DR Diagnostic Dashboard",
    page_icon="👁️",
    layout="wide"
)

# Model Caching
# Load the model only once and cache it for future use
@st.cache_resource
def load_dr_model():
    model_path = os.path.join("expert_checkpoints", "final_xai_corrected_model.keras")

    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Kindly ensure the model is available.")
        st.stop()
    
    return load_model(model_path)

model = load_dr_model()

# Preprocessing Function
def apply_graham_preprocessing(img_arr, target_size=380):
    """
    Applies Ben Graham's spatial colour blending to the input image array.
    """
    # Resize the img
    img_resized = cv2.resize(
        img_arr,
        (target_size, target_size)
    )
    blurred_img = cv2.GaussianBlur(
        img_resized,
        (0, 0),
        10
    )
    processed_img = cv2.addWeighted(
        img_resized,
        4,
        blurred_img,
        -4,
        128
    )

    return processed_img

# Prediction Function
def predict_ordinal_class(model, img_tensor):
    """
    Translates independent sigmoid inputs into an ordinal stage.
    """
    sigmoids = model.predict(
        img_tensor,
        verbose=0
    )[0]
    pred_class = int(np.sum(sigmoids > 0.5))
    class_names = [
        "No DR (0)",
        "Mild (1)",
        "Moderate (2)",
        "Severe (3)",
        "Proliferative DR (4)"
    ]

    return pred_class, class_names[pred_class], sigmoids

# Grad-CAM Function
def generate_gradcam_heatmap(model, img_tensor):
    """
    Generates a spatial heatmap mapping the gradients of the last convolutional layer to the output class.
    """
    # 1. Isolate the nested EfficientNet base and your custom classification head layers
    base_model = model.get_layer("efficientnetb4")
    gap_layer = model.get_layer("global_average_pooling2d")
    bn_layer = model.get_layer("batch_normalization")
    dropout_layer = model.get_layer("dropout")
    dense_layer = model.get_layer("dense")

    with tf.GradientTape() as tape:
        # 2. Pass the img thru the base model to get the final 4D spatial feature maps
        conv_outputs = base_model(img_tensor)

        # Explicitly command the GradientTape to track these spatial features
        tape.watch(conv_outputs)

        # 3. Pass those features through the rest of the custom head
        x = gap_layer(conv_outputs)
        x = bn_layer(x, training=False)
        x = dropout_layer(x, training=False)
        preds = dense_layer(x)

        # 4. Calculate the loss (we use the sum of active nodes for the ordinal head)
        loss = tf.reduce_sum(preds)
    
    # 5. Calculate the mathematically pure gradients of the loss with respect to the feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Average the gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Multiply each feature map channel by its "gradient importance" weight
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. Apply ReLU (discard negative values) and normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy().astype(np.float32)

def overlay_gradcam_heatmap(img_arr, heatmap):
    """
    Overlays the Grad-CAM heatmap on the original image.
    """

    heatmap = cv2.resize(
        heatmap,
        (img_arr.shape[1], img_arr.shape[0])
    )
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(
        img_arr,
        0.6,
        heatmap,
        0.4,
        0
    )

    return superimposed_img

# Lime Function
def get_lime_pred_func(model):
    """
    Wrapper to unspool ordinal probabilities for LIME.
    """
    def pred_fn(images):
        sigmoids = model.predict(
            images,
            verbose=0
        )

        p0 = 1.0 - sigmoids[:, 0]
        p1 = sigmoids[:, 0] - sigmoids[:, 1]
        p2 = sigmoids[:, 1] - sigmoids[:, 2]
        p3 = sigmoids[:, 2] - sigmoids[:, 3]
        p4 = sigmoids[:, 3]

        probs = np.column_stack(
            (p0, p1, p2, p3, p4)
        )
        probs = np.clip(
            probs,
            0.0,
            1.0
        )
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        return probs
    
    return pred_fn

def generate_lime_explanation(img_tensor, model):
    """
    Generates LIME superpixels.
    """
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_tensor[0].astype('double'),
        get_lime_pred_func(model),
        top_labels=1,
        hide_color=0,
        num_samples=250
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    return mark_boundaries(temp / 255.0, mask)

# UI
st.title("👁️ Active XAI Diabetic Retinopathy Diagnostic Dashboard")
st.markdown("Upload a raw retinal fundus image to view the ordinal diagnosis and the Dual-XAI audit (Grad-CAM and LIME).")

uploaded_file = st.sidebar.file_uploader(
    "Upload Fundus Image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)
run_lime = st.sidebar.checkbox("Run LIME Audit (Takes ~15 seconds)", value=False)

if uploaded_file is not None:
    # Read and process the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    st.sidebar.image(raw_img, caption="Raw Uploaded Image", use_container_width=True)

    with st.spinner("Applying Graham Normalization and Extracting Features..."):
        processed_img = apply_graham_preprocessing(raw_img, target_size=380)
        input_tensor = np.expand_dims(processed_img, axis=0).astype(np.float32)

        # Prediction
        pred_stage, class_name, sigmoids = predict_ordinal_class(model, input_tensor)

        st.markdown("---")
        st.subheader(f"🧠 Diagnostic Output: **{class_name}**")
        st.progress(pred_stage / 4.0)

        # Visual Layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 1. Graham Normalized Input")
            st.image(processed_img, use_container_width=True, caption="Illumination Neutralized")
        
        with col2:
            st.markdown("### 2. Grad-CAM Spatial Audit")
            heatmap = generate_gradcam_heatmap(model, input_tensor)
            gradcam_img = overlay_gradcam_heatmap(processed_img, heatmap)
            st.image(gradcam_img, use_container_width=True, caption="Gradient-Based Focus")
        
        with col3:
            st.markdown("### 3. LIME Superpixel Audit")

            if run_lime:
                with st.spinner("Simulating perturbations..."):
                    lime_img = generate_lime_explanation(input_tensor, model)
                    st.image(lime_img, use_container_width=True, caption="Perturbation-Based Focus")
            else:
                st.info("LIME audit skipped. Check the box in the sidebar to execute.")