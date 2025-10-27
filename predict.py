import os
import cv2
import numpy as np
import tensorflow as tf

# --- 1. CONFIGURATION ---
MODEL_PATH = 'dr_best_model.keras'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Class 0 (No DR)', 'Class 1 (Mild)', 'Class 2 (Moderate)', 'Class 3 (Severe)', 'Class 4 (Proliferative)']

# --- 2. PREPROCESSING FUNCTIONS ---
# These MUST be the *exact same* steps you used for training.

def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return image[y:y+h, x:x+w]

def preprocess_image(image_path):
    """
    Loads and applies all preprocessing steps to a single image.
    """
    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # 2. Crop
    image = crop_black_borders(image)
    
    # 3. Resize
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 4. Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 5. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)

    # 6. Replicate Channels
    final_image = cv2.merge([clahe_image, clahe_image, clahe_image])
    
    # 7. ResNet-50 Preprocessing
    # This step adds batch dimension and normalizes the pixels
    img_array = tf.keras.applications.resnet50.preprocess_input(final_image)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch of 1
    
    return img_array

# --- 3. MAIN PREDICTION ---

def main():
    print("--- 🩺 Diabetic Retinopathy Prediction Tool ---")
    
    # --- !! CHANGE THIS to an image you want to test !! ---
    # Use an image from your *original* test set
    TEST_IMAGE_PATH = 'image.png'
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        print("Please update the TEST_IMAGE_PATH variable in this script.")
        return

    try:
        # 1. Load the trained model
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")

        # 2. Preprocess the test image
        print(f"Loading and preprocessing image: {TEST_IMAGE_PATH}...")
        processed_image = preprocess_image(TEST_IMAGE_PATH)

        # 3. Make a prediction
        print("Running prediction...")
        prediction = model.predict(processed_image)
        
        # 4. Interpret the result
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction) * 100
        
        print("\n--- 🔬 Prediction Result ---")
        print(f"  Image:       {os.path.basename(TEST_IMAGE_PATH)}")
        print(f"  Diagnosis:   {predicted_class_name}")
        print(f"  Confidence:  {confidence:.2f}%")
        
        print("\nFull Prediction Probabilities:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {prediction[0][i]*100:.2f}%")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()