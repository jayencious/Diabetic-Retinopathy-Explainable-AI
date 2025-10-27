import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# !! Update these paths to match your original dataset structure !!

# Define the sets to process:
# (Input_Original_Folder, Output_Processed_Folder)
PATHS_TO_PROCESS = [
    (
        'IDRiD_data/Disease_Grading/Original_Images/Training_Set', 
        'pre_processed_data/train'
    ),
    (
        'IDRiD_data\Disease_Grading\Original_Images\Testing_Set', 
        'pre_processed_data/test'
    )
]

TARGET_SIZE = (224, 224) # Standard ResNet-50 input size
# ---------------------

def crop_black_borders(image):
    """
    Crops the image to remove black borders.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to find non-black areas
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
        
    # Find the bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image

def preprocess_and_save(image_path, output_path, clahe, target_size):
    """
    Loads, processes (crops, grayscales, CLAHE, resizes), 
    and saves a single image.
    """
    try:
        # 1. Load Image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return

        # 2. Crop black borders
        image = crop_black_borders(image)

        # 3. Resize
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # 4. Grayscaling
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 5. Apply CLAHE
        clahe_image = clahe.apply(gray_image)
        
        # 6. Replicate channels to create a 3-channel image for ResNet-50
        final_image = cv2.merge([clahe_image, clahe_image, clahe_image])
        
        # 7. Save the processed image
        cv2.imwrite(output_path, final_image)
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    """
    Main function to run the preprocessing pipeline for all defined paths.
    """
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Initialize CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if not PATHS_TO_PROCESS:
        print("Error: `PATHS_TO_PROCESS` list is empty. Nothing to do.")
        return

    print("--- Starting Preprocessing ---")

    # --- Loop through each path pair (e.g., train, then test) ---
    for input_dir, output_dir in PATHS_TO_PROCESS:
        
        print(f"\n--- Processing Directory ---")
        print(f"Input Source:  {input_dir}")
        print(f"Output Target: {output_dir}")

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all images in the input directory
        image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            # Use recursive=True if images are in subfolders
            image_paths.extend(glob.glob(os.path.join(input_dir, ext))) 
        
        if not image_paths:
            print(f"Warning: No images found in {input_dir}. Skipping.")
            continue

        print(f"Found {len(image_paths)} images. Starting preprocessing...")
        
        # Process all images for this directory
        for image_path in tqdm(image_paths, desc=f"Processing {os.path.basename(input_dir)}"):
            # Get the original filename (e.g., "IDRiD_001.jpg")
            filename = os.path.basename(image_path)
            # Create the full destination path
            output_path = os.path.join(output_dir, filename)
            
            preprocess_and_save(image_path, output_path, clahe, TARGET_SIZE)

        print(f"✅ Finished processing {input_dir}.")
        
    print("\n--- All Preprocessing Complete ---")
    print(f"Processed data is saved in your 'IDRID_Dataset_Processed' folder.")

if __name__ == "__main__":
    main()