import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
# !! Update these paths to match your project !!

# Path to your PREPROCESSED image folders
PREPROCESSED_TRAIN_DIR = 'pre_processed_data/train'
PREPROCESSED_TEST_DIR = 'pre_processed_data/test'

# Path to your ORIGINAL IDRiD label CSV files
TRAIN_LABELS_CSV = 'IDRiD_data/Disease_Grading/Groundtruths/IDRiD_Disease Grading_Training_Labels.csv'
TEST_LABELS_CSV = 'IDRiD_data\Disease_Grading\Groundtruths\IDRiD_Disease_Grading_Testing_Labels.csv'

# Model & Training Parameters
IMG_SIZE = 224
NUM_CLASSES = 5  # (0, 1, 2, 3, 4)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
VALIDATION_SPLIT_SIZE = 0.2 # 20% of training data will be used for validation
RANDOM_SEED = 42

# --- 2. DATA LOADING & SPLITTING ---

def load_labels_and_paths(image_dir, csv_path):
    """
    Loads image paths and one-hot encoded labels from a CSV.
    Matches labels against *existing* preprocessed image files.
    """
    try:
        labels_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Label file not found at {csv_path}")
        return None, None
        
    # Standardize column names (IDRiD has different names in train/test files)
    if 'Image name' in labels_df.columns:
        image_col = 'Image name'
        label_col = 'Retinopathy grade'
    elif 'image' in labels_df.columns: # For test file
        image_col = 'image'
        label_col = 'level'
    else:
        print(f"Error: Could not find image/label columns in {csv_path}")
        return None, None

    # Filter for necessary columns
    labels_df = labels_df[[image_col, label_col]]
    
    # Create the full file path to the *preprocessed* images
    # Assuming preprocessed images were saved as .jpg
    labels_df['full_path'] = labels_df[image_col].apply(
        lambda x: os.path.join(image_dir, f"{x}.jpg")
    )
    
    # CRITICAL: Filter out any labels for images that don't exist
    print(f"Checking {len(labels_df)} labels against files in {image_dir}...")
    original_count = len(labels_df)
    labels_df = labels_df[labels_df['full_path'].apply(os.path.exists)]
    found_count = len(labels_df)
    
    if found_count == 0:
        print(f"Error: No matching preprocessed images found in {image_dir}.")
        return None, None
        
    print(f"Found {found_count} matching images (out of {original_count} labels).")
        
    image_paths = labels_df['full_path'].values
    labels = labels_df[label_col].values
    
    # Convert labels to one-hot encoding
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
    
    return image_paths, labels_one_hot

# --- 3. TF.DATA PIPELINE FUNCTIONS ---

def load_and_preprocess_image(image_path, label):
    """Loads, decodes, and normalizes an image for ResNet-50."""
    # Read the file
    img = tf.io.read_file(image_path)
    # Decode the JPEG
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize (good practice, though our images are already 224x224)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    
    # Normalize the image using the *exact* method ResNet50 was trained on
    img = tf.keras.applications.resnet50.preprocess_input(img)
    
    return img, label

def create_tf_dataset(image_paths, labels, is_training=True):
    """Creates an efficient, batched tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=RANDOM_SEED)
        
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

# --- 4. MAIN EXECUTION ---

def main():
    print("--- 🚀 Starting Data Preparation Script ---")
    
    # --- Step 1: Load and Create the TEST Dataset ---
    print("\n[Phase 1: Loading TEST Data]")
    print(f"Reading test labels from: {TEST_LABELS_CSV}")
    test_paths, test_labels_one_hot = load_labels_and_paths(
        PREPROCESSED_TEST_DIR, 
        TEST_LABELS_CSV
    )
    
    if test_paths is not None:
        test_ds = create_tf_dataset(test_paths, test_labels_one_hot, is_training=False)
        print("✅ Successfully created 'test_ds' (Test Dataset)")
    else:
        print("❌ Critical Error: Could not create 'test_ds'. Halting.")
        # !! CHANGE 1: Return None on failure !!
        return None, None, None, None

    # --- Step 2: Load, Split, and Create TRAIN/VALIDATION Datasets ---
    print("\n[Phase 2: Loading and Splitting TRAIN Data]")
    print(f"Reading training labels from: {TRAIN_LABELS_CSV}")
    all_train_paths, all_train_labels_one_hot = load_labels_and_paths(
        PREPROCESSED_TRAIN_DIR,
        TRAIN_LABELS_CSV
    )
    
    if all_train_paths is None:
        print("❌ Critical Error: Could not load training data. Halting.")
        # !! CHANGE 2: Return None on failure !!
        return None, None, None, None
        
    print(f"\nSplitting {len(all_train_paths)} training images into train/validation sets...")
    
    stratify_labels = np.argmax(all_train_labels_one_hot, axis=1)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_paths, 
        all_train_labels_one_hot, 
        test_size=VALIDATION_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_labels
    )
    
    print(f"-> New Training set size:   {len(train_paths)}")
    print(f"-> New Validation set size: {len(val_paths)}")

    train_ds = create_tf_dataset(train_paths, train_labels, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_labels, is_training=False)

    print("✅ Successfully created 'train_ds' (Training Dataset)")
    print("✅ Successfully created 'val_ds' (Validation Dataset)")
    
    print("\n--- 🏁 Data Preparation Complete ---")
    
    # !! CHANGE 3: Return the datasets for the other script to use !!
    return train_ds, val_ds, test_ds, test_labels_one_hot

# !! CHANGE 4: Modify this block !!
if __name__ == "__main__":
    print("Running data preparation script as standalone...")
    # This allows you to run this file by itself to test it
    main()