import os
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# For EyePACS dataset
# CSV_PATH = "data/EyePACS/trainLabels.csv"
# ORIGINAL_IMAGES_PATH = "data/EyePACS/train"
# OUTPUT_IMAGES_PATH = "data/EyePACS/train_graham"

# ID_COLUMN = "image"
# INPUT_EXTENSION = ".jpeg"
# IMAGE_SIZE = 380

# For APTOS 2019 dataset
# CSV_PATH = "data/APTOS_2019/train.csv"
# ORIGINAL_IMAGES_PATH = "data/APTOS_2019/train_images"
# OUTPUT_IMAGES_PATH = "data/APTOS_2019/train_graham"

# ID_COLUMN = "id_code"
# INPUT_EXTENSION = ".png"
# IMAGE_SIZE = 380

# For IDRiD dataset
CSV_PATH = "data/IDRID-dataset/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
ORIGINAL_IMAGES_PATH = "data/IDRID-dataset/B. Disease Grading/1. Original Images/a. Training Set"
OUTPUT_IMAGES_PATH = "data/IDRID-dataset/test_graham"

ID_COLUMN = "Image name"
INPUT_EXTENSION = ".jpg"
IMAGE_SIZE = 380

os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)

# Graham Spatial Colour Blending
def apply_graham_pipeline(file_name):
    input_path = os.path.join(ORIGINAL_IMAGES_PATH, f"{file_name}{INPUT_EXTENSION}")
    output_path = os.path.join(OUTPUT_IMAGES_PATH, f"{file_name}.png")

    if os.path.exists(output_path):
        return True
    
    img = cv2.imread(input_path)
    if img is None:
        return False
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ROI Extraction (Means to crop the dead space)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        img = img[y:y+h, x:x+w]
    
    # Standardizing the size using Lanczos4 for edge preservation
    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LANCZOS4
    )

    # Ben Graham's Spatial Colour Blending
    # 4 * img - 4 * GaussianBlur + 128
    sigmaX = IMAGE_SIZE / 30.0
    gaussian_blur = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX
    )
    img_graham = cv2.addWeighted(
        img,
        4,
        gaussian_blur,
        -4,
        128
    )

    # Converting the img back to BGR to save thru OpenCV
    cv2.imwrite(
        output_path,
        cv2.cvtColor(img_graham, cv2.COLOR_RGB2BGR)
    )

    return True

# Processing the dataset
def dataset_processing():
    print(f"Loading Data from {CSV_PATH}...")

    df = pd.read_csv(CSV_PATH, dtype={ ID_COLUMN: str })
    file_names = df[ID_COLUMN].tolist()

    print(f"Starting Graham Preprocessing on {len(file_names)} images...")

    # To implement multi-threading
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(
            executor.map(apply_graham_pipeline, file_names),
            total=len(file_names))
        )
    
    print("\nSUCCESS: Graham Preprocessing Complete! Lighting is fully normalized.")

if __name__ == "__main__":
    dataset_processing()