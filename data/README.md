# Dataset Information

- This directory is intended to store the retinal fundus images that are used for training, validating and testing our Active XAI framework.

### ⚠️ IMPORTANT: Why are there no images in this repository?
- If you have cloned this repository, you will notice that this folder does not contain any images.
- The raw image datasets are strictly excluded from version for two critical reasons:
1. **GitHub Size Constraints:** Medical datasets contain thousands of high-resolution image files. Pushing gigabytes of image data violates GitHub's strict 100MB file limit and bloats the size of the repository.
2. **Medical Licensing and Data Privacy:** These datasets are distributed under specific data-sharing agreements for academic research purposes. Redistribution of these datasets publicly without authorization violates their Terms of Service and general medical data privacy norms.

- To run this project locally and reproduce the results, you are required to download the datasets from their official canonical sources and structure them as they are outlined below.

---

## Required Datasets and Download Links

### 1. EyePACS (Foundational Training)
- Used to establish the foundational convolutional weights, feature extraction boundaries and ordinal classification logic.
* **Source:** [Kaggle - Diabetic Retinopathy Detection 2015](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
* **Description:** A massive dataset consisting of left and right eye images, graded by clinicians.

### 2. APTOS 2019 Blindness Detection (Cross-Domain Validation and Fine-Tuning)
- Used to evaluate the architecture's capacity for cross-dataset generalization and fine-tune the model accordingly for better performance.
* **Source:** [Kaggle - APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
* **Description:** A large set of high-resolution retinal images captured under diverse conditions at the Aravind Eye Hospital, Madurai, India.

### 3. IDRiD (Blind Cross-Dataset Clinical Evaluation)
- Used exclusively as the completely blind, cross-domain testing set to evaluate the clinical safety and domain shift resistance (yielded the 99.25% No-DR recall rate).
* **Source:** [IEEE Dataport - Indian Diabetic Retinopathy Image Dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
* **Description:** Images captured at an eye clinic in Nanded, Maharashtra, India, providing a distinct hardware and illumination profile.

---

## Expected Local Directory Structure

- Once you have downloaded and extracted the datasets, organize them exactly as shown below. The scripts utilize the relative paths that depend on this exact architecture:

```text
data/
│
├── EyePACS/
│   ├── train/  	        					# .jpeg files
│   └── trainLabels.csv        					# Ground truth labels
│
├── APTOS_2019/
│   ├── train_images/         					        # .png files
│   └── train.csv             					        # Ground truth labels
│
└── IDRiD-dataset/
     └── B. Disease Grading/
     	  ├── 1. Original Images/
              ├── a. Training Set/    				# Original IDRiD training images
              ├── b. Testing Set/					# Original IDRiD test images
    	  └── 2. Groundtruths/
              ├── a. IDRiD_Disease Grading_Training Labels.csv
              └── b. IDRiD_Disease Grading_Testing Labels.csv		# Ground truth testing labels

```
