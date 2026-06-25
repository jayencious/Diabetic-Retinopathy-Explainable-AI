# Expert Checkpoints Directory
- This directory contains the CNN definitions and acts as the designated storage location for the trained `.keras` model weights.

### ⚠️ IMPORTANT: Where are the trained weights?
- If you have cloned the repository, you will notice that the actual model weights files are missing.
- Deep learning models with millions of parameters generate massive weight files that easily exceed GitHub's strict 100MB file size limit.
- In order to maintain the repository performance and adhere to the best practices of version control, the heavy binary weight files are hosted externally.
- To run the project or the evaluation scripts locally, you must download the trained weights and place them inside this folder.

---

## Download the Trained Weights

### 1. Active XAI Core Model (EfficientNetB4 - Ordinal)
- This model contains the foundational weights trained on EyePACS dataset, generalized on APTOS 2019 dataset and heavily penalized using our closed-loop Spatial Dropout mechanism.
* **Download Link:** [Google Drive Model Download Link](https://drive.google.com/drive/folders/1h_upxRxm6BOyh1fnH9H4dEyeyB_lsPOP?usp=drive_link)
* **File 1 Name:** `final_eyepacs_foundation_model.keras`
* **File 2 Name:** `final_eyepacs_aptos_finetuned_model.keras`
* **File 3 Name:** `final_xai_corrected_model.keras`
* **Files Size:** ~337MB (Approx.)

---

## Expected Directory Structure
- Once you have downloaded the `.keras` files, place them directly inside this directory (`expert_checkpoints`).
- Your local structure must look exactly like this to successfully load the models into memory, when running the different project related scripts:

```text
expert_checkpoints/
│
├── README.md						# This file
├── final_eyepacs_aptos_finetuned_model.keras		# Downloaded Weights file finetuned on the APTOS 2019 dataset after being initially trained on the foundation EyePACS dataset
├── final_eyepacs_foundation_model.keras		# Downloaded foundational weights file (Primary requirement)
└──final_xai_corrected_model.keras			# Downloaded weights file corrected by using dual-xai feedback loop mechanism
```
