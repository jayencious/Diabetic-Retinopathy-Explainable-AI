# Diabetic Retinopathy Active XAI Framework

![Python 3.8+](https://img.shields.io/badge/python-3.13+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Spatial%20Blending-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)

## Project Abstract
- A domain-agnostic deep learning pipeline that transitions Explainable AI (XAI) from a passive observation tool into an **active logic-correction mechanism**.
- Standard clinical AI models frequently collapse under domain shifts because they memorize spurious camera artifacts rather than biological lesions.
- This framework resolves that bottleneck by integrating **Ben Graham's spatial colour normalization**, an **EfficientNetB4** backbone and a mathematical **Ordinal Regression** constraint.
- Most critically, the pipeline utilizes a closed-loop Dual-XAI audit (Grad-CAM and LIME) to automatically flag the logical hallucinations and actively penalize the network using Spatial Dropout and Binary Focal Cross-entropy which forces it to unlearn the camera artifacts.

## Clinical Performance and Results
- The framework was systematically trained and evaluated across three major datasets to test for foundational extraction, generalization and severe domain shift resistance.
- The most significant metric is the **99.25% recall rate on baseline healthy eyes** during the blind cross-domain test which proves the model's exceptional safety profile for real-world clinical triage.

| Evaluation Phase | Dataset | Macro-AUC | Exact-Match Accuracy | Key Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Foundational Training** | EyePACS | 0.8563 | 77.16% | Established robust feature boundaries. |
| **Cross-Domain Validation** | APTOS 2019 | 0.9601 | 84.90% | Demonstrated high transferability. |
| **Blind Clinical Test** | IDRiD | 0.7572 | - | **99.25% Recall (No-DR Cases)** |

---

## Visualizing the Active XAI Logic

![Dual-XAI Audit Example-1](https://github.com/jayencious/Diabetic-Retinopathy-Explainable-AI/blob/master/final_xai_reports/final_visualization/0104b032c141_audit.png)
![Dual-XAI Audit Example-2](https://github.com/jayencious/Diabetic-Retinopathy-Explainable-AI/blob/master/final_xai_reports/final_visualization/1638404f385c_audit.png)
![Dual-XAI Audit Example-3](https://github.com/jayencious/Diabetic-Retinopathy-Explainable-AI/blob/master/final_xai_reports/final_visualization/b37aae3c8fe1_audit.png)
![Dual-XAI Audit Example-4](https://github.com/jayencious/Diabetic-Retinopathy-Explainable-AI/blob/master/final_xai_reports/final_visualization/f576e45d1da2_audit.png)
![Dual-XAI Audit Example-5](https://github.com/jayencious/Diabetic-Retinopathy-Explainable-AI/blob/master/final_xai_reports/final_visualization/ea9e0fb6fb0b_audit.png)


> Our system's active XAI audit. The Grad-CAM heatmap and LIME superpixels successfully converge on exact pathological lesions (microaneurysms and hard exudates) rather than camera borders or lens dust, proving domain-agnostic feature extraction.

---

## Repository Structure

```text
Diabetic-Retinopathy-Explainable-AI/
│
├── data/											# (See data/README.md for information regarding the datasets and the source links to download them)
├── expert_checkpoints/								# (See expert_checkpoints/README.md for more information regarding the directory)
├── final_evaluation_results/
│   └── final_idrid_final_confusion_matrix.png		# IDRiD Dataset Blind Test (Active XAI + Ordinal Framework Confusion Matrix)
├── final_xai_reports/
│   └── final_visualization/						# Contains the dual-xai audit logic visualization subplots (Original Image + Grad-CAM Image + Lime Superpixels)
├── .gitignore
├── final_dual_xai.py								# Dual-XAI Audit Reports Logic
├── final_eyepacs_transfer_learning_aptos.py		# Transfer learning logic for the foundational model on the APTOS 2019 dataset (fine-tuning)
├── final_graham_preprocessing.py					# Ben Graham's Preprocessing Logic for the dataset images before model training
├── final_idrid_evaluation.py						# Blind Cross-Dataset Validation on the IDRiD dataset for proving the model's clinical implementation capability
├── final_master_evaluation.py						# Logic for generating the evaluation (validation) results of the model's performance on each dataset (foundational data, transfer learning or fine-tuning data and blind validation dataset)
├── final_model_training.py							# Foundation model training logic on the EyePACS dataset
├── final_ordinal_feedback_loop.py					# Logic for Ordinal Dual-XAI-Driven Feedback Loop Mechanism
├── final_streamlit_app.py							# Streamlit offline clinical dashboard
├── requirements.txt								# Python dependencies utilized in the development of this project
```

---

## Installation and Usage
- To ensure patient data privacy, this diagnostic dashboard is designed to run entirely offline on your local machine.

1. Clone the Repository
```bash
git clone [https://github.com/jayencious/Diabetic-Retinopathy-Explainable-AI.git](https://github.com/jayencious/Diabetic-Retinopathy-Explainable-AI.git)
cd Diabetic-Retinopathy-Explainable-AI
```

2. Install Dependencies
- It is highly recommended to use a virtual environment (`venv` or `conda`).
- Preferred virtual environment: `venv`.
```bash
pip install -r requirements.txt
```

3. Download the Pre-Trained Weights
- Due to GitHub file size constraints, the final `.keras` model weights are hosted externally.
a) Download the trained model files from here: [Google Drive Link](https://drive.google.com/drive/folders/1h_upxRxm6BOyh1fnH9H4dEyeyB_lsPOP?usp=sharing)
b) Place the files in the folder into the `/expert_checkpoints/` directory.

4. Run the Clinical Dashboard
- Launch the interactive offline Streamlit interface:
```bash
streamlit run final_streamlit_app.py
```

- This will open the dashboard in your default web browser at `http://localhost:8501`.
- Upload any raw retinal fundus image to view the normalized input, ordinal predictions, ben-graham preprocessed image and the real-time XAI logic audit.

## License and Academic Integrity
- The project is developed as a research thesis.
- The datasets utilized - EyePACS, APTOS 2019, IDRiD - are subject to their respective medical licensing agreements and are not redistributed within this repository.
