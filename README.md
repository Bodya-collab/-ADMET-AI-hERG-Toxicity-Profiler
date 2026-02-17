# -ADMET-AI-hERG-Toxicity-Profiler
🧬 AI-driven tool for predicting cardiac toxicity (hERG blockage) in early-stage drug discovery using XGBoost &amp; RDKit.
# 🧬 ADMET-AI: hERG Toxicity Profiler

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io/)
[![RDKit](https://img.shields.io/badge/Cheminformatics-RDKit-green.svg)](https://www.rdkit.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

**ADMET-AI** is an *in silico* toxicology screening tool designed to predict **hERG channel blockage**—a primary cause of drug-induced cardiac arrhythmia (Long QT syndrome).

By leveraging machine learning (XGBoost) and cheminformatics (RDKit), this application helps researchers identify potential cardiotoxicity risks in the early stages of drug discovery, reducing the cost of late-stage failures.


## 📸 Screenshots

*<img width="1837" height="775" alt="image" src="https://github.com/user-attachments/assets/a6def210-0f06-4628-94ac-eaab08f10c9a" />
                                                        *Image 1* Main Dashboard 
*<img width="1389" height="692" alt="image" src="https://github.com/user-attachments/assets/5024eee4-49b3-43a3-b8e4-364f51ed8859" />
                                                        *Image 2* Toxicity according to ML prediction for Aspirin 
<img width="1358" height="663" alt="image" src="https://github.com/user-attachments/assets/84337fbf-2f6f-48e0-85cb-238011fd9431" />
                                                        *Image 3* Physicochemical properties 
<img width="1850" height="647" alt="image" src="https://github.com/user-attachments/assets/dcc2ad5f-6c52-458b-952a-f325ecd1854d" />
                                                        *Image 4* Batch Screening Module with random data  

## 🚀 Key Features

### 1. 🧪 Single Molecule Analysis
* **Real-time Prediction:** Input a SMILES string to get an instant toxicity probability.
* **Physicochemical Profiling:** Calculates Molecular Weight, LogP, TPSA, and H-Donors.
* **Bioavailability Radar:** Visualizes the "drug-likeness" balance using a radar chart.
* **Lipinski's Rule of 5:** Automatically checks if the molecule violates drug-likeness rules.

### 2. 🏭 High-Throughput Batch Screening
* **CSV Processing:** Upload datasets containing thousands of molecules.
* **Smart Parsing:** Automatically handles CSV delimiters (commas or semicolons).
* **Exportable Results:** Download a comprehensive report with `Safe/Toxic` labels and probability scores.

---

## 🧠 Model Performance

The model was trained on a curated dataset from **ChEMBL v33**, containing over **18,000 compounds** with experimental IC50 values.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.830** | Excellent discrimination between toxic and safe classes. |
| **Accuracy** | **~75%** | Overall prediction accuracy on the test set. |
| **Algorithm** | XGBoost | Gradient Boosting Classifier (Optimized). |
| **Features** | ECFP4 | Morgan Fingerprints (Radius=2, 2048 bits). |

---

## 🛠 Tech Stack

* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **Cheminformatics:** RDKit
* **Web Framework:** Streamlit
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib

---

## 💻 Installation & Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ADMET-AI.git](https://github.com/YOUR_USERNAME/ADMET-AI.git)
    cd ADMET-AI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

4.  **Open your browser:**
    The app will automatically launch at `http://localhost:8501`.

---

## 📂 Project Structure

```text
ADMET-AI/
├── data/
│   ├── hERG_processed.csv   # Cleaned training data
│   └── hERG_xgboost.pkl     # Trained ML Model file
├── src/
│   ├── preprocess.py        # Data cleaning & standardization script
│   └── train_model.py       # XGBoost training pipeline
├── app.py                   # Main Streamlit application
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
