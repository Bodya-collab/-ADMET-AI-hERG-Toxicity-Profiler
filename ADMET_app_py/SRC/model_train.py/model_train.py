import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pickle
import os

PROCESSED_FILE_PATH = r"C:\python projects\Project 8\data\hERG_processed.csv"

def train():
    print(f" Loading data from: {PROCESSED_FILE_PATH}")
    
    if not os.path.exists(PROCESSED_FILE_PATH):
        print(" Error: Data file not found.")
        return

    df = pd.read_csv(PROCESSED_FILE_PATH)
    
    # 1. FEATURIZATION (SMILES -> Numbers)
    print(" Generating Morgan Fingerprints (Radius=2, Bits=2048)...")
    
    X = []
    y = []
    
    valid_indices = []
    
    # Transforming SMILES to numerical features using RDKit's Morgan Fingerprints
    for idx, smile in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            # Vector of 2048 bits, where each bit represents the presence of a particular substructure
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.array(fp)
            X.append(arr)
            y.append(df['label'].iloc[idx])
            valid_indices.append(idx)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f" Matrix shape: {X.shape}")
    
    # 2. SPLITTING (Train / Test)
    # 20% of data will be used for testing, 80% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. TRAINING (XGBoost)
    print("Training XGBoost Classifier...")
    
    model = xgb.XGBClassifier(
        n_estimators=200,      # Amount of trees (more = better, but slower)
        learning_rate=0.05,    # Spped of learning (lower = more accurate, but slower)
        max_depth=6,           # Deepth of each tree (higher = more complex, but risk of overfitting)
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 4. EVALUATION 
    print("⚖️  Evaluating model...")
    
    # Predictions and probabilities for the test set
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] # Pobability of being in class 1 (toxic)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f" FINAL RESULTS (Test Set):")
    print(f"   Accuracy: {acc:.2%} ")
    print(f"   ROC-AUC:  {auc:.3f} ")
    print("="*40)
    print("Confusion Matrix (Матрица ошибок):")
    print(f"True Negatives (Correctly Safe): {cm[0][0]}")
    print(f"False Positives (False Alarm):   {cm[0][1]}")
    print(f"False Negatives (Missed Danger): {cm[1][0]}")
    print(f"True Positives (Correctly Toxic):{cm[1][1]}")
    print("="*40)
    
    # 5. SAVING
    # In folder where is CSV file, save model as 'hERG_xgboost.pkl'
    base_directory = os.path.dirname(PROCESSED_FILE_PATH)
    save_path = os.path.join(base_directory, 'hERG_xgboost.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f" Model saved to: {save_path}")

if __name__ == "__main__":
    train()