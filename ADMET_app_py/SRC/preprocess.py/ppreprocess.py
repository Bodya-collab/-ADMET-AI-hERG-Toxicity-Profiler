import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover
import os

# Letter r !


RAW_FILE_PATH = r"C:\python projects\Project 8\data\hERG_raw.csv"


def preprocess_data():
    print(f" Reading file directly from: {RAW_FILE_PATH}")
    
    if not os.path.exists(RAW_FILE_PATH):
        print(" Error: Raw data file not found. Please run the data fetching script first.")
        return

    df = pd.read_csv(RAW_FILE_PATH)
    print(f" Loaded {len(df)} rows.")

    # Cleaning and preprocessing
    print("🧹 Cleaning data...")
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])
    df = df.drop_duplicates(subset=['canonical_smiles'])
    
    remover = SaltRemover.SaltRemover()
    valid_data = []
    
    for index, row in df.iterrows():
        try:
            smiles = row['canonical_smiles']
            val = float(row['standard_value'])
            
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = remover.StripMol(mol)
                clean_smiles = Chem.MolToSmiles(mol)
                # IC50 < 10000 nM -> Toxic
                label = 1 if val < 10000 else 0
                valid_data.append({'smiles': clean_smiles, 'label': label})
        except:
            continue
            
    clean_df = pd.DataFrame(valid_data)
    
    # Saving processed data
    output_path = RAW_FILE_PATH.replace("_raw.csv", "_processed.csv")
    clean_df.to_csv(output_path, index=False)
    
    # Results summary
    print("-" * 30)
    print(f"💾 Saved to: {output_path}")
    print(f"🔴 Toxic: {clean_df['label'].sum()}")
    print(f"🟢 Safe:  {len(clean_df) - clean_df['label'].sum()}")
    print("-" * 30)

if __name__ == "__main__":
    preprocess_data()