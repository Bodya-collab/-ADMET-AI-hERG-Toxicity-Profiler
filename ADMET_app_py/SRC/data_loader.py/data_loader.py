import pandas as pd
from chembl_webresource_client.new_client import new_client
import os
import time


def fetch_chembl_data(target_chembl_id, output_filename):
    print(f"Connecting to ChEMBL database for target: {target_chembl_id}...")

    # Client Inicialization
    activity = new_client.activity

    # IC50 data retrieval
    print("⏳ Downloading data... This might take a minute.")
    start_time = time.time()
    res = activity.filter(target_chembl_id=target_chembl_id).filter(
        standard_type="IC50"
    )
    print(f"⏱️  API query completed in {time.time() - start_time:.2f} seconds")

    # Convertation to Df
    print("📊 Converting to DataFrame...")
    start_time = time.time()
    df = pd.DataFrame.from_dict(res)
    print(f"✅ Downloaded {len(df)} records in {time.time() - start_time:.2f} seconds")

    # Creating data directory if not exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Saving to CSV
    output_path = os.path.join("data", output_filename)
    df.to_csv(output_path, index=False)
    print(f" Raw data saved to: {output_path}")


if __name__ == "__main__":
    # CHEMBL240 — is hERG  target ID in ChEMBL database
    fetch_chembl_data("CHEMBL240", "hERG_raw.csv")
