#!/usr/bin/env python

import os
import sys
import requests
from tqdm import tqdm

GITHUB_RELEASE_BASE = "https://github.com/daklab/IsoformGazer/releases/download/v0.0.0"
ISOFORM_URL = f"{GITHUB_RELEASE_BASE}/mt_isoform_gazers_250514.tsv"
JUNCTION_URL = f"{GITHUB_RELEASE_BASE}/pseudobulk_final_broad_cell_type_20250514_072922.csv"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ISOFORM_OUTPUT_PATH = os.path.join(DATA_DIR, "mt_isoform_gazers_250514.tsv")
JUNCTION_OUTPUT_PATH = os.path.join(DATA_DIR, "pseudobulk_final_broad_cell_type_20250514_072922.csv")


def download_from_github_release(url, destination, data_type):
    """
    Download master table asset files from GitHub pre-release tagged v0.0.0. 
    """
    print(f"Downloading {os.path.basename(destination)}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    file_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Successfully downloaded {data_type}-level master table data to {destination}")


def main():
    print("IsoformGazer Master Table Data Downloader")
    print("--------------------------")
    print(f"Data directory: {DATA_DIR}")
    print("Starting downloads...\n")
    print("Downloading isoform-level master table data...")
    download_from_github_release(ISOFORM_URL, ISOFORM_OUTPUT_PATH)
    print("Downloading junction-level master table data...")
    download_from_github_release(JUNCTION_URL, JUNCTION_OUTPUT_PATH)
    print("\nDownload process complete.")
    

if __name__ == "__main__":
    main()
