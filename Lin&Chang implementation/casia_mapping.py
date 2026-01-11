import os
import re
import csv
from glob import glob
from tqdm import tqdm

# =========================================================
# CONFIGURATION
# =========================================================
# Use raw string for Windows paths
CASIA_ROOT = r"C:\Users\tusha\Deep\Dataset\CASIA 2.0\CASIA2"
OUTPUT_FILE = "casia_filename_mapping.csv"

AU_DIR = os.path.join(CASIA_ROOT, "Au")
TP_DIR = os.path.join(CASIA_ROOT, "Tp")

# Regex: Matches Category (3 letters) + ID (5 digits) -> 'ani00123'
ID_PATTERN = re.compile(r'([a-z]{3})(\d{5})')

def get_source_candidate_name(tp_filename):
    """
    Parses 'Tp' filename to predict 'Au' filename.
    Returns: (Predicted_Au_Name_Base, Tamper_Type)
    """
    parts = tp_filename.split('_')
    matches = ID_PATTERN.findall(tp_filename)
    
    if len(parts) < 2 or len(matches) < 1:
        return None, "Unknown"
        
    tamper_type = parts[1] # 'S' or 'D'
    
    if tamper_type == 'D':
        # Splicing: 2nd ID is the Background/Source
        if len(matches) < 2: return None, "Splicing (No Source ID)"
        cat, num = matches[1]
    else:
        # Copy-Move: 1st ID is the Source
        cat, num = matches[0]
        
    # Format: Au_cat_num
    return f"Au_{cat}_{num}", tamper_type

def main():
    print("=" * 60)
    print("CASIA 2.0 CLEAN MAPPER (FILENAMES ONLY)")
    print("=" * 60)

    # 1. Index Authentic Filenames (Hash Map for Speed)
    print(f"[1/3] Indexing Authentic files...")
    if not os.path.exists(AU_DIR):
        print(f"ERROR: {AU_DIR} not found.")
        return

    au_map = {}
    # glob gives full paths, we strip them immediately
    for path in glob(os.path.join(AU_DIR, "*")):
        filename = os.path.basename(path)
        # Key: 'Au_ani_00023' (No extension) -> Value: 'Au_ani_00023.jpg'
        name_no_ext = os.path.splitext(filename)[0]
        au_map[name_no_ext] = filename

    print(f"      Indexed {len(au_map)} authentic images.")

    # 2. Map Tampered Files
    print(f"[2/3] Mapping Tampered images...")
    tp_files = glob(os.path.join(TP_DIR, "Tp*"))
    
    mapping_data = []
    found_count = 0

    for path in tqdm(tp_files, unit="img"):
        tp_filename = os.path.basename(path)
        
        # Predict the source name (without extension)
        candidate_key, tamper_type = get_source_candidate_name(tp_filename)
        
        # Check if it actually exists in our Au index
        if candidate_key and candidate_key in au_map:
            # Get the real filename with extension (e.g., .jpg or .tif)
            real_source_name = au_map[candidate_key]
            
            mapping_data.append({
                'Tampered_File': tp_filename,
                'Source_File': real_source_name,
                'Type': 'Splicing' if tamper_type == 'D' else 'Copy-Move'
            })
            found_count += 1
        else:
            # Optional: Log missing sources if you want, 
            # currently skipping to keep CSV clean with valid pairs only.
            pass

    # 3. Save to CSV
    print(f"\n[3/3] Saving valid pairs to '{OUTPUT_FILE}'...")
    
    headers = ['Tampered_File', 'Source_File', 'Type']
    
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(mapping_data)

    print("-" * 60)
    print(f"Total Pairs Mapped: {found_count}")
    print(f"File Saved:         {os.path.abspath(OUTPUT_FILE)}")
    print("-" * 60)

if __name__ == "__main__":
    main()