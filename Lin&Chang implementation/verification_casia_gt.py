import pandas as pd
import shutil
import os
import glob

# ================= CONFIGURATION =================
# Paths provided by you
CASIA_ROOT = r"C:\Users\tusha\Deep\Dataset\CASIA 2.0\CASIA2"
AU_DIR = os.path.join(CASIA_ROOT, "Au")
TP_DIR = os.path.join(CASIA_ROOT, "Tp")
GT_DIR = r"C:\Users\tusha\Deep\Dataset\CASIA 2.0\casia2groundtruth\CASIA2.0_Groundtruth"

# Input/Output
CSV_FILE = 'casia_diagnosis_report_size_mismatch.csv' 
OUTPUT_DIR = 'presentation_samples'
# =================================================

def find_file(base_name, search_dir, is_gt=False):
    """
    Searches for a file in search_dir ignoring extension case or specific type.
    """
    # Possible extensions in CASIA
    extensions = ['.jpg', '.tif', '.png', '.bmp', '.JPG', '.TIF', '.PNG']
    
    # If it's a mask, it usually ends in _gt.png or _gt.bmp
    if is_gt:
        candidates = [f"{base_name}_gt{ext}" for ext in extensions]
        # Sometimes masks don't have _gt, just check rare case
        candidates += [f"{base_name}{ext}" for ext in extensions]
    else:
        candidates = [f"{base_name}{ext}" for ext in extensions]

    for fname in candidates:
        full_path = os.path.join(search_dir, fname)
        if os.path.exists(full_path):
            return full_path
            
    return None

def get_authentic_name_base(tp_filename):
    """
    Infers the base name of the authentic image from the tampered filename.
    CASIA Syntax: Tp_Type_Mod_Src_Tgt_ID.ext
    We need the Target (Tgt) part to find the Authentic source.
    """
    name_body = os.path.splitext(tp_filename)[0]
    parts = name_body.split('_')
    
    # Example: Tp_D_CRN_S_N_cha00063_art00014_11818
    # Parts: ['Tp', 'D', 'CRN', 'S', 'N', 'cha00063', 'art00014', '11818']
    # The target image is always the second to last segment before the final ID.
    
    target_part = parts[-2] # e.g., 'art00014'
    
    # Authentic files are named Au_cat_id (e.g., Au_art_00014)
    # Split 'art00014' -> 'art' and '00014'
    cat = target_part[:3]
    num_id = target_part[3:]
    
    return f"Au_{cat}_{num_id}"

# ================= MAIN EXECUTION =================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# 1. Load Data
print(f"Reading {CSV_FILE}...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Error: Could not find {CSV_FILE}. Make sure it is in the same folder.")
    exit()

# 2. Select Samples
# A. One Failed Case (Silent False Negative / Image Size Mismatch)
failed_pool = df[df['Diagnosis'] == 'Image Size Mismatch']
if failed_pool.empty:
    print("Warning: No 'Image Size Mismatch' found. Using generic MISSED.")
    failed_pool = df[df['Status'] == 'MISSED']

sample_failed = failed_pool.sample(1, random_state=42)

# B. Nine Successful Cases
success_pool = df[(df['Diagnosis'] == 'Correct Detection') & (df['Blocks_Detected'] > 0)]
sample_success = success_pool.sample(9, random_state=42)

selection = pd.concat([sample_failed, sample_success])

print(f"\nFound {len(selection)} samples. Starting extraction...\n")

# 3. Process and Copy
count = 0
for idx, row in selection.iterrows():
    count += 1
    tp_full_name = row['Filename']
    tp_base = os.path.splitext(tp_full_name)[0]
    diagnosis = row['Diagnosis']
    
    print(f"[{count}/10] Processing {tp_full_name} ({diagnosis})")
    
    # --- A. Find Tampered Image ---
    tp_path = find_file(tp_base, TP_DIR)
    if not tp_path:
        print(f"  [X] Tampered file not found in {TP_DIR}")
        continue
    
    # --- B. Find Authentic Image ---
    au_base = get_authentic_name_base(tp_full_name)
    au_path = find_file(au_base, AU_DIR)
    
    # --- C. Find Ground Truth Mask ---
    gt_path = find_file(tp_base, GT_DIR, is_gt=True)
    
    # --- D. Copy Files ---
    # Destination names (renaming slightly for clarity in presentation folder)
    prefix = f"Sample_{count}_"
    if diagnosis == 'Image Size Mismatch':
        prefix = "FAILED_Sample_"
        
    # Copy Tampered
    dst_tp = os.path.join(OUTPUT_DIR, prefix + "Tampered_" + os.path.basename(tp_path))
    shutil.copy2(tp_path, dst_tp)
    
    # Copy Authentic
    if au_path:
        dst_au = os.path.join(OUTPUT_DIR, prefix + "Authentic_" + os.path.basename(au_path))
        shutil.copy2(au_path, dst_au)
    else:
        print(f"  [!] Authentic source {au_base} not found.")

    # Copy GT
    if gt_path:
        # GT often has different extension, keep original extension
        dst_gt = os.path.join(OUTPUT_DIR, prefix + "GT_" + os.path.basename(gt_path))
        shutil.copy2(gt_path, dst_gt)
    else:
        print(f"  [!] Ground Truth mask not found for {tp_base}")

print(f"\nExtraction complete. Files are in: {os.path.abspath(OUTPUT_DIR)}")