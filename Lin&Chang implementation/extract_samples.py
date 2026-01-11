import pandas as pd
import os
import shutil
from PIL import Image

# ================= CONFIGURATION =================
CASIA_ROOT = r"C:\Users\tusha\Deep\Dataset\CASIA 2.0\CASIA2"
AU_DIR = os.path.join(CASIA_ROOT, "Au")
TP_DIR = os.path.join(CASIA_ROOT, "Tp")
GT_DIR = r"C:\Users\tusha\Deep\Dataset\CASIA 2.0\casia2groundtruth\CASIA2.0_Groundtruth"
OUTPUT_DIR = "presentation_samples"

def find_file(base, search_dir, is_gt=False):
    exts = ['.jpg', '.tif', '.png', '.bmp', '.JPG', '.TIF']
    cands = [f"{base}_gt{e}" for e in exts] + [f"{base}{e}" for e in exts] if is_gt else [f"{base}{e}" for e in exts]
    for c in cands:
        p = os.path.join(search_dir, c)
        if os.path.exists(p): return p
    return None

def main():
    # 1. NUKE THE FOLDER (Clean Slate)
    if os.path.exists(OUTPUT_DIR):
        print(f"Deleting old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(OUTPUT_DIR)
    print(f"Created fresh {OUTPUT_DIR}...")
    
    # 2. Load Data
    try:
        df_diag = pd.read_csv('casia_diagnosis_report.csv')
        df_map = pd.read_csv('casia_filename_mapping.csv')
    except:
        print("Error: CSV files not found!")
        return

    merged = pd.merge(df_diag, df_map, left_on='Filename', right_on='Tampered_File', how='left')

    # 3. Select Samples (1 Failed + 9 Success)
    failed = merged[(merged['Diagnosis']=='Correct Detection') & (merged['Blocks_Detected']==0)].sample(1, random_state=42)
    success = merged[(merged['Diagnosis']=='Correct Detection') & (merged['Blocks_Detected']>0)].sample(9, random_state=42)
    selection = pd.concat([failed, success])

    count = 0
    for idx, row in selection.iterrows():
        count += 1
        print(f"Processing Set {count}...")
        
        # Find paths
        tp_src = find_file(os.path.splitext(row['Filename'])[0], TP_DIR)
        au_src = find_file(os.path.splitext(row['Source_File'])[0], AU_DIR) if pd.notna(row['Source_File']) else None
        gt_src = find_file(os.path.splitext(row['Filename'])[0], GT_DIR, is_gt=True)

        # 4. CONVERT AND SAVE AS JPG
        for tag, src in [('Tampered', tp_src), ('Authentic', au_src), ('GT', gt_src)]:
            if src:
                try:
                    img = Image.open(src).convert('RGB')
                    # Force filename to be .jpg
                    save_name = f"Set{count}_{tag}.jpg"
                    img.save(os.path.join(OUTPUT_DIR, save_name), "JPEG", quality=95)
                except Exception as e:
                    print(f"  Error converting {tag}: {e}")
            else:
                # Create a placeholder dummy image if missing (prevents Latex crash)
                print(f"  Missing {tag}, creating dummy...")
                img = Image.new('RGB', (200, 200), color = 'gray')
                img.save(os.path.join(OUTPUT_DIR, f"Set{count}_{tag}.jpg"), "JPEG")

    print("\nSUCCESS. The folder 'presentation_samples' now contains ONLY .jpg files.")

if __name__ == "__main__":
    main()