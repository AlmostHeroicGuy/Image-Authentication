"""
CASIA 2.0 Parallelized Verification Script (Full Logging)
Features:
1. Multiprocessing (High Speed)
2. Hash Map Indexing (Instant Lookup)
3. Full CSV Logging (Saves every single result)
"""

import os
import re
import time
import csv
import multiprocessing
from glob import glob
from tqdm import tqdm
from image_auth import ImageAuthenticator

# =========================================================
# CONFIGURATION
# =========================================================
# USE RAW STRINGS (r"...") TO AVOID UNICODE ERRORS
CASIA_ROOT = r"C:\Users\tusha\Deep\Dataset\CASIA 2.0\CASIA2"  
AU_DIR = os.path.join(CASIA_ROOT, "Au")
TP_DIR = os.path.join(CASIA_ROOT, "Tp")

# Set to None to process EVERYTHING
MAX_TEST_SAMPLES = None 

# Number of CPU processes (Default: All available - 1)
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)

# Output log file
LOG_FILE = "casia_full_report.csv"

# =========================================================
# WORKER FUNCTIONS
# =========================================================

def get_authentic_candidate_name(tp_filename):
    """
    Decodes 'Tp_D_..._sec10107_sec10101_...' -> 'Au_sec_10101'
    """
    id_pattern = re.compile(r'([a-z]{3})(\d{5})')
    parts = tp_filename.split('_')
    
    if len(parts) < 2: return None
    matches = id_pattern.findall(tp_filename)
    if len(matches) < 2: return None
    
    # Splicing (D) -> 2nd ID is Background
    # Copy-Move (S) -> 1st/2nd ID are usually same
    if parts[1] == 'D':
        bg_cat, bg_id = matches[1]
    else:
        bg_cat, bg_id = matches[0]
        
    return f"Au_{bg_cat}_{bg_id}"

def verify_single_pair_task(args):
    """
    Worker function run by each CPU core.
    """
    original_path, test_path = args
    fname = os.path.basename(test_path)
    
    # Re-instantiate Authenticator (Isolates memory per process)
    auth = ImageAuthenticator(
        block_size=8, 
        num_thresholds=4, 
        num_coefficients=10, 
        base_power=7
    )

    try:
        # 1. Generate Signature from Original
        original_features = auth.extract_features(original_path)
        _, signature_data = auth.generate_signature(original_features)
        
        # 2. Authenticate Test Image
        # We expect FALSE for Tampered images
        result = auth.authenticate(test_path, signature_data)
        
        # 3. Determine Outcome
        # Since we are ONLY testing Tampered images in this script:
        # Authentic=False means "Correct Detection"
        # Authentic=True means "False Negative" (Missed it)
        is_tampered = not result['authenticated'] 
        
        return {
            'Filename': fname,
            'Status': 'DETECTED' if is_tampered else 'MISSED',
            'Is_Correct': is_tampered,
            'Mismatch_Rate': f"{result.get('mismatch_rate', 0):.4f}",
            'Blocks_Affected': len(result.get('manipulated_pairs', [])),
            'Confidence': f"{result.get('tampering_confidence', 0):.4f}",
            'Error': 'None'
        }
    except Exception as e:
        return {
            'Filename': fname,
            'Status': 'ERROR',
            'Is_Correct': False,
            'Mismatch_Rate': '0.0000',
            'Blocks_Affected': 0,
            'Confidence': '0.0000',
            'Error': str(e)
        }

# =========================================================
# MAIN EXECUTION
# =========================================================

def run_evaluation_parallel():
    print("=" * 70)
    print(f"CASIA 2.0 FULL LOGGING RUN ({NUM_CORES} Cores)")
    print("=" * 70)

    # 1. INDEXING
    print("[1/4] Indexing Authentic files...")
    au_files_map = {}
    all_au = glob(os.path.join(AU_DIR, "*"))
    for f_path in all_au:
        f_name = os.path.basename(f_path)
        name_no_ext = os.path.splitext(f_name)[0]
        au_files_map[name_no_ext] = f_path
    print(f"      Indexed {len(au_files_map)} authentic images.")

    # 2. MATCHING PAIRS
    print("[2/4] Matching Tampered images to Originals...")
    tp_files = sorted(glob(os.path.join(TP_DIR, "Tp*")))
    valid_pairs = []
    
    for tp in tp_files:
        if MAX_TEST_SAMPLES and len(valid_pairs) >= MAX_TEST_SAMPLES: break
        candidate_name = get_authentic_candidate_name(os.path.basename(tp))
        if candidate_name and candidate_name in au_files_map:
            src_path = au_files_map[candidate_name]
            valid_pairs.append((src_path, tp))
            
    print(f"      Found {len(valid_pairs)} valid pairs to process.")
    
    if not valid_pairs:
        print("[ERROR] No pairs found.")
        return

    # 3. PARALLEL PROCESSING
    print(f"[3/4] Processing on {NUM_CORES} CPU cores...")
    results = []
    start_time = time.time()
    
    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        for res in tqdm(pool.imap_unordered(verify_single_pair_task, valid_pairs), 
                       total=len(valid_pairs), 
                       unit="img"):
            results.append(res)

    duration = time.time() - start_time

    # 4. WRITING LOGS
    print(f"[4/4] Writing full logs to '{LOG_FILE}'...")
    
    # Sort results by filename for easier reading
    results.sort(key=lambda x: x['Filename'])
    
    fieldnames = ['Filename', 'Status', 'Is_Correct', 'Mismatch_Rate', 'Blocks_Affected', 'Confidence', 'Error']
    
    with open(LOG_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 5. FINAL SUMMARY
    correct_count = sum(1 for r in results if r['Is_Correct'])
    accuracy = (correct_count / len(results)) * 100 if results else 0
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total Images:    {len(results)}")
    print(f"Time Taken:      {duration:.2f}s")
    print(f"Accuracy:        {accuracy:.2f}%")
    print(f"Full Log:        {os.path.abspath(LOG_FILE)}")
    print("=" * 70)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if not os.path.exists(AU_DIR):
        print(f"Critical Error: {AU_DIR} does not exist.")
    else:
        run_evaluation_parallel()