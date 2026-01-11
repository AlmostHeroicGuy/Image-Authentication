"""
Image Authentication Verification Script

This script evaluates the image authentication algorithm against a controlled dataset
where the ground truth is known (based on image index classification).

For each processed image:
  - Extract features from BOTH original and processed versions
  - Generate signature from original
  - Authenticate processed image against signature
  - Compare result with expected outcome based on index
  - Compute statistics (accuracy, FPR, FNR, etc.)
"""

import os
import json
import numpy as np
from pathlib import Path
from image_auth import ImageAuthenticator
from generate_dataset import PRIMES, MULTIPLES_OF_5


# Classification by index (ground truth)
def classify_image(idx: int) -> str:
    """
    Determine the true classification of an image based on its index.
    
    Args:
        idx: Image index number
        
    Returns:
        'compressed', 'identical', or 'tampered'
    """
    if idx in PRIMES:
        return 'compressed'
    elif idx in MULTIPLES_OF_5:
        return 'identical'
    else:
        return 'tampered'


def extract_index(filename: str) -> int:
    """
    Extract image index from filename format imgN_processed.jpg
    
    Args:
        filename: Filename like 'img5_processed.jpg'
        
    Returns:
        Index number N
    """
    # Parse "imgN_processed.jpg"
    parts = filename.replace('img', '').replace('_processed.jpg', '')
    return int(parts)


def verify_single_image(auth: ImageAuthenticator, 
                       idx: int,
                       original_path: str,
                       processed_path: str) -> dict:
    """
    Verify authentication for a single image pair.
    
    Args:
        auth: ImageAuthenticator instance
        idx: Image index
        original_path: Path to original image
        processed_path: Path to processed image
        
    Returns:
        Result dictionary with authentication details
    """
    try:
        # Step 1: Extract features from original image
        original_features = auth.extract_features(original_path)
        
        # Step 2: Generate signature from original
        signature_hex, signature_data = auth.generate_signature(original_features)
        
        # Step 3: Authenticate processed image against signature
        auth_result = auth.authenticate(processed_path, signature_data)
        
        # Step 4: Get ground truth
        ground_truth = classify_image(idx)
        
        # Step 5: Determine if result is correct
        # For identical and compressed: authenticated must be True
        # For tampered: authenticated must be False
        expected_authenticated = (ground_truth in ['identical', 'compressed'])
        actual_authenticated = auth_result['authenticated']
        is_correct = (expected_authenticated == actual_authenticated)
        
        return {
            'index': idx,
            'ground_truth': ground_truth,
            'authenticated': actual_authenticated,
            'expected': expected_authenticated,
            'correct': is_correct,
            'mismatch_rate': auth_result['mismatch_rate'],
            'tampering_confidence': auth_result['tampering_confidence'],
            'manipulated_pairs': len(auth_result['manipulated_pairs']),
            'reason': auth_result['reason'],
            'error': None
        }
    
    except Exception as e:
        return {
            'index': idx,
            'ground_truth': classify_image(idx),
            'authenticated': None,
            'expected': None,
            'correct': False,
            'mismatch_rate': None,
            'tampering_confidence': None,
            'manipulated_pairs': None,
            'reason': None,
            'error': str(e)
        }


def verify_dataset():
    """
    Main verification function: test all images in the dataset.
    
    Compares original images with processed versions and verifies
    that the authentication algorithm produces expected results.
    """
    input_folder = "input_images"
    processed_folder = "generated"
    
    # Initialize authenticator
    auth = ImageAuthenticator(
        block_size=8,
        num_thresholds=4,
        num_coefficients=10,
        base_power=7
    )
    
    print("\n" + "=" * 80)
    print("IMAGE AUTHENTICATION VERIFICATION")
    print("=" * 80)
    print(f"\nLoading images from:")
    print(f"  Original:  {input_folder}/")
    print(f"  Processed: {processed_folder}/")
    print()
    
    # Get list of processed images
    processed_files = sorted([f for f in os.listdir(processed_folder) 
                             if f.endswith('_processed.jpg')])
    
    if not processed_files:
        print("[ERROR] No processed images found in 'generated/' folder")
        return
    
    print(f"Found {len(processed_files)} processed images\n")
    print("=" * 80)
    print(f"{'Index':<8} {'Type':<12} {'Auth':<8} {'Expected':<10} {'Result':<10} {'Mismatches':<12}")
    print("=" * 80)
    
    # Verification results
    results = []
    
    for processed_fname in processed_files:
        # Extract index
        idx = extract_index(processed_fname)
        
        # Build paths
        original_path = os.path.join(input_folder, f"img{idx}.jpg")
        processed_path = os.path.join(processed_folder, processed_fname)
        
        # Verify this image pair
        result = verify_single_image(auth, idx, original_path, processed_path)
        results.append(result)
        
        # Print result
        if result['error']:
            status = "ERROR"
            mismatches_str = "N/A"
        else:
            status = "PASS" if result['correct'] else "FAIL"
            mismatches_str = f"{result['manipulated_pairs']}"
        
        auth_str = "T" if result['authenticated'] else "F"
        exp_str = "T" if result['expected'] else "F"
        
        print(f"[{result['index']:<6}] {result['ground_truth']:<12} {auth_str:<8} "
              f"{exp_str:<10} {status:<10} {mismatches_str:<12}")
    
    # Compute statistics
    print("=" * 80)
    
    successful_results = [r for r in results if r['error'] is None]
    correct_results = [r for r in successful_results if r['correct']]
    
    # Breakdown by category
    compressed_imgs = [r for r in successful_results if r['ground_truth'] == 'compressed']
    identical_imgs = [r for r in successful_results if r['ground_truth'] == 'identical']
    tampered_imgs = [r for r in successful_results if r['ground_truth'] == 'tampered']
    
    compressed_correct = len([r for r in compressed_imgs if r['correct']])
    identical_correct = len([r for r in identical_imgs if r['correct']])
    tampered_correct = len([r for r in tampered_imgs if r['correct']])
    
    print(f"\nSTATISTICS:")
    print(f"  Total images:      {len(successful_results)}")
    print(f"  Correct:           {len(correct_results)}/{len(successful_results)} ({100*len(correct_results)/len(successful_results):.1f}%)")
    print(f"  Errors:            {len(results) - len(successful_results)}")
    
    print(f"\nBY CATEGORY:")
    if compressed_imgs:
        print(f"  Compressed: {compressed_correct}/{len(compressed_imgs)} correct ({100*compressed_correct/len(compressed_imgs):.1f}%)")
    if identical_imgs:
        print(f"  Identical:  {identical_correct}/{len(identical_imgs)} correct ({100*identical_correct/len(identical_imgs):.1f}%)")
    if tampered_imgs:
        print(f"  Tampered:   {tampered_correct}/{len(tampered_imgs)} correct ({100*tampered_correct/len(tampered_imgs):.1f}%)")
    
    # Confusion matrix
    print(f"\nCONFUSION MATRIX:")
    tp = len([r for r in tampered_imgs if r['correct']])  # Correctly detected tampering
    fn = len(tampered_imgs) - tp  # Missed tampering
    fp = len([r for r in successful_results if r['correct'] == False and r['ground_truth'] != 'tampered'])  # False alarms
    tn = len([r for r in successful_results if r['correct'] and r['ground_truth'] != 'tampered'])  # Correct authentications
    
    print(f"  True Positives (tampered detected):    {tp}")
    print(f"  False Negatives (tampered missed):     {fn}")
    print(f"  False Positives (authentic rejected):  {fp}")
    print(f"  True Negatives (authentic accepted):   {tn}")
    
    if (tp + fn) > 0:
        print(f"\n  Tampering Detection Rate (Recall):     {100*tp/(tp+fn):.1f}%")
    if (tn + fp) > 0:
        print(f"  False Rejection Rate (FPR):            {100*fp/(tn+fp):.1f}%")
    
    # Save detailed results
    results_file = "verification_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = []
        for r in results:
            json_r = dict(r)
            if json_r['mismatch_rate'] is not None:
                json_r['mismatch_rate'] = float(json_r['mismatch_rate'])
            if json_r['tampering_confidence'] is not None:
                json_r['tampering_confidence'] = float(json_r['tampering_confidence'])
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    
    print(f"\n[SAVED] Detailed results in: {results_file}")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    verify_dataset()
