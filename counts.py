import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore")
import cupyx.scipy.ndimage as scipy_ndimage
import cupy as cp

def get_TissueWiseSeg_gpu(prediction_matrix, gt_matrix, tissue_type):
    """
    Optimized GPU tissue-wise segmentation using lookup tables
    """
    prediction_matrix = cp.asarray(prediction_matrix, dtype=cp.int32)
    gt_matrix = cp.asarray(gt_matrix, dtype=cp.int32)
    
    # Precompute lookup tables for different tissue types
    tissue_maps = {
        'WT': cp.array([0, 1, 1, 1], dtype=cp.int32),  # 0->0, 1,2,3->1
        'TC': cp.array([0, 1, 0, 1], dtype=cp.int32),  # 0,2->0, 1,3->1  
        'ET': cp.array([0, 0, 0, 1], dtype=cp.int32)   # 0,1,2->0, 3->1
    }
    
    if tissue_type not in tissue_maps:
        raise ValueError(f"Unknown tissue_type: {tissue_type}")
    
    lookup = tissue_maps[tissue_type]
    
    # Clamp values to valid range and apply lookup
    pred_clamped = cp.clip(prediction_matrix, 0, 3)
    gt_clamped = cp.clip(gt_matrix, 0, 3)
    
    return lookup[pred_clamped], lookup[gt_clamped]

def get_GTseg_combinedByDilation_gpu(gt_dilated_cc_mat, gt_label_cc):
    """
    Optimized GPU-accelerated connected component combination
    """
    gt_dilated_cc_mat = cp.asarray(gt_dilated_cc_mat, dtype=cp.int32)
    gt_label_cc = cp.asarray(gt_label_cc, dtype=cp.int32)
    
    # Vectorized approach: where both matrices have non-zero values,
    # use the component ID from gt_dilated_cc_mat
    mask = (gt_dilated_cc_mat > 0) & (gt_label_cc > 0)
    result = cp.where(mask, gt_dilated_cc_mat, 0)
    
    return result

def get_LesionWiseScores(prediction_seg, gt_seg, label_value, dil_factor):
    """
    GPU-accelerated lesion-wise score computation
    """
    # Load segmentation files
    pred_nii = nib.load(prediction_seg)
    gt_nii = nib.load(gt_seg)
    pred_mat = pred_nii.get_fdata()
    gt_mat = gt_nii.get_fdata()
    
    # Extract specific tissue types on GPU
    pred_mat_gpu, gt_mat_gpu = get_TissueWiseSeg_gpu(
        prediction_matrix=pred_mat,
        gt_matrix=gt_mat,
        tissue_type=label_value
    )
    
    # Set up for connected component analysis
    dilation_struct = scipy_ndimage.generate_binary_structure(3, 2)
    
    # Ensure consistent dtypes for connected component analysis
    gt_mat_gpu = cp.asarray(gt_mat_gpu, dtype=cp.bool_)
    pred_mat_gpu = cp.asarray(pred_mat_gpu, dtype=cp.bool_)

    # Perform connected component analysis on GPU
    gt_mat_cc, _ = scipy_ndimage.label(gt_mat_gpu, structure=dilation_struct)
    pred_mat_cc, _ = scipy_ndimage.label(pred_mat_gpu, structure=dilation_struct)
    
    # Convert to consistent integer dtype
    gt_mat_cc = gt_mat_cc.astype(cp.int32)
    pred_mat_cc = pred_mat_cc.astype(cp.int32)

    # Dilate ground truth for overlap analysis
    if dil_factor > 0:
        gt_mat_dilation = scipy_ndimage.binary_dilation(
            gt_mat_gpu, structure=dilation_struct, iterations=dil_factor
        )
    else:
        gt_mat_dilation = gt_mat_gpu.copy()
        
    gt_mat_dilation_cc, _ = scipy_ndimage.label(gt_mat_dilation, structure=dilation_struct)
    gt_mat_dilation_cc = gt_mat_dilation_cc.astype(cp.int32)

    # Combine ground truth lesions based on dilation (on GPU)
    gt_label_cc = get_GTseg_combinedByDilation_gpu(
        gt_dilated_cc_mat=gt_mat_dilation_cc, 
        gt_label_cc=gt_mat_cc
    )

    num_cc_gt_label_cc = cp.unique(gt_label_cc).size - 1  # Exclude background label
    num_cc_pred_label_cc = cp.unique(pred_mat_cc).size - 1  # Exclude background label
    
    return num_cc_gt_label_cc, num_cc_pred_label_cc

def get_LesionWiseResults(pred_file, gt_file, output=None, 
                         override_dilation=None, override_threshold=None):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations with GPU acceleration
    """
    # Set default parameters
    dilation_factor = override_dilation if override_dilation is not None else 1
    
    label_values = ['WT', 'TC', 'ET']
    final_metrics_dict = {}
    
    # Process each tissue type
    for label in label_values:
        num_cc_gt, num_cc_pred = get_LesionWiseScores(
            prediction_seg=pred_file,
            gt_seg=gt_file,
            label_value=label,
            dil_factor=dilation_factor
        )
        
        # Store results
        final_metrics_dict[label] = {
            'num_cc_gt': int(num_cc_gt),
            'num_cc_pred': int(num_cc_pred)
        }

"""
Simplified BraTS-MET Dataset Evaluation Script
"""

def evaluate_brats_predictions(
    dataset_root="dataset",
    teams=None,
    dilation=None,
    threshold=None,
    output_dir="results",
    num_cases=None
):
    """Evaluate BraTS predictions and count lesions in GT and predictions."""
    
    if teams is None:
        teams = ["NVAUTO", "S_Y", "blackbean"]
    
    # Setup paths
    test_path = Path(dataset_root) / "ASNR-MICCAI-BraTS2023-MET-Challenge-TestingData"
    pred_path = Path(dataset_root) / "BratSMets_PredictedSegs" / "PredictedSegs"
    
    # Initialize data structures
    team_counts = {team: {'ET': {'gt': [], 'pred': []}, 
                          'TC': {'gt': [], 'pred': []}, 
                          'WT': {'gt': [], 'pred': []}} 
                   for team in teams}
    
    # Cache for ground truth counts
    gt_cache = {}  # {case_name: {label: count}}
    
    case_dirs = sorted(test_path.glob("BraTS-MET-*-000"))
    if num_cases is not None:
        case_dirs = case_dirs[:num_cases]
    
    # Process each case
    for case_dir in tqdm(case_dirs):
        case_name = case_dir.name
        gt_file = case_dir / f"{case_name}-seg.nii.gz"
        
        if not gt_file.exists():
            continue
            
        # Process ground truth only once per case
        gt_cache[case_name] = {}
        for label in ['ET', 'TC', 'WT']:
            # Calculate GT lesion count once
            pred_nii = nib.load(str(gt_file))  # Use GT as "prediction" to reuse function
            gt_nii = nib.load(str(gt_file))
            pred_mat = pred_nii.get_fdata()
            gt_mat = gt_nii.get_fdata()
            
            # Extract specific tissue types on GPU
            _, gt_mat_gpu = get_TissueWiseSeg_gpu(
                prediction_matrix=pred_mat,
                gt_matrix=gt_mat,
                tissue_type=label
            )
            
            # Set up for connected component analysis
            dilation_struct = scipy_ndimage.generate_binary_structure(3, 2)
            
            # Ensure consistent dtypes for connected component analysis
            gt_mat_gpu = cp.asarray(gt_mat_gpu, dtype=cp.bool_)
            
            # Perform connected component analysis on GPU
            gt_mat_cc, _ = scipy_ndimage.label(gt_mat_gpu, structure=dilation_struct)
            
            # Convert to consistent integer dtype
            gt_mat_cc = gt_mat_cc.astype(cp.int32)
            
            # Dilate ground truth for overlap analysis
            dil_factor = dilation if dilation is not None else 1
            if dil_factor > 0:
                gt_mat_dilation = scipy_ndimage.binary_dilation(
                    gt_mat_gpu, structure=dilation_struct, iterations=dil_factor
                )
            else:
                gt_mat_dilation = gt_mat_gpu.copy()
                
            gt_mat_dilation_cc, _ = scipy_ndimage.label(gt_mat_dilation, structure=dilation_struct)
            gt_mat_dilation_cc = gt_mat_dilation_cc.astype(cp.int32)
            
            # Combine ground truth lesions based on dilation (on GPU)
            gt_label_cc = get_GTseg_combinedByDilation_gpu(
                gt_dilated_cc_mat=gt_mat_dilation_cc, 
                gt_label_cc=gt_mat_cc
            )
            
            # Store ground truth lesion count
            num_cc_gt = cp.unique(gt_label_cc).size - 1  # Exclude background label
            gt_cache[case_name][label] = int(num_cc_gt)
        
        # Process predictions for each team
        for team in teams:
            pred_file = pred_path / team / f"{case_name}.nii.gz"
            
            if pred_file.exists():
                for label in ['ET', 'TC', 'WT']:
                    # Add GT count from cache
                    num_cc_gt = gt_cache[case_name][label]
                    team_counts[team][label]['gt'].append(num_cc_gt)
                    
                    # Calculate prediction lesion count
                    pred_nii = nib.load(str(pred_file))
                    pred_mat = pred_nii.get_fdata()
                    
                    # We only need to calculate the prediction segments
                    pred_mat_gpu, _ = get_TissueWiseSeg_gpu(
                        prediction_matrix=pred_mat,
                        gt_matrix=pred_mat,  # Not used for predictions
                        tissue_type=label
                    )
                    
                    # Set up for connected component analysis
                    dilation_struct = scipy_ndimage.generate_binary_structure(3, 2)
                    
                    # Process prediction
                    pred_mat_gpu = cp.asarray(pred_mat_gpu, dtype=cp.bool_)
                    pred_mat_cc, _ = scipy_ndimage.label(pred_mat_gpu, structure=dilation_struct)
                    
                    # Count prediction lesions
                    num_cc_pred = cp.unique(pred_mat_cc).size - 1  # Exclude background label
                    team_counts[team][label]['pred'].append(int(num_cc_pred))
    
    # Create summary
    summary = []
    for team in teams:
        row = {'Team': team}
        for metric in ['ET', 'TC', 'WT']:
            gt_counts = team_counts[team][metric]['gt']
            pred_counts = team_counts[team][metric]['pred']
            
            if gt_counts and pred_counts:
                row[f'{metric}_gt_total'] = np.sum(gt_counts)
                row[f'{metric}_gt_mean'] = np.mean(gt_counts)
                row[f'{metric}_gt_std'] = np.std(gt_counts)
                
                row[f'{metric}_pred_total'] = np.sum(pred_counts)
                row[f'{metric}_pred_mean'] = np.mean(pred_counts)
                row[f'{metric}_pred_std'] = np.std(pred_counts)
                
                # Calculate difference ratio (pred/gt)
                if row[f'{metric}_gt_total'] > 0:
                    row[f'{metric}_ratio'] = row[f'{metric}_pred_total'] / row[f'{metric}_gt_total']
                else:
                    row[f'{metric}_ratio'] = float('inf') if row[f'{metric}_pred_total'] > 0 else 1.0
            else:
                row[f'{metric}_gt_total'] = 0
                row[f'{metric}_gt_mean'] = 0.0
                row[f'{metric}_gt_std'] = 0.0
                row[f'{metric}_pred_total'] = 0
                row[f'{metric}_pred_mean'] = 0.0
                row[f'{metric}_pred_std'] = 0.0
                row[f'{metric}_ratio'] = 1.0
            
        summary.append(row)
    
    df = pd.DataFrame(summary)
    
    # Sort by team name for simplicity
    df = df.sort_values('Team').reset_index(drop=True)
    
    # Format for display
    display_df = pd.DataFrame()
    display_df['Team'] = df['Team']
    
    for metric in ['ET', 'TC', 'WT']:
        display_df[f'{metric} GT'] = df.apply(
            lambda row: f"{int(row[f'{metric}_gt_total'])} ({row[f'{metric}_gt_mean']:.2f} ± {row[f'{metric}_gt_std']:.2f})",
            axis=1
        )
        display_df[f'{metric} Pred'] = df.apply(
            lambda row: f"{int(row[f'{metric}_pred_total'])} ({row[f'{metric}_pred_mean']:.2f} ± {row[f'{metric}_pred_std']:.2f})",
            axis=1
        )
        display_df[f'{metric} Ratio'] = df[f'{metric}_ratio'].apply(lambda x: f"{x:.2f}")
    
    # Save results
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        df.to_csv(f"{output_dir}/lesion_counts__dil_{dilation}__thresh_{threshold}.csv", index=False)

    return display_df


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="BraTS-MET Dataset Lesion Counting")
    parser.add_argument("--dilation", type=int, default=1,
                        help="Dilation value for evaluation")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold value for evaluation")

    args = parser.parse_args()
    
    results = evaluate_brats_predictions(
        dilation=args.dilation,
        threshold=args.threshold
    )
    print("\nBRATS-MET LESION COUNT RESULTS")
    print("=" * 100)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()