import numpy as np
import cupy as cp
import nibabel as nib
import cupyx.scipy.ndimage as scipy_ndimage
import pandas as pd

def dice_gpu(im1, im2):
    """
    GPU-accelerated Dice score computation
    """
    im1 = cp.asarray(im1, dtype=cp.bool_)
    im2 = cp.asarray(im2, dtype=cp.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient on GPU
    intersection = cp.logical_and(im1, im2)
    dice_score = 2.0 * cp.sum(intersection) / (cp.sum(im1) + cp.sum(im2))
    
    return float(dice_score)

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

    # Get voxel spacing for volume calculations
    sx, sy, sz = pred_nii.header.get_zooms()
    
    # Extract specific tissue types on GPU
    pred_mat_gpu, gt_mat_gpu = get_TissueWiseSeg_gpu(
        prediction_matrix=pred_mat,
        gt_matrix=gt_mat,
        tissue_type=label_value
    )
    
    # Calculate full image Dice score
    if cp.all(gt_mat_gpu == 0) and cp.all(pred_mat_gpu == 0):
        full_dice = 1.0
    else:
        full_dice = dice_gpu(pred_mat_gpu, gt_mat_gpu)
    
    # Calculate ground truth volume
    full_gt_vol = float(cp.sum(gt_mat_gpu) * sx * sy * sz)
    
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

    # Initialize lists for lesion-wise analysis
    gt_tp_lesions = []
    tp_lesions = []
    fn_lesions = []
    metric_pairs = []

    # Get max component for iteration
    max_gt_comp = int(cp.max(gt_label_cc))

    # Analyze each ground truth lesion
    for gtcomp in range(1, max_gt_comp + 1):
        # Extract current ground truth lesion on GPU
        gt_tmp_gpu = cp.where(gt_label_cc == gtcomp, 1, 0).astype(cp.int32)
        
        # Calculate lesion volume
        gt_vol = float(cp.sum(gt_tmp_gpu) * sx * sy * sz)
        
        # Dilate current lesion for overlap detection
        if dil_factor > 0:
            gt_tmp_dilation_gpu = scipy_ndimage.binary_dilation(
                gt_tmp_gpu.astype(cp.bool_), structure=dilation_struct, iterations=dil_factor
            )
        else:
            gt_tmp_dilation_gpu = gt_tmp_gpu.astype(cp.bool_)
        
        # Find predicted lesions that overlap with dilated ground truth
        pred_tmp_gpu = cp.where(gt_tmp_dilation_gpu, pred_mat_cc, 0)
        intersecting_cc = cp.unique(pred_tmp_gpu)
        intersecting_cc = intersecting_cc[intersecting_cc > 0]  # Remove 0 value
        intersecting_cc = [int(x) for x in cp.asnumpy(intersecting_cc)]
        
        # Add intersecting lesions to true positives
        tp_lesions.extend(intersecting_cc)
        
        # Prepare mask of predicted lesions for metrics calculation
        if len(intersecting_cc) > 0:
            pred_tmp_gpu = cp.zeros_like(pred_mat_cc, dtype=cp.int32)
            for cc_val in intersecting_cc:
                pred_tmp_gpu = cp.where(pred_mat_cc == cc_val, 1, pred_tmp_gpu)
            gt_tp_lesions.append(gtcomp)
        else:
            pred_tmp_gpu = cp.zeros_like(gt_tmp_gpu, dtype=cp.int32)
            fn_lesions.append(gtcomp)
        
        # Calculate Dice score for this lesion on GPU
        dice_score = dice_gpu(pred_tmp_gpu, gt_tmp_gpu)
        
        # Store metrics for this lesion
        metric_pairs.append((intersecting_cc, gtcomp, gt_vol, dice_score))
    
    # Find false positive lesions (those not matched to any ground truth)
    all_pred_components = cp.unique(pred_mat_cc)
    all_pred_components = all_pred_components[all_pred_components > 0]  # Remove 0
    
    fp_lesions_gpu = []
    for comp in all_pred_components:
        comp_int = int(comp)
        if comp_int not in tp_lesions:
            fp_lesions_gpu.append(comp_int)
    
    return tp_lesions, fn_lesions, fp_lesions_gpu, gt_tp_lesions, metric_pairs, full_dice, full_gt_vol

def get_LesionWiseResults(pred_file, gt_file, output=None, 
                         override_dilation=None, override_threshold=None):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations with GPU acceleration
    """
    # Set default parameters
    dilation_factor = override_dilation if override_dilation is not None else 1
    lesion_volume_thresh = override_threshold if override_threshold is not None else 2
    
    label_values = ['WT', 'TC', 'ET']
    final_metrics_dict = {}
    
    # Process each tissue type
    for label in label_values:
        _, fn_lesions, fp_lesions, gt_tp_lesions, metric_pairs, full_dice, _ = get_LesionWiseScores(
            prediction_seg=pred_file,
            gt_seg=gt_file,
            label_value=label,
            dil_factor=dilation_factor
        )
        
        # Create and process DataFrame
        metric_df = pd.DataFrame(
            metric_pairs,
            columns=['predicted_lesion_numbers', 'gt_lesion_numbers', 'gt_lesion_vol', 'dice_lesionwise']
        )
        metric_df['_len'] = metric_df['predicted_lesion_numbers'].map(len)
        metric_df.replace([np.inf, -np.inf], 374, inplace=True)
        
        # Calculate lesion-wise Dice score
        large_lesions = metric_df[metric_df['gt_lesion_vol'] > lesion_volume_thresh]
        
        if len(large_lesions) + len(fp_lesions) > 0:
            lesion_wise_dice = large_lesions['dice_lesionwise'].sum() / (len(large_lesions) + len(fp_lesions))
        else:
            lesion_wise_dice = 1.0
            
        # Handle NaN
        if pd.isna(lesion_wise_dice):
            lesion_wise_dice = 1.0
        
        # Store results
        final_metrics_dict[label] = {
            'Legacy_Dice': full_dice,
            'LesionWise_Score_Dice': lesion_wise_dice,
        }
    
    # Create results DataFrame
    results_df = pd.DataFrame(final_metrics_dict).T.reset_index()
    results_df.columns = ['Labels'] + list(results_df.columns[1:])
    results_df.replace([np.inf, -np.inf], 374, inplace=True)
    
    # Save if requested
    if output:
        results_df.to_csv(output, index=False)
    
    return results_df