import numpy as np
import cupy as cp
import nibabel as nib
import cupyx.scipy.ndimage as scipy_ndimage
import pandas as pd
import math

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

def dice(im1, im2):
    """
    CPU-based Dice score for compatibility
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    return 2.0 * np.sum(intersection) / (np.sum(im1) + np.sum(im2))

def get_TissueWiseSeg_gpu(prediction_matrix, gt_matrix, tissue_type):
    """
    GPU-accelerated tissue-wise segmentation extraction
    """
    # Ensure inputs are CuPy arrays with consistent dtype
    prediction_matrix = cp.asarray(prediction_matrix, dtype=cp.int32)
    gt_matrix = cp.asarray(gt_matrix, dtype=cp.int32)

    if tissue_type == 'WT':
        # Whole tumor: labels 1, 2, 3
        pred_mask = (prediction_matrix == 1) | (prediction_matrix == 2) | (prediction_matrix == 3)
        prediction_matrix = cp.where(pred_mask, 1, 0)

        gt_mask = (gt_matrix == 1) | (gt_matrix == 2) | (gt_matrix == 3)
        gt_matrix = cp.where(gt_mask, 1, 0)

    elif tissue_type == 'TC':
        # Tumor core: labels 1, 3
        pred_mask = (prediction_matrix == 1) | (prediction_matrix == 3)
        prediction_matrix = cp.where(pred_mask, 1, 0)

        gt_mask = (gt_matrix == 1) | (gt_matrix == 3)
        gt_matrix = cp.where(gt_mask, 1, 0)

    elif tissue_type == 'ET':
        # Enhancing tumor: label 3 only
        pred_mask = (prediction_matrix == 3)
        prediction_matrix = cp.where(pred_mask, 1, 0)

        gt_mask = (gt_matrix == 3)
        gt_matrix = cp.where(gt_mask, 1, 0)

    return prediction_matrix, gt_matrix

def get_GTseg_combinedByDilation_gpu(gt_dilated_cc_mat, gt_label_cc):
    """
    GPU-accelerated connected component combination after dilation
    """
    # Ensure inputs are CuPy arrays with consistent dtype
    gt_dilated_cc_mat = cp.asarray(gt_dilated_cc_mat, dtype=cp.int32)
    gt_label_cc = cp.asarray(gt_label_cc, dtype=cp.int32)
    
    gt_seg_combinedByDilation_mat = cp.zeros_like(gt_dilated_cc_mat, dtype=cp.int32)
    max_comp = int(cp.max(gt_dilated_cc_mat))

    for comp in range(1, max_comp + 1):
        # Create mask for current component
        comp_mask = (gt_dilated_cc_mat == comp)
        gt_d_tmp = cp.where(comp_mask, gt_label_cc, 0)
        
        # Set all non-zero values to current component number
        gt_d_tmp = cp.where(gt_d_tmp > 0, comp, 0)
        gt_seg_combinedByDilation_mat += gt_d_tmp
        
    return gt_seg_combinedByDilation_mat

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

def get_LesionWiseResults(pred_file, gt_file, challenge_name, output=None, override_dilation=None, override_threshold=None):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations with GPU acceleration
    """
    # Set parameters based on challenge type
    challenge_params = {
        'BraTS-GLI': (3, 50),
        'BraTS-SSA': (3, 50),
        'BraTS-MEN': (1, 50),
        'BraTS-PED': (3, 50),
        'BraTS-MET': (1, 2)
    }
    
    # Get default parameters or use challenge-specific ones
    dilation_factor, lesion_volume_thresh = challenge_params.get(
        challenge_name, (3, 50)  # Default values if challenge not found
    )
    
    # Override parameters if provided
    if override_dilation is not None:
        dilation_factor = override_dilation
    
    if override_threshold is not None:
        lesion_volume_thresh = override_threshold

    final_metrics_dict = {}
    label_values = ['WT', 'TC', 'ET']
    all_lesion_metrics = []

    # Process each tissue type
    for label in label_values:
        tp_lesions, fn_lesions, fp_lesions, gt_tp_lesions, metric_pairs, full_dice, full_gt_vol = get_LesionWiseScores(
            prediction_seg=pred_file,
            gt_seg=gt_file,
            label_value=label,
            dil_factor=dilation_factor
        )
        
        # Create DataFrame with lesion metrics
        metric_df = pd.DataFrame(
            metric_pairs,
            columns=['predicted_lesion_numbers', 'gt_lesion_numbers', 'gt_lesion_vol', 'dice_lesionwise']
        ).sort_values(by=['gt_lesion_numbers'], ascending=True).reset_index(drop=True)
        
        # Mark number of predicted lesions per ground truth lesion
        metric_df['_len'] = metric_df['predicted_lesion_numbers'].map(len)
        
        # Add label information
        metric_df['Label'] = label
        
        # Replace infinity values
        metric_df.replace([np.inf, -np.inf], 374, inplace=True)
        
        # Filter metrics based on lesion volume threshold
        small_fn = metric_df[(metric_df['_len'] == 0) & 
                             (metric_df['gt_lesion_vol'] <= lesion_volume_thresh)].shape[0]
        small_tp = metric_df[(metric_df['_len'] != 0) & 
                             (metric_df['gt_lesion_vol'] <= lesion_volume_thresh)].shape[0]
        
        # Get metrics for lesions above threshold
        metric_df_thresh = metric_df[metric_df['gt_lesion_vol'] > lesion_volume_thresh]
        
        # Calculate lesion-wise Dice score
        try:
            if len(metric_df_thresh) + len(fp_lesions) > 0:
                lesion_wise_dice = np.sum(metric_df_thresh['dice_lesionwise']) / (len(metric_df_thresh) + len(fp_lesions))
            else:
                lesion_wise_dice = 1.0
        except:
            lesion_wise_dice = 1.0
        
        # Handle NaN values
        if math.isnan(lesion_wise_dice):
            lesion_wise_dice = 1.0
        
        # Store metrics for this tissue type
        final_metrics_dict[label] = {
            'Num_TP': len(gt_tp_lesions) - small_tp,
            'Num_FP': len(fp_lesions),
            'Num_FN': len(fn_lesions) - small_fn,
            'Legacy_Dice': full_dice,
            'LesionWise_Score_Dice': lesion_wise_dice,
        }
        
        # Collect all lesion metrics
        all_lesion_metrics.append(metric_df)
    
    # Create final results DataFrame
    results_df = pd.DataFrame(final_metrics_dict).T
    results_df['Labels'] = results_df.index
    results_df = results_df.reset_index(drop=True)
    results_df = results_df[['Labels'] + [col for col in results_df.columns if col != 'Labels']]
    results_df.replace([np.inf, -np.inf], 374, inplace=True)
    
    # Save results if output path is provided
    if output:
        results_df.to_csv(output, index=False)
    
    return results_df