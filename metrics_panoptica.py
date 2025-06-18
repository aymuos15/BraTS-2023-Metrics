import numpy as np
import nibabel as nib
import pandas as pd

from panoptica.panoptica_evaluator import Panoptica_Evaluator
from panoptica.instance_matcher import MaxBipartiteMatching, NaiveThresholdMatching
from panoptica.utils.processing_pair import InputType
from panoptica.utils.segmentation_class import SegmentationClassGroups
from panoptica.utils.label_group import LabelMergeGroup, LabelPartGroup

# Constants
MATCHING_THRESHOLD = 0.5
DEFAULT_MATCHER_STYLE = 'bipartite' # or naive

def get_LesionWiseScores(pred_file, gt_file, tissue_type, threshold=MATCHING_THRESHOLD, matcher='bipartite'):
    """Evaluate lesion-wise Dice score for a specific tissue type"""
    
    # Load data
    pred_data = nib.load(pred_file).get_fdata()
    gt_data = nib.load(gt_file).get_fdata()
    
    # Setup matcher
    if matcher == 'naive':
        instance_matcher = NaiveThresholdMatching(matching_threshold=threshold)
    else:
        instance_matcher = MaxBipartiteMatching(matching_threshold=threshold)
    
    # Evaluate
    evaluator = Panoptica_Evaluator(
        expected_input=InputType.UNMATCHED_INSTANCE,
        instance_matcher=instance_matcher,
        segmentation_class_groups=SegmentationClassGroups(
            {
                "NETC": (1, False),
                "ET": (3, False),
                "SNFH": (2, False),
                "TC": LabelMergeGroup([1, 3], False),
                "WT": LabelMergeGroup([1, 2, 3], False),
            }
        ),
    )

    result_dict = evaluator.evaluate(gt_data, pred_data, verbose=False)
    return result_dict

def get_PartLesionWiseScores(pred_file, gt_file, tissue_type, threshold=MATCHING_THRESHOLD, matcher='bipartite'):
    """Evaluate lesion-wise Dice score for a specific tissue type"""
    
    # Load data
    pred_data = nib.load(pred_file).get_fdata()
    gt_data = nib.load(gt_file).get_fdata()
    
    # Setup matcher
    if matcher == 'naive':
        instance_matcher = NaiveThresholdMatching(matching_threshold=threshold)
    elif matcher == 'bipartite':
        instance_matcher = MaxBipartiteMatching(matching_threshold=threshold)
    else:
        raise ValueError(f"Unknown matcher style: {matcher}")
    
    # Evaluate
    evaluator = Panoptica_Evaluator(
        expected_input=InputType.UNMATCHED_INSTANCE,
        instance_matcher=instance_matcher,
        segmentation_class_groups=SegmentationClassGroups(
            {
                "NETC": (1, False),
                "ET": (3, False),
                "SNFH": (2, False),
                "TC": LabelPartGroup([1], [3], False),
                "WT": LabelMergeGroup([1, 2, 3], False),
            }
        ),
    )

    result_dict = evaluator.evaluate(gt_data, pred_data, verbose=False)
    return result_dict

def get_LesionWiseResults(style, pred_file, gt_file, threshold=MATCHING_THRESHOLD, matcher='bipartite', output=None):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations with GPU acceleration
    """
    
    final_metrics_dict = {}

    if style == 'part':
        score_dict = get_PartLesionWiseScores(
            pred_file,
            gt_file,
            tissue_type=None,  # Assuming None means process all types
            matcher=matcher,
            threshold=threshold
        )
    else:
        score_dict = get_LesionWiseScores(
            pred_file,
            gt_file,
            tissue_type=None,  # Assuming None means process all types
            matcher=matcher,
            threshold=threshold
        )

    # Extract results for each label
    final_metrics_dict['WT'] = {
        'LesionWise_Score_Dice': score_dict['wt'].pq_dsc,
    }
    final_metrics_dict['TC'] = {
        'LesionWise_Score_Dice': score_dict['tc'].pq_dsc,
    }
    final_metrics_dict['ET'] = {
        'LesionWise_Score_Dice': score_dict['et'].pq_dsc,
    }

    # Create results DataFrame
    results_df = pd.DataFrame(final_metrics_dict).T.reset_index()
    results_df.columns = ['Labels'] + list(results_df.columns[1:])
    results_df.replace([np.inf, -np.inf], 374, inplace=True)
    
    # Save if requested
    if output:
        results_df.to_csv(output, index=False)
    
    return results_df