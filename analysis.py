import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from metrics import get_LesionWiseResults
# from metric_cupy import get_LesionWiseResults

def analyze_brats_predictions():
    """
    Iterate through BraTS-MET dataset segmentation files and their corresponding predictions
    """
    # Define paths
    dataset_root = "dataset"
    testing_data_path = os.path.join(dataset_root, "ASNR-MICCAI-BraTS2023-MET-Challenge-TestingData")
    predictions_path = os.path.join(dataset_root, "BratSMets_PredictedSegs", "PredictedSegs")
    
    # Get list of prediction teams/methods
    prediction_teams = [
        "CNMC_PMI2023",
        "MIA_SINTEF", 
        "NVAUTO",
        "S_Y",
        "blackbean",
        "i_sahajmistry"
    ]
    
    # Get all case directories
    case_dirs = glob.glob(os.path.join(testing_data_path, "BraTS-MET-*-000"))

    # # Taking the first 2 cases for testing
    # case_dirs = case_dirs[2:4]

    case_dirs.sort()
    
    print(f"Found {len(case_dirs)} cases")
    print(f"Found {len(prediction_teams)} prediction teams")
    
    # Initialize results storage
    all_results = []
    
    # Iterate through each case
    for case_dir in tqdm(case_dirs, desc="Processing cases", unit="case"):
        case_name = os.path.basename(case_dir)
        print(f"\nProcessing case: {case_name}")
        
        # Path to ground truth segmentation
        gt_seg_path = os.path.join(case_dir, f"{case_name}-seg.nii.gz")
        
        if not os.path.exists(gt_seg_path):
            print(f"  Warning: Ground truth segmentation not found: {gt_seg_path}")
            continue
            
        # Load ground truth segmentation
        try:
            gt_seg = nib.load(gt_seg_path)
            gt_data = gt_seg.get_fdata()
            print(f"  Ground truth shape: {gt_data.shape}")
            print(f"  Ground truth unique values: {np.unique(gt_data)}")
        except Exception as e:
            print(f"  Error loading ground truth: {e}")
            continue
        
        # Iterate through each prediction team
        for team in tqdm(prediction_teams, desc=f"  Teams for {case_name}", unit="team", leave=False):
            pred_path = os.path.join(predictions_path, team, f"{case_name}.nii.gz")
            
            if os.path.exists(pred_path):
                # Load prediction
                pred_seg = nib.load(pred_path)
                pred_data = pred_seg.get_fdata()
                
                print(f"    {team}: shape={pred_data.shape}, unique_values={np.unique(pred_data)}")
                
                # Run BraTS-MET evaluation
                results = get_LesionWiseResults(
                    pred_path, 
                    gt_seg_path, 
                    challenge_name="BraTS-MET"
                )
                print(f"      Lesion-wise results: {results}")
                
                # Store results for table generation
                result_row = {
                    'case': case_name,
                    'team': team,
                    'results': results
                }
                all_results.append(result_row)
                
            else:
                print(f"    {team}: Prediction file not found")
    
    # Create summary table
    create_summary_table(all_results, prediction_teams)

def create_summary_table(all_results, prediction_teams):
    """Create a summary table with team rankings similar to the provided image"""
    
    # Initialize data storage
    team_metrics = {team: {'ET': [], 'TC': [], 'WT': []} for team in prediction_teams}
    
    # Extract metrics from results
    for result in all_results:
        team = result['team']
        metrics_df = result['results']
        
        # Extract Dice scores from the DataFrame
        # The DataFrame has rows for WT, TC, ET with LesionWise_Score_Dice column
        for _, row in metrics_df.iterrows():
            label = row['Labels']
            dice_score = row['LesionWise_Score_Dice']
            
            if label in team_metrics[team]:
                team_metrics[team][label].append(dice_score)
    
    # Calculate mean, std, and median for each team and metric
    summary_data = []
    
    for team in prediction_teams:
        row = {'Team Name': team}
        
        for metric in ['ET', 'TC', 'WT']:
            scores = team_metrics[team][metric]
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                median_score = np.median(scores)
                row[f'{metric}_score'] = mean_score
                row[f'{metric}_std'] = std_score
                row[f'{metric}_median'] = median_score
            else:
                row[f'{metric}_score'] = 0.0
                row[f'{metric}_std'] = 0.0
                row[f'{metric}_median'] = 0.0
        
        summary_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Add rankings
    for metric in ['ET', 'TC', 'WT']:
        df[f'{metric}_rank'] = df[f'{metric}_score'].rank(ascending=False, method='min').astype(int)
    
    # Sort by average ranking
    avg_ranks = []
    for _, row in df.iterrows():
        avg_rank = (row['ET_rank'] + row['TC_rank'] + row['WT_rank']) / 3
        avg_ranks.append(avg_rank)
    
    df['avg_rank'] = avg_ranks
    df = df.sort_values('avg_rank').reset_index(drop=True)
    
    # Create formatted display table using pandas
    display_df = pd.DataFrame()
    display_df['Team Name'] = df['Team Name'].apply(lambda x: 
        'isahajmistry' if x == 'i_sahajmistry' else
        'MIASINTEF' if x == 'MIA_SINTEF' else
        'SY' if x == 'S_Y' else
        'CNMCPMI2023' if x == 'CNMC_PMI2023' else x)
    
    # Format columns for each metric with mean ± std (median)
    for metric in ['ET', 'TC', 'WT']:
        score_col = f'{metric}_score'
        std_col = f'{metric}_std'
        median_col = f'{metric}_median'
        rank_col = f'{metric}_rank'
        
        display_df[f'{metric} Dice score'] = df.apply(
            lambda row: f"{row[score_col]:.2f} ± {row[std_col]:.2f} ({row[median_col]:.2f})", axis=1)
        display_df[f'{metric} Rank'] = df[rank_col]
    
    print("\n" + "="*120)
    print("BRATS-MET EVALUATION RESULTS SUMMARY")
    print("="*120)
    
    # Display the table using pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(display_df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    # Run the analysis
    analyze_brats_predictions() #? No exceptions found, for all cases