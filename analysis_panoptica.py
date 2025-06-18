"""
Simplified BraTS-MET Dataset Evaluation Script with Panoptica
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from metrics_panoptica import get_LesionWiseResults

def evaluate_brats_predictions(
    style="normal",
    dataset_root="dataset",
    teams=None,
    matcher=None,
    threshold=None,
    output_dir="results",
    num_cases=None
):
    """Evaluate BraTS predictions and return formatted results."""
    
    if teams is None:
        teams = ["NVAUTO", "S_Y", "blackbean"]
    
    # Setup paths
    test_path = Path(dataset_root) / "ASNR-MICCAI-BraTS2023-MET-Challenge-TestingData"
    pred_path = Path(dataset_root) / "BratSMets_PredictedSegs" / "PredictedSegs"
    
    # Collect results
    team_scores = {team: {'ET': [], 'TC': [], 'WT': []} for team in teams}
    
    case_dirs = sorted(test_path.glob("BraTS-MET-*-000"))
    if num_cases is not None:
        case_dirs = case_dirs[:num_cases]
    
    for case_dir in tqdm(case_dirs):
        case_name = case_dir.name
        gt_file = case_dir / f"{case_name}-seg.nii.gz"
        
        for team in teams:
            pred_file = pred_path / team / f"{case_name}.nii.gz"
            
            if gt_file.exists() and pred_file.exists():
                results = get_LesionWiseResults(
                    style, str(pred_file), str(gt_file), matcher=matcher, threshold=threshold
                )
                
                for _, row in results.iterrows():
                    label = row['Labels']
                    if label in team_scores[team]:
                        team_scores[team][label].append(row['LesionWise_Score_Dice'])
    
    # Create summary
    summary = []
    for team in teams:
        row = {'Team': team}
        for metric in ['ET', 'TC', 'WT']:
            scores = team_scores[team][metric]
            if scores:
                row[f'{metric}_score'] = np.mean(scores)
                row[f'{metric}_std'] = np.std(scores)
                row[f'{metric}_rank'] = 0  # Will calculate after
            else:
                row[f'{metric}_score'] = 0.0
                row[f'{metric}_std'] = 0.0
                row[f'{metric}_rank'] = len(teams)
        summary.append(row)
    
    df = pd.DataFrame(summary)
    
    # Calculate ranks
    for metric in ['ET', 'TC', 'WT']:
        df[f'{metric}_rank'] = df[f'{metric}_score'].rank(ascending=False, method='min')
    
    # Sort by average rank
    df['avg_rank'] = df[['ET_rank', 'TC_rank', 'WT_rank']].mean(axis=1)
    df = df.sort_values('avg_rank').reset_index(drop=True)
    
    # Format for display
    display_df = pd.DataFrame()
    display_df['Team'] = df['Team']
    
    for metric in ['ET', 'TC', 'WT']:
        display_df[f'{metric} Score'] = df.apply(
            lambda row: f"{row[f'{metric}_score']:.2f} ± {row[f'{metric}_std']:.2f}",
            axis=1
        )
        display_df[f'{metric} Rank'] = df[f'{metric}_rank'].astype(int)
    
    # Save results
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        if style == 'part':
            df.to_csv(f"{output_dir}/results__part__match_{matcher}__matchthresh_{threshold}.csv", index=False)
        else:
            df.to_csv(f"{output_dir}/results__match_{matcher}__matchthresh_{threshold}.csv", index=False)

    return display_df


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="BraTS-MET Dataset Evaluation")
    parser.add_argument("--style", type=str, default="normal",
                        help="Evaluation style: 'normal' or 'part'")
    parser.add_argument("--matcher", type=str, default="naive",
                        help="matcher for evaluation")
    parser.add_argument("--threshold", type=float, default=0.000001,
                        help="Threshold value for evaluation")

    args = parser.parse_args()
    
    results = evaluate_brats_predictions(
        style=args.style,  # Pass the style parameter from command line
        matcher=args.matcher,
        threshold=args.threshold,
    )
    
    print("\nBRATS-MET EVALUATION RESULTS")
    print("=" * 80)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()