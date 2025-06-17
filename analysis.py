"""
BraTS-MET Dataset Evaluation Script

This module evaluates segmentation predictions against ground truth data
for the BraTS-MET challenge dataset.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure warnings
warnings.filterwarnings(
    "ignore", 
    message="CuPy may not function correctly because multiple CuPy packages are installed",
    category=UserWarning
)
warnings.filterwarnings("ignore")

from metric_cupy import get_LesionWiseResults


@dataclass
class EvaluationConfig:
    """Configuration for BraTS evaluation."""
    dataset_root: Path = Path("dataset")
    prediction_teams: List[str] = None
    override_dilation: Optional[int] = None
    override_threshold: Optional[int] = None
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        if self.prediction_teams is None:
            self.prediction_teams = [
                "NVAUTO",
                "S_Y",
                "blackbean",
                # "i_sahajmistry",
                # "MIA_SINTEF",
                # "CNMC_PMI2023",
            ]
    
    @property
    def testing_data_path(self) -> Path:
        """Path to testing data."""
        return self.dataset_root / "ASNR-MICCAI-BraTS2023-MET-Challenge-TestingData"
    
    @property
    def predictions_path(self) -> Path:
        """Path to predictions."""
        return self.dataset_root / "BratSMets_PredictedSegs" / "PredictedSegs"
    
    @property
    def param_description(self) -> str:
        """Parameter description for filenames."""
        if self.override_dilation is not None or self.override_threshold is not None:
            return f"dil_{self.override_dilation}_thresh_{self.override_threshold}"
        return "default"


class BratsEvaluator:
    """Evaluates BraTS-MET predictions against ground truth."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.config.testing_data_path.exists():
            raise FileNotFoundError(f"Testing data path not found: {self.config.testing_data_path}")
        if not self.config.predictions_path.exists():
            raise FileNotFoundError(f"Predictions path not found: {self.config.predictions_path}")
    
    def _get_case_directories(self) -> List[Path]:
        """Get sorted list of case directories."""
        pattern = "BraTS-MET-*-000"
        case_dirs = list(self.config.testing_data_path.glob(pattern))
        
        # TODO: Remove this testing limitation
        # case_dirs = case_dirs[1:3]  # Testing with specific cases
        
        return sorted(case_dirs)
    
    def _evaluate_case_team(self, case_dir: Path, team: str) -> Optional[Dict]:
        """Evaluate a single case for a specific team."""
        case_name = case_dir.name
        gt_seg_path = case_dir / f"{case_name}-seg.nii.gz"
        pred_path = self.config.predictions_path / team / f"{case_name}.nii.gz"
        
        if not gt_seg_path.exists() or not pred_path.exists():
            return None
        
        results = get_LesionWiseResults(
            str(pred_path),
            str(gt_seg_path),
            challenge_name="BraTS-MET",
            override_dilation=self.config.override_dilation,
            override_threshold=self.config.override_threshold
        )
        
        return {
            'case': case_name,
            'team': team,
            'results': results
        }
    
    def evaluate_all(self) -> pd.DataFrame:
        """Run evaluation for all cases and teams."""
        case_dirs = self._get_case_directories()
        all_results = []
        
        for case_dir in tqdm(case_dirs, desc="Processing cases"):
            for team in tqdm(self.config.prediction_teams, 
                           desc=f"Teams for {case_dir.name}", 
                           leave=False):
                result = self._evaluate_case_team(case_dir, team)
                if result is not None:
                    all_results.append(result)
        
        return self._create_summary_table(all_results)
    
    def _extract_team_metrics(self, all_results: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
        """Extract metrics by team and label."""
        team_metrics = {
            team: {'ET': [], 'TC': [], 'WT': []} 
            for team in self.config.prediction_teams
        }
        
        for result in all_results:
            team = result['team']
            metrics_df = result['results']
            
            for _, row in metrics_df.iterrows():
                label = row['Labels']
                dice_score = row['LesionWise_Score_Dice']
                
                if label in team_metrics[team]:
                    team_metrics[team][label].append(dice_score)
        
        return team_metrics
    
    def _calculate_team_statistics(self, team_metrics: Dict) -> List[Dict]:
        """Calculate statistics for each team."""
        summary_data = []
        
        for team in self.config.prediction_teams:
            row = {'Team Name': team}
            
            for metric in ['ET', 'TC', 'WT']:
                scores = team_metrics[team][metric]
                if scores:
                    row.update({
                        f'{metric}_score': np.mean(scores),
                        f'{metric}_std': np.std(scores),
                        f'{metric}_median': np.median(scores)
                    })
                else:
                    row.update({
                        f'{metric}_score': 0.0,
                        f'{metric}_std': 0.0,
                        f'{metric}_median': 0.0
                    })
            
            summary_data.append(row)
        
        return summary_data
    
    def _add_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ranking columns to the dataframe."""
        for metric in ['ET', 'TC', 'WT']:
            df[f'{metric}_rank'] = df[f'{metric}_score'].rank(
                ascending=False, method='min'
            ).astype(int)
        
        # Sort by average ranking
        df['avg_rank'] = df[['ET_rank', 'TC_rank', 'WT_rank']].mean(axis=1)
        return df.sort_values('avg_rank').reset_index(drop=True)
    
    def _create_summary_table(self, all_results: List[Dict]) -> pd.DataFrame:
        """Create summary table with team rankings."""
        team_metrics = self._extract_team_metrics(all_results)
        summary_data = self._calculate_team_statistics(team_metrics)
        
        df = pd.DataFrame(summary_data)
        df = self._add_rankings(df)
                
        if self.config.output_dir:
            self._save_results(df)
        
        return df
    
    def _format_display_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create formatted display table."""
        # Team name mappings
        name_mapping = {
            'i_sahajmistry': 'isahajmistry',
            'MIA_SINTEF': 'MIASINTEF',
            'S_Y': 'SY',
            'CNMC_PMI2023': 'CNMCPMI2023'
        }
        
        display_df = pd.DataFrame()
        display_df['Team Name'] = df['Team Name'].map(name_mapping).fillna(df['Team Name'])
        
        # Format metrics columns
        for metric in ['ET', 'TC', 'WT']:
            display_df[f'{metric} Dice score'] = df.apply(
                lambda row: (f"{row[f'{metric}_score']:.2f} ± {row[f'{metric}_std']:.2f} "
                           f"({row[f'{metric}_median']:.2f})"),
                axis=1
            )
            display_df[f'{metric} Rank'] = df[f'{metric}_rank']
        
        return display_df
    
    def _display_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format results as a DataFrame."""
        display_df = self._format_display_table(df)
        
        # Configure pandas display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        return display_df
    
    def _save_results(self, df: pd.DataFrame) -> None:
        """Save results to CSV file only."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save only raw data as CSV
        csv_path = self.config.output_dir / f"summary_{self.config.param_description}.csv"
        df.to_csv(csv_path, index=False)


def analyze_brats_predictions(
    override_dilation: Optional[int] = None,
    override_threshold: Optional[int] = None,
    output_dir: Optional[str] = "results"  # Default output directory
) -> pd.DataFrame:
    """
    Analyze BraTS predictions with specified parameters.
    
    Args:
        override_dilation: Override default dilation factor (0 to skip)
        override_threshold: Override default volume threshold (0 for all lesions)
        output_dir: Directory to save results
    
    Returns:
        Summary DataFrame with evaluation results
    """
    config = EvaluationConfig(
        override_dilation=override_dilation,
        override_threshold=override_threshold,
        output_dir=Path(output_dir) if output_dir else None
    )
    
    evaluator = BratsEvaluator(config)
    results_df = evaluator.evaluate_all()
    
    # Return formatted display table
    return evaluator._display_results(results_df)


if __name__ == "__main__":
    results = analyze_brats_predictions()
    # To display results in a notebook or console
    print("\n" + "=" * 120)
    print("BRATS-MET EVALUATION RESULTS SUMMARY")
    print("=" * 120)
    print(results.to_string(index=False))