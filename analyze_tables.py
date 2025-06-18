import pandas as pd
import glob
import os
from pathlib import Path

def load_all_csv_results():
   """Load all CSV files and return a list of dataframes"""
   csv_files = glob.glob("*.csv")
   results = []
   
   for csv_file in csv_files:
       df = pd.read_csv(csv_file)
       results.append({
           'df': df,
           'filename': csv_file
       })
   
   return results

def format_score_with_std(score, std):
   """Format score with standard deviation"""
   return f"{score:.2f} ± {std:.2f}"

def print_brats_table(df, filename):
   """Print formatted BRATS-MET evaluation table"""
   print(filename)
   print("=" * 80)
   
   # Check if this is a standard results file or a different format
   if 'Team' not in df.columns or 'ET_score' not in df.columns:
       # For non-standard files, just display the dataframe as is
       print(df.to_string(index=False))
       return
   
   # Header
   header = f"{'Team':>10} {'ET Score':>12} {'ET Rank':>8} {'TC Score':>12} {'TC Rank':>8} {'WT Score':>12} {'WT Rank':>8}"
   print(header)
   
   # Sort by average rank if available, otherwise don't sort
   if 'avg_rank' in df.columns:
       df_sorted = df.sort_values('avg_rank')
   else:
       # Calculate avg_rank if ET_rank, TC_rank, and WT_rank are available
       if all(col in df.columns for col in ['ET_rank', 'TC_rank', 'WT_rank']):
           df['avg_rank'] = df[['ET_rank', 'TC_rank', 'WT_rank']].mean(axis=1)
           df_sorted = df.sort_values('avg_rank')
       else:
           df_sorted = df  # No sorting if rank columns are missing
   
   for _, row in df_sorted.iterrows():
       et_score_str = format_score_with_std(row['ET_score'], row['ET_std'])
       tc_score_str = format_score_with_std(row['TC_score'], row['TC_std'])
       wt_score_str = format_score_with_std(row['WT_score'], row['WT_std'])
       
       print(f"{row['Team']:>10} {et_score_str:>12} {int(row['ET_rank']):>8} {tc_score_str:>12} {int(row['TC_rank']):>8} {wt_score_str:>12} {int(row['WT_rank']):>8}")

if __name__ == "__main__":
   # Change to results directory
   os.chdir('/home/localssk23/backup/soumya/BraTS-2023-Metrics/results')
   
   # Load and process all CSV files (excluding combined_results.csv)
   all_results = load_all_csv_results()
   
   # Print tables for each CSV
   for result in all_results:
       print_brats_table(result['df'], result['filename'])
       print()