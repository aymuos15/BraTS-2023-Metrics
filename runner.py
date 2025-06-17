"""
BraTS-MET Evaluation Runner

This script runs the BraTS-MET evaluation with these parameter combinations:
1. Default (using default dilation and threshold from metric_cupy.py)
2. With dilation=0, using default threshold
3. Using default dilation, with threshold=0
4. With dilation=0 and threshold=0

Results for each configuration are saved to separate directories.

Note: Due to CuPy implementation limitations, dilation=0 may cause
NotImplementedError. If this occurs, you might need to modify metric_cupy.py
to handle the dilation=0 case differently.
"""

import time
from pathlib import Path

import pandas as pd
from analysis import analyze_brats_predictions

def run_all_combinations(base_output_dir: str = "results") -> None:
    """
    Run BraTS-MET evaluation with parameter combinations.
    
    Args:
        base_output_dir: Base directory for all outputs
    """
    # Create base output directory
    base_dir = Path(base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Define combinations with dilation=0 as requested
    combinations = [
        {
            "name": "default",
            "dilation": None,  # Use default from metric_cupy.py
            "threshold": None,  # Use default from metric_cupy.py
            "description": "Default (using defaults from metric_cupy.py)"
        },
        {
            "name": "dil_0_only",
            "dilation": 0,     # Using 0
            "threshold": None,  # Use default from metric_cupy.py
            "description": "With dilation=0, using default threshold"
        },
        {
            "name": "thresh_0_only",
            "dilation": None,  # Use default from metric_cupy.py
            "threshold": 0,     
            "description": "Using default dilation, with threshold=0"
        },
        {
            "name": "dil_0_thresh_0",
            "dilation": 0,     # Using 0
            "threshold": 0,     
            "description": "With dilation=0 and threshold=0"
        }
    ]
    
    # Run analysis for each combination
    results = []
    
    for combo in combinations:
        print("\n" + "=" * 80)
        print(f"Running: {combo['description']}")
        print("=" * 80)
        
        # Create specific output directory
        output_dir = base_dir / combo['name']
        output_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        # Run analysis with error handling
        try:
            display_df = analyze_brats_predictions(
                override_dilation=combo['dilation'],
                override_threshold=combo['threshold'],
                output_dir=str(output_dir)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Store successful results
            results.append({
                "name": combo['name'],
                "description": combo['description'],
                "duration": duration,
                "display_df": display_df,
                "status": "success"
            })
            
            # Display results
            print("\n" + "=" * 120)
            print(f"RESULTS: {combo['description']}")
            print(f"Duration: {duration:.2f} seconds")
            print("=" * 120)
            print(display_df.to_string(index=False))
            print("\n")
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            # Store error information
            error_msg = f"{type(e).__name__}: {str(e)}"
            results.append({
                "name": combo['name'],
                "description": combo['description'],
                "duration": duration,
                "error": error_msg,
                "status": "error"
            })
            
            # Display error
            print("\n" + "=" * 120)
            print(f"ERROR: {combo['description']}")
            print(f"Duration: {duration:.2f} seconds")
            print("=" * 120)
            print(f"An error occurred: {error_msg}")
            print("\n")
            
            # Write error to a file in the output directory
            error_file = output_dir / "error.txt"
            with open(error_file, "w") as f:
                f.write(f"Error running {combo['description']}:\n\n{error_msg}\n")
    
    # Save summary of all runs
    summary_data = []
    for r in results:
        summary_row = {
            "Configuration": r['description'],
            "Runtime (seconds)": r['duration'],
            "Output Directory": str(base_dir / r['name']),
            "Status": r['status']
        }
        
        if r['status'] == 'error':
            summary_row["Error"] = r['error']
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_path = base_dir / "run_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Display final summary
    print("\n" + "=" * 120)
    print("SUMMARY OF ALL RUNS")
    print("=" * 120)
    pd.set_option('display.max_colwidth', None)
    print(summary_df.to_string(index=False))
    print("\nResults have been saved to respective directories")


if __name__ == "__main__":
    run_all_combinations(base_output_dir="results")
