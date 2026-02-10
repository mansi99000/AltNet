import torch
from torch.profiler import profile
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def load_and_analyze_profile():
    # Load the latest profile
    log_dir = './profiler_logs'
    latest_file = max([os.path.join(log_dir, f) for f in os.listdir(log_dir)], key=os.path.getctime)
    
    # Load the event file
    ea = EventAccumulator(latest_file)
    ea.Reload()
    
    # Get profiler data
    df = pd.DataFrame({
        'Operation': [],
        'CPU Time (ms)': [],
        'CUDA Time (ms)': [],
        'Memory Used (MB)': []
    })
    
    # Print summary
    print("\n=== Performance Profile Summary ===")
    print("\nTop 10 Most Time-Consuming Operations:")
    print(df.nlargest(10, 'CUDA Time (ms)'))
    
    print("\nMemory Usage Summary:")
    print(df.nlargest(10, 'Memory Used (MB)'))
    
    # Save detailed results
    df.to_csv('profile_analysis.csv', index=False)
    print("\nDetailed results saved to 'profile_analysis.csv'")

if __name__ == "__main__":
    load_and_analyze_profile() 