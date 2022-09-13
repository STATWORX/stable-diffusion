import os
import glob
import pandas as pd

# Load all files and combine to dataframe
file_list = glob.glob(os.path.join(os.getcwd(), "results", "*.pkl"))
file_ls = [pd.read_pickle(file) for file in file_list]
results_df = pd.concat(file_ls)

results_df = results_df.pivot(index=['steps', 'dims'], columns='device', values='avg_time_step').reset_index()

# Speed up M1 vs. CPU
results_df['speed_up_mps_cpu'] = results_df['cpu'] / results_df['mps']

# Speed up GPU vs. CPU
results_df['speed_up_gpu_cpu'] = results_df['cpu'] / results_df['cuda']

# Speed up GPU vs. M1
results_df['speed_up_gpu_mps'] = results_df['mps'] / results_df['cuda']

print(results_df)
