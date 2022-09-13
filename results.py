import os
import glob
import pandas as pd

# Load all files
file_list = glob.glob(os.path.join(os.getcwd(), "results", "*.pkl"))

file_ls = [pd.read_pickle(file) for file in file_list]
