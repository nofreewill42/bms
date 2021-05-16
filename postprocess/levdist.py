from pathlib import Path
ds_path = Path('/media/nofreewill/Datasets_nvme/kaggle/bms-data/')
submission_path = ds_path/'submission.csv'
norm_path = submission_path.with_name(submission_path.stem+'_norm.csv')
clean_path = norm_path.with_name(norm_path.stem+'_clean.csv')

# Input - START
sub1_path = submission_path
sub2_path = clean_path
# Input - END

import pandas as pd
sub1_df = pd.read_csv(sub1_path)
sub2_df = pd.read_csv(sub2_path)

import edlib
from tqdm import tqdm

lev = 0
N = len(sub1_df)
for i in tqdm(range(N)):
    inchi1, inchi2 = sub1_df.iloc[i,1], sub2_df.iloc[i,1]
    lev += edlib.align(inchi1, inchi2)['editDistance']

print(lev/N)
