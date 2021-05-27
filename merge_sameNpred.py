import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    ds_path = Path("/home/nofreewill/Documents/kaggle/bms/bms-data/")
    pred_df = pd.read_csv(ds_path/'predictions/valid/submission_2.csv')
    same_df = pd.read_csv(ds_path/'predictions/valid/same_norm.csv', header=None)
    same_df.columns = pred_df.columns
    submission_df = pd.concat([pred_df, same_df])
    submission_df.to_csv(ds_path/'predictions/valid/merged_submission.csv', index=False)