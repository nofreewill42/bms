import edlib
import pandas as pd
from tqdm import tqdm
from pathlib import Path


if __name__ == '__main__':
    ds_path = Path('/home/nofreewill/Documents/kaggle/bms/bms-data/')
    preds_path = ds_path/'predictions'
    type = 'valid/norm'
    dfs_path = preds_path/type

    pred_path = dfs_path.parent/'submission_2.csv'#f'same_{type.split("/")[1]}.csv'#dfs_path/'submission_160_224_210.csv'#
    pred_df = pd.read_csv(pred_path,header=None)
    gt_df = pd.read_csv(ds_path/'samples/valid_labels.csv')
    pred_df.columns = gt_df.columns

    df1 = pred_df
    df2 = gt_df

    tqdm.pandas()
    merge_df = df2.merge(df1, on='image_id')
    ld_df = merge_df.progress_apply(lambda x: edlib.align(x[1], x[2])['editDistance'], axis=1)
    print(len(ld_df), sum(ld_df)/len(ld_df))