import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    ds_path = Path('/home/nofreewill/Documents/kaggle/bms/bms-data/')
    preds_path = ds_path/'predictions'
    type = 'valid/norm'

    dfs_path = preds_path/type
    df_paths = list(dfs_path.iterdir())

    dfs = [pd.read_csv(df_path, header=None) for df_path in df_paths]
    same_preds = (dfs[0].iloc[:,1] == dfs[1].iloc[:,1])
    for i in range(2,len(dfs)):
        same_preds = (same_preds & (dfs[0].iloc[:,1] == dfs[i].iloc[:,1]))

    same_df = dfs[0][same_preds]
    same_df.to_csv(dfs_path.parent/f'same_{type.split("/")[1]}.csv',header=False,index=False)