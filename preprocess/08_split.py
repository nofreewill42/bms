import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def row2inchi(row):
    inchi_str = 'InChI=1S/' + '/'.join(row[13:])
    inchi_str = re.sub(r'/+', '/', inchi_str)
    inchi_str = inchi_str[:-1] if inchi_str.endswith('/') else inchi_str
    return inchi_str


if __name__ == '__main__':
    ds_path_str = Path('data_path.txt').read_text()[:-1]
    ds_path = Path(ds_path_str)
    imgs_path = ds_path / 'images/train'
    df = pd.read_csv(ds_path / 'train_labels_processed.csv', low_memory=False)
    keep_ids = (df.C > 0) & df.ib.isna()
    df = df[keep_ids].fillna('')
    is_valid = np.array([i % 50 == 0 for i in range(len(df))])  # Train-Valid Split
    valid_df = df.iloc[is_valid]

    tqdm.pandas()
    inchis_df = valid_df.progress_apply(row2inchi, axis=1)
    validation_df = pd.concat([valid_df.image_id, inchis_df], axis=1)
    validation_df.columns = ['image_id', 'InChI']
    validation_df.to_csv(ds_path/'samples/valid_labels.csv', index=False)

    pickle.dump(validation_df.image_id.to_list(), (ds_path/'samples/valid_img_stems.pkl').open('wb'))