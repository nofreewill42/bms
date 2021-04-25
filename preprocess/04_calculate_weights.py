import numpy as np
import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    ds_path_str = Path('data_path.txt').read_text()
    ds_path = Path(ds_path_str)
    processed_labels_path = ds_path / 'train_labels_processed.csv'  # input
    label_weights_path = ds_path / 'train_labels_weights.csv'   # output

    # Do the job
    print(f'Reading processed labels - {processed_labels_path}')
    train_df = pd.read_csv(processed_labels_path, low_memory=False)
    train_df.fillna('', inplace=True)

    print('Complexity')
    complexity_df = train_df.iloc[:, 14].apply(lambda x: x.count('-') + x.count('-') * x.count('(') + x.count('('))

    print('Atom counts')
    atom_counts_df = train_df.iloc[:, 1:13].sum(axis=1)

    print('Atom (non)rarity')
    atom_present = (train_df.iloc[:, 1:13] > 0)
    atom_rarity_df = atom_present/(1+atom_present.sum(axis=0))
    atom_not_present = (~atom_present)
    atom_non_rarity_df = atom_not_present/(1+atom_not_present.sum(axis=0))
    atom_rarity_df = (atom_rarity_df + atom_non_rarity_df).sum(axis=1)

    print('Layer (non)rarity')
    layer_present = (train_df.iloc[:, 14:] != '')  # not  # ih,ib,it,im,is are not considered
    layer_rarity_df = layer_present/(1+layer_present.sum(axis=0))
    layer_not_present = (~layer_present)
    layer_non_rarity_df = layer_not_present/(1+layer_not_present.sum(axis=0))
    layer_rarity_df = (layer_rarity_df + layer_non_rarity_df).sum(axis=1)

    print(f'Saving to {label_weights_path}')
    weights_df = pd.concat([train_df['image_id'], complexity_df, atom_counts_df, atom_rarity_df, layer_rarity_df], axis=1)
    weights_df.columns = ['image_id', 'complexity', 'atom_count', 'atom_rarity', 'layer_rarity']
    weights_df.to_csv(label_weights_path, index=False)