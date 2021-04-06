import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    ds_path_str = Path('data_path.txt').read_text()
    ds_path = Path(ds_path_str)
    train_labels_path = ds_path / 'train_labels.csv'     # input
    all_texts_path = ds_path/'subwords'/'all_texts.txt'  # output

    # Do the job
    all_texts_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Reading training labels - {train_labels_path}')
    train_df = pd.read_csv(train_labels_path)
    print(f'Writing InChI texts to file - {all_texts_path}')
    train_df.InChI.apply(lambda x: x[9:]).to_csv(all_texts_path, sep='\t', header=False, index=False)  # \t : not inside quotes -> subwords processor
    print('Done')
