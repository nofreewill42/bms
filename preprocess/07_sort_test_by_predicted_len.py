import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sentencepiece as sp


if __name__ == '__main__':
    ds_path_str = Path('data_path.txt').read_text()
    ds_path = Path(ds_path_str)

    submission_path = ds_path/'samples/valid_labels.csv'#'submissions'/'submission_192_12.csv'  # input
    type = 'valid'
    test_img_stems_path = ds_path/'samples'/f'{type}_img_stems.pkl'                # input
    test_img_stems_sorted_path = ds_path/'samples'/f'{type}_img_stems_sorted.pkl'  # output

    bpe_num = 4096
    subwords_path = ds_path/'subwords'/f'bpe_{bpe_num}.model'
    swp = sp.SentencePieceProcessor(str(subwords_path))

    submission_df = pd.read_csv(submission_path)

    tqdm.pandas()
    bpe_len_df = submission_df.progress_apply(lambda x: len(swp.encode(x[1])), axis=1)
    sortit_by_len_df = pd.concat([submission_df['image_id'], bpe_len_df],axis=1)
    sorted_by_len_df = sortit_by_len_df.sort_values(0)['image_id']

    pickle.dump(sorted_by_len_df.to_list(), test_img_stems_sorted_path.open('wb'))

