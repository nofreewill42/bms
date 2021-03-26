import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sentencepiece as sp


if __name__ == '__main__':
    ds_path_str = Path('data_path.txt').read_text()[:-1]
    ds_path = Path(ds_path_str)

    # INPUT - START
    # image_id, inchi_str
    train_labels_path = ds_path/'train_labels.csv'
    train_df = pd.read_csv(train_labels_path)
    # bpe_ids (create it now)
    bpe_num = 2**15
    subwords_path = ds_path/'subwords'/f'bpe_{bpe_num}.model'
    swp = sp.SentencePieceProcessor(str(subwords_path))
    # INPUT - END

    # OUTPUT - START
    samples_path = ds_path/'samples'/'samples.pkl'
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    # OUTPUT - END

    # Do the job
    samples = []
    sos, eos = swp.bos_id(), swp.eos_id()
    for i in tqdm(range(len(train_df))):
        image_id, inchi_str = train_df.iloc[i]
        bpe_ids = [sos] + swp.encode(inchi_str[9:]) + [eos]
        sample = [image_id, bpe_ids]
        samples.append(sample)

    pickle.dump(samples, samples_path.open('wb'))
