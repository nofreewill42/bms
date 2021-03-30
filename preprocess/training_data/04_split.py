import pickle
import numpy as np
from pathlib import Path


if __name__ == '__main__':
    ds_path_str = Path('data_path.txt').read_text()
    ds_path = Path(ds_path_str)
    samples_dir_path = ds_path/'samples'
    samples_path = samples_dir_path/'samples.pkl'

    valid_num = 20000
    test_num =  10000

    print('Reading samples')
    samples = pickle.load(samples_path.open('rb'))
    samples_num = len(samples)
    print(f'{samples_num} number of samples loaded')

    # sort samples by number of bpe_ids
    print('Sorting samples by number of bpe_ids')
    samples = sorted(samples, key=lambda x: len(x[1]))

    # valid
    print('Selecting validation samples')
    valid_freq = samples_num//valid_num
    valid_samples = samples[::valid_freq]
    samples = [sample for i, sample in enumerate(samples) if (i%valid_freq)!=0]
    print(f'{len(valid_samples)} number of samples selected as validation')

    # test
    print('Selecting test samples')
    samples_num = len(samples)
    test_freq = samples_num//test_num
    test_samples = samples[::test_freq]
    samples = [sample for i, sample in enumerate(samples) if (i%test_freq)!=0]
    print(f'{len(test_samples)} number of samples selected as test')

    # train
    train_samples = samples
    print(f'Remaining {len(train_samples)} number of samples is training data')

    # save samples
    print(f'Saving samples under {ds_path.absolute()}')
    pickle.dump(train_samples, (samples_dir_path/'train_samples.pkl').open('wb'))
    pickle.dump(valid_samples, (samples_dir_path/'valid_samples.pkl').open('wb'))
    pickle.dump(test_samples, (samples_dir_path/'test_samples.pkl').open('wb'))
