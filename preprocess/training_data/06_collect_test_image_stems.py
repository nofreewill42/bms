import pickle
from tqdm import tqdm
from pathlib import Path


if __name__ == '__main__':
    # collect images
    ds_path_str = Path('data_path.txt').read_text()[:-1]
    ds_path = Path(ds_path_str)
    tests_path = ds_path/'images'/'test'
    pkl_path = ds_path/'samples'/'test_img_stems.pkl'
    test_stems = []
    chars = list(map(str, range(10))) + list('abcdef')
    for i in tqdm(chars):
        for j in chars:
            for k in chars:
                dir_stems = [p.stem for p in (tests_path / i / j / k).iterdir()]
                test_stems += dir_stems
    pickle.dump(test_stems, pkl_path.open('wb'))
