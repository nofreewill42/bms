import os
import pickle
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import sentencepiece as sp
import multiprocessing
cpus = multiprocessing.cpu_count()
cpus = min(48,cpus)


if __name__=='__main__':
    ds_path_str = Path('data_path.txt').read_text()
    ds_path = Path(ds_path_str)
    pubchem_dir_path = ds_path/'external/pubchem/'
    pubchem_texts_dir_path = pubchem_dir_path/'inchi_texts'
    pubchem_imgs_dir_path = pubchem_dir_path/'images'
    pubchem_imgs_dir_path.mkdir(parents=True,exist_ok=True)

    pubchem_samples_dir_path = pubchem_dir_path/'samples'
    pubchem_samples_dir_path.mkdir(parents=True,exist_ok=True)
    bpe_num = 2**15

    # Do the job
    swp = sp.SentencePieceProcessor(str(ds_path/f'subwords/bpe_{bpe_num}.model'))

    sos = swp.bos_id()
    eos = swp.eos_id()
    part_nums = [0,1,2]  # input
    partNidxs = []
    for part_num in part_nums:
        print('part', part_num)
        pubchem_samples_path = pubchem_samples_dir_path/(str(part_num)+'.pkl')
        if pubchem_samples_path.exists():
            print('pickle exists, skipping')
            continue
        pubchem_part_text_path = pubchem_texts_dir_path/(str(part_num)+'.txt')
        inchi_strs = pubchem_part_text_path.read_text().split('\n')

        pubchem_samples = []
        for idx, inchi_str in enumerate(tqdm(inchi_strs)):
            c1,c2 = idx//1000, idx%1000
            image_id = f'{part_num}{c1:04d}{c2:03d}'
            img_path = pubchem_imgs_dir_path / f'{part_num}/{c1:04d}/{c2:03d}.png'
            if not img_path.exists():
                continue
            if os.stat(img_path).st_size == 0:
                print('0 size:', img_path)
                continue
            try:
                img_pil = Image.open(img_path)
                img_pil.verify()
            except:
                print('corrupted:', img_path)
                continue

            bpe_ids = [sos] + swp.encode(inchi_str[9:]) + [eos]
            sample = [image_id, bpe_ids]
            pubchem_samples.append(sample)

        pickle.dump(pubchem_samples, pubchem_samples_path.open('wb'))