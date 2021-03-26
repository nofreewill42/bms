import pickle
from tqdm import tqdm
from pathlib import Path
import sentencepiece as sp
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
cpus = multiprocessing.cpu_count()
cpus = min(48,cpus)

# save only those that are correct
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
import signal

from data_loaders.synthetic import synth_img


def iamge_from_inchi_str(partNidx):
    part_num, idx = partNidx
    if idx % 10000 == 0:
        print(part_num, idx)
    inchi_str = inchi_strs[idx]
    image_id = f'{part_num}{idx // 1000:04d}{idx % 1000:03d}'
    img_path = pubchem_imgs_dir_path / f'{part_num}/{idx // 1000:04d}/{idx % 1000:03d}.png'
    if img_path.exists():
        return
    img_pil = synth_img(inchi_str, 384)
    if img_pil is not None:
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_pil.save(img_path)
    return

if __name__=='__main__':
    ds_path_str = Path('data_path.txt').read_text()[:-1]
    ds_path = Path(ds_path_str)
    pubchem_dir_path = ds_path/'external/pubchem/'
    pubchem_texts_dir_path = pubchem_dir_path/'inchi_texts'
    pubchem_imgs_dir_path = pubchem_dir_path/'images'
    pubchem_imgs_dir_path.mkdir(parents=True,exist_ok=True)

    part_nums = [2]  # input
    inchi_strs = []
    partNidxs = []
    for part_num in part_nums:
        pubchem_part_text_path = pubchem_texts_dir_path/(str(part_num)+'.txt')
        inchi_strs = inchi_strs + pubchem_part_text_path.read_text().split('\n')
        partNidxs = partNidxs + list(zip([part_num]*len(inchi_strs), list(range(len(inchi_strs)))))
    with ProcessPoolExecutor(None) as e: e.map(iamge_from_inchi_str, partNidxs)

