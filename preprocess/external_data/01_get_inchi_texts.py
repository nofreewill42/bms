import pickle
from tqdm import tqdm
from pathlib import Path
import sentencepiece as sp


if __name__=='__main__':
    ds_path_str = Path('data_path.txt').read_text()
    ds_path = Path(ds_path_str)
    trn_samples_path = ds_path/'samples/train_samples.pkl'
    pubchem_dir_path = ds_path/'external/pubchem/'
    pubchem_texts_dir_path = pubchem_dir_path/'inchi_texts'
    if pubchem_texts_dir_path.exists():
        print(f'{pubchem_texts_dir_path} directory already exists. Delete that to proceed.')
        import sys; sys.exit()
    pubchem_texts_dir_path.mkdir(parents=True,exist_ok=True)

    allowed_max_len = 128
    bpe_num = 2**15
    N = 60  #-> 1,2,3,4,5,6 external packs per epoch can be chosen

    # How many samples in training set
    print('Counting training samples...')
    trn_samples = pickle.load(trn_samples_path.open('rb'))
    trn_samples_num = len(trn_samples)
    print(f'{trn_samples_num} number of training samples...')
    del trn_samples

    # How many samples in pubchem
    print('Counting inchis in PubChem...')
    pubchem_inchis_path = pubchem_dir_path/'CID-InChI-Key'
    with open(str(pubchem_inchis_path)) as r:
        pubchem_inchis_num = sum([1 for line in tqdm(r)])
    print(f'{pubchem_inchis_num} number of inchis in PubChem...')

    # Do the job
    print(f'Writing {pubchem_inchis_num} number of inchis into {N} number of txt files under {pubchem_texts_dir_path}')
    swp = sp.SentencePieceProcessor(str(ds_path/f'subwords/bpe_{bpe_num}.model'))

    r = open(str(pubchem_inchis_path))
    ws = [(pubchem_texts_dir_path/f'{i}.txt').open('w') for i in range(N)]
    w_pointer = 0
    too_long_drops_num = 0

    for _ in tqdm(range(pubchem_inchis_num)):
        line = r.readline()
        if not line:
            break
        
        inchi_str = line.split('\t')[1]
        
        # 0: <\unk> ; len-1: <\eos> at target end
        bpe_ids = swp.encode(inchi_str[9:])  # [:9] -> InChI=1S/
        if (0 in bpe_ids) or len(bpe_ids)==0 or (len(bpe_ids) > allowed_max_len-1):
            if (len(bpe_ids) > allowed_max_len-1):
                too_long_drops_num += 1
            continue
        
        w_pointer %= N
        ws[w_pointer].write(inchi_str+'\n')
        w_pointer += 1

    for w in ws:
        w.close()
    
    print(f'{too_long_drops_num} number of too long inchis was dropped')
    print('Done')
    r.close()
