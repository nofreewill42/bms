from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pathlib import Path

def normalize_inchi(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return inchi
    else:
        try: return Chem.MolToInchi(mol)
        except: return inchi


# Segfault in rdkit taken care of, run it with:
# while [ 1 ]; do python normalize_inchis.py && break; done
if __name__=='__main__':
    ds_path = Path('/media/nofreewill/Datasets_nvme/kaggle/bms-data/')
    submission_path = ds_path/'submission.csv'
    norm_path = submission_path.with_name(submission_path.stem+'_norm.csv')
    
    N = norm_path.read_text().count('\n') if norm_path.exists() else 0
    print(N)

    r = submission_path.open('r')
    w = norm_path.open('a', buffering=1)

    for _ in range(N):
        r.readline()
    line = r.readline()  # this line is the header or is where it died last time
    w.write(line)

    for line in tqdm(r):
        image_id = line.split(',')[0]
        inchi = ','.join(line[:-1].split(',')[1:]).replace('"','')
        inchi_norm = normalize_inchi(inchi)
        w.write(f'{image_id},"{inchi_norm}"\n')

    r.close()
    w.close()
