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
    ds_path = Path('/home/nofreewill/Documents/kaggle/bms/bms-data/')
    submissions_path = ds_path/'predictions/valid/raw'
    for p in submissions_path.iterdir():
        print(p.name)
        norm_path = ds_path/'predictions/valid/norm'/p.name

        N = norm_path.read_text().count('\n') if norm_path.exists() else 0
        print(N)

        r = p.open('r')
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
