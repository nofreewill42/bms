import re
from tqdm import tqdm
from pathlib import Path

def clean_inchi(inchi):
    # p/q
    inchi_clean = re.sub(r'/q.*?([/\n])', r'\1', inchi)
    inchi_clean = re.sub(r'/p.*?([/\n])', r'\1', inchi_clean)
    # ,...?
    inchi_clean = re.sub(r'(.*/[hbtms])?.*?\?,', r'\1', inchi_clean)
    inchi_clean = re.sub(r'(.*),.*?\?([,\n])', r'\1\2', inchi_clean)
    return inchi_clean


if __name__=='__main__':
    ds_path = Path('/media/nofreewill/Datasets_nvme/kaggle/bms-data/')
    submission_path = ds_path/'submission.csv'
    norm_path = submission_path.with_name(submission_path.stem+'_norm.csv')
    clean_path = norm_path.with_name(norm_path.stem+'_clean.csv')

    r = norm_path.open('r')
    w = clean_path.open('w', buffering=1)

    line = r.readline()  # this line is the header or is where it died last time
    w.write(line)

    for line in tqdm(r):
        image_id = line.split(',')[0]
        inchi_norm = ','.join(line[:-1].split(',')[1:]).replace('"','')
        inchi_clean = clean_inchi(inchi_norm)
        w.write(f'{image_id},"{inchi_clean}"\n')

    r.close()
    w.close()
