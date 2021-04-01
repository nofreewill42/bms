import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def process_inchi(inchi_str):
    atoms = ['C', 'H', 'B', 'Br', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Si']
    columns = atoms+['c','h','b','t','m','s','i','ih','ib','it','im','is']
    processed_inchi = pd.Series(['']*len(columns), index=columns)

    inchi_parts = inchi_str.split('/')
    # Formulae
    formulae_str = inchi_parts[1]
    atom_numbers = num_atoms_from_inchi(formulae_str, atoms)
    processed_inchi.values[:len(atoms)] = atom_numbers
    processed_inchi.values[len(atoms):] = ''

    column_prefix = ''
    # c, h,b,t,m,s, i, ih,ib,it,im,is
    for inchi_part in inchi_parts[2:]:
        layer_type = inchi_part[0]
        processed_inchi[column_prefix+layer_type] = inchi_part[1:]
        if layer_type == 'i':
            column_prefix = 'i'

    return processed_inchi


def num_atoms_from_inchi(formulae_str, atoms):
    atomswithnumbers = re.sub(r'([A-Z][a-z]?)([A-Z])', lambda x: x.group(1) + '1' + x.group(2), formulae_str)
    atomswithnumbers = re.sub(r'([A-Z][a-z]?)([A-Z])', lambda x: x.group(1) + '1' + x.group(2), atomswithnumbers)
    atomswithnumbers = re.sub(r'([A-Z][a-z]?)$',       lambda x: x.group(1) + '1', atomswithnumbers)
    atoms_and_numbersstr = dict(re.findall(r'([A-Z][a-z]*)([0-9]+)', atomswithnumbers))
    atoms_numbers = [int(atoms_and_numbersstr[atom]) if atom in atoms_and_numbersstr else 0 for atom in atoms]
    return atoms_numbers


if __name__ == '__main__':
    ds_path_str = Path('data_path.txt').read_text()
    ds_path = Path(ds_path_str)
    train_labels_path = ds_path / 'train_labels.csv'                # input
    processed_labels_path = ds_path / 'train_labels_processed.csv'  # output

    # Do the job
    print(f'Reading training labels - {train_labels_path}')
    train_df = pd.read_csv(train_labels_path)

    print('Processing')
    tqdm.pandas()
    translated_series = train_df.progress_apply(lambda x: process_inchi(x[1]), axis=1)
    translated_df = pd.concat([train_df.image_id, translated_series], axis=1)

    print(f'Saving to {processed_labels_path}')
    translated_df.to_csv(processed_labels_path, index=False)
    print('Done')