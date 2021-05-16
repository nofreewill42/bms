import torch
from torch.utils.data import DataLoader
from data_loaders.test_loader import TestDS

import kornia as K

import pickle
from tqdm import tqdm
from pathlib import Path
import sentencepiece as sp

from model_architecture.model import Model
from model_architecture.beam_search import BeamSearcher


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img_size = (288,512)
    bpe_num = 4096#2**15
    max_len = 256
    bs = 64
    torch.backends.cudnn.benchmark = True

    ds_path = Path("/media/nofreewill/Datasets_nvme/kaggle/bms-data/")

    imgs_path = ds_path / 'images/test'
    test_stems = pickle.load((ds_path/'samples/test_img_stems_sorted.pkl').open('rb'))[::-1]

    sp.SentencePieceProcessor()
    subwords_path = ds_path/'subwords'/f'bpe_{bpe_num}.model'
    swp = sp.SentencePieceProcessor(str(subwords_path))

    test_ds = TestDS(imgs_path, test_stems, img_size)
    test_dl = DataLoader(test_ds, batch_size=bs, pin_memory=True, num_workers=4)

    # model
    N, n, ff, first_k, first_s, last_s = 32, 128, 128, 3,2,1
    enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers =  512, 8, 2048, 6
    dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers =  512, 8, 2048, 6
    model = Model(bpe_num, N, n, ff, first_k, first_s, last_s,
                  enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers,
                  dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers,
                  max_trn_len=max_len, tta=None).to(device)#114, tta=tfms2).to(device)
    model.load_state_dict(torch.load(f'/media/nofreewill/Datasets_nvme/kaggle/bms-code/model_weights/done/model_288_512.pth', map_location=device))
    model.eval()

    models = [model]*1
    weights = [1.]*1
    bw = 0

    from valid_utils import levenshtein, validate
    w = (ds_path/'submission.csv').open('w')
    m = model if (bw == 0) else BeamSearcher(models, weights, bpe_num, bw)

    for j, batch in enumerate(tqdm(test_dl)):
        imgs_tensor, img_stems = batch
        imgs_tensor = imgs_tensor.to(device)

        with torch.no_grad():
            lbl_ids, lens = m.predict(imgs_tensor)
            lbl_ids = torch.stack(lbl_ids).T if not isinstance(lbl_ids, torch.Tensor) else lbl_ids

        for i in range(len(lbl_ids)):
            pred = lbl_ids[i].tolist()[:lens[i]]
            pred_text = swp.decode(pred)

            if w is not None:
                w.write(f'{img_stems[i]},"InChI=1S/C{pred_text}"\n')
    w.close()

