import torch
from torch.utils.data import DataLoader
from data_loaders.test_loader import TestDS

import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sentencepiece as sp

from model_architecture.model import Model
from model_architecture.beam_search import BeamSearcher


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bpe_num = 4096
    max_len = 256
    bs = 8
    bw = 2#0
    torch.backends.cudnn.benchmark = True

    type = 'valid'#'test'#

    names = ['160_224_210', '160_224_210_192_192_32_192_192_36',
             '256_544_128', '256_544_128_224_640_24', '256_544_128_224_640_24_320_480_16',
             '288_512_112', '288_512_112_320_480_20', '288_512_112_320_480_20_480_320_4',
             '288_512_160', '288_512_160_256_576_20', '288_512_160_256_576_20_320_448_24',
             '384_384_120', '384_384_120_352_416_20', '384_384_120_352_416_20_192_768_4']
    img_sizes = [(160,224), (192,192),
                 (256,544), (224,640), (320,480),
                 (288,512), (320,480), (480,320),
                 (288,512), (256,576), (320,448),
                 (384,384), (352,416), (192,768),]
    weights = [1., 1.,
               1., 1., 1.,
               1., 1., 1.,
               1., 1., 1.,
               1., 1., 1.,]

    ds_path = Path("/home/nofreewill/Documents/kaggle/bms/bms-data/")
    imgs_path = ds_path / 'images' / ('train' if type=='valid' else 'test')
    test_stems = pickle.load((ds_path/'samples'/f'{type}_img_stems_sorted.pkl').open('rb'))[::-1]
    same_df = pd.read_csv(ds_path/'predictions'/type/'same_norm.csv')
    same_stems = set(same_df.iloc[:,0])
    test_stems = [stem for stem in test_stems if stem not in same_stems]
    print(len(test_stems)//bs)

    sp.SentencePieceProcessor()
    subwords_path = ds_path/'subwords'/f'bpe_{bpe_num}.model'
    swp = sp.SentencePieceProcessor(str(subwords_path))

    models = []
    test_dls = []
    ws = []
    for name, img_size in zip(names, img_sizes):
        print(name, img_size)

        if bw == 0:
            w_path = (ds_path/'predictions'/type/f'submission_{name}.csv')
            if w_path.exists():
                print('\tSKIP')
                continue
            ws.append(w_path)

        test_ds = TestDS(imgs_path, test_stems, img_size)
        test_dl = DataLoader(test_ds, batch_size=bs, pin_memory=True, num_workers=4)

        # model
        nhead = 8 if name in ('160_224_210', '160_224_210_192_192_32_192_192_36') else 16
        N, n, ff, first_k, first_s, last_s = 32, 128, 128, 3,2,1
        enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers =  512, nhead, 2048, 6
        dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers =  512, nhead, 2048, 6
        model = Model(bpe_num, N, n, ff, first_k, first_s, last_s,
                      enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers,
                      dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers,
                      max_trn_len=max_len).to(device)
        model.load_state_dict(torch.load(f'model_weights/done/model_{name}.pth', map_location=device))
        model.eval()

        models.append(model)
        test_dls.append(test_dl)

    if bw == 0:
        for k in range(len(models)):
            w = ws[k].open('w', buffering=1)
            m = models[k]
            test_dl = test_dls[k]
            w.write('image_id,InChI\n')

            for j, batch in enumerate(tqdm(test_dl)):
                imgs_tensor, img_stems, _ = batch
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

    else:
        w = (ds_path/'predictions'/type/f'submission_{bw}.csv').open('w', buffering=1)
        m = BeamSearcher(models, weights, bpe_num, bw)

        for j, batch in enumerate(tqdm(zip(*test_dls))):
            imgs_tensors = [b[0].to(device) for b in batch]
            img_stems = batch[0][1]
            ratios_tensors = torch.stack([b[2] for b in batch]).float().to(device)

            with torch.no_grad():
                lbl_ids, lens = m.predict(imgs_tensors, ratios_tensors, max_pred_len=256)
                lbl_ids = torch.stack(lbl_ids).T if not isinstance(lbl_ids, torch.Tensor) else lbl_ids
            if (lens == test_dls[0].dataset.max_len).sum() == len(lens):
                print('ERR')

            for i in range(len(lbl_ids)):
                pred = lbl_ids[i].tolist()[:lens[i]]
                pred_text = swp.decode(pred)

                w.write(f'{img_stems[i]},"InChI=1S/C{pred_text}"\n')