import torch
from torchvision import transforms as T
import kornia as K
from kornia import augmentation as KA

import numpy as np
import pandas as pd
from pathlib import Path
import sentencepiece as sp

from model_architecture.model import Model
from model_architecture.beam_search import BeamSearcher


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img_size = 160
    bpe_num = 4096
    max_len = 256
    bs = 64
    torch.backends.cudnn.benchmark = True

    import pickle
    from torch.utils.data import DataLoader
    from data_loaders.data_loader import DS, SplitterSampler
    ds_path = Path("/home/nofreewill/Documents/kaggle/bms/bms-data")

    swp = sp.SentencePieceProcessor(f'{ds_path}/subwords/bpe_{bpe_num}.model')
    imgs_path = ds_path / 'images/train'
    df = pd.read_csv(ds_path/'train_labels_processed.csv', low_memory=False)
    keep_ids = (df.C > 0) & df.ib.isna()
    df = df[keep_ids].fillna('')
    is_valid = np.array([i%50==0 for i in range(len(df))])
    valid_df = df.iloc[is_valid]

    # model_288_512
    N, n, ff, first_k, first_s, last_s = 32, 128, 128, 3,2,1
    enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers =  512, 16, 512*4, 6
    dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers =  512, 16, 512*4, 6
    model_288_512 = Model(bpe_num, N, n, ff, first_k, first_s, last_s,
                          enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers,
                          dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers,
                          max_trn_len=max_len, tta=None).to(device)#114, tta=tfms2).to(device)
    model_288_512.load_state_dict(torch.load(f'/media/nofreewill/Datasets_nvme/kaggle/bms-code/model_weights/model_38.pth', map_location=device))
    model_288_512.eval()
    # # model_384_384
    # N, n, ff, first_k, first_s, last_s = 32, 128, 128, 3,2,1
    # enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers =  512, 16, 512*4, 6
    # dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers =  512, 16, 512*4, 6
    # model_384_384 = Model(bpe_num, N, n, ff, first_k, first_s, last_s,
    #                       enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers,
    #                       dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers,
    #                       max_trn_len=max_len, tta=None).to(device)#114, tta=tfms2).to(device)
    # model_384_384.load_state_dict(torch.load(f'/media/nofreewill/Datasets_nvme/kaggle/bms-code/model_weights/done/model_384_384.pth', map_location=device))
    # model_384_384.eval()

    models = [model_288_512]#, model_384_384]#*4
    weights = [1.]#, 1.]
    img_sizes = [(256,544)]#(288, 512), (384, 384)]

    val_dls = []
    for img_size in img_sizes:
        val_ds = DS(imgs_path, img_size, valid_df, swp, train=False)
        val_sampler = SplitterSampler(val_ds, shuffle=False)
        val_dl = DataLoader(val_ds, batch_sampler=val_sampler, pin_memory=True, num_workers=2, prefetch_factor=2)
        val_dl.dataset.build_new_split(bs, randomize=False, drop_last=False)
        val_dls.append(val_dl)

    from valid_utils import levenshtein, validate
    w = (ds_path/'beam_search_output.csv').open('a', buffering=1)
    for bw in [1]:#[0,1,2,3,4,6,8,12,16,32,96]:  # 3
        print(f'{bw}: ', end='')
        w.write(f',,,,{bw}\n')
        if (bw == 0):
            m = models[0]
            val_score = validate(m, val_dls, device)
            print(val_score)
        else:
            m = BeamSearcher(models, weights, bpe_num, bw)
        lev_score = levenshtein(m, val_dls, swp, device, w)
        print(lev_score)
        w.write(f',,,,{bw},{lev_score}\n')
    w.close()

