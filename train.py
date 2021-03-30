import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sentencepiece as sp

# HOTFIX?
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# HOTFIX?

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data_loaders.data_loader import DS, SplitterSampler
from model_architecture.model import Model
from loss import LabelSmoothingLoss
from valid_utils import validate, levenshtein

def imgs_mixup(imgs_tensor, alpha=0.4):
    bs = len(imgs_tensor)
    device = imgs_tensor.device
    lam = torch.tensor(np.random.beta(alpha,alpha, size=(bs//2,1,1,1)), dtype=torch.float32).to(device)
    imgs_tensor = imgs_tensor[:bs//2] *lam + (1-lam)* imgs_tensor[bs//2:]
    return imgs_tensor, lam


if __name__ == '__main__':
    #
    img_size = 192
    bpe_num = 4096
    max_len = 256
    lr = 3e-4
    bs = 128
    BS = None
    epochs_num = 24
    start_epoch_num = 0
    #
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    # Dataset
    ds_path = Path('/media/nofreewill/Datasets_nvme/kaggle/bms-data/').absolute()
    imgs_path = ds_path/'images/resized'/str(img_size)/'train'
    imgs_path = imgs_path if imgs_path.exists() else ds_path / 'images/train'
    df = pd.read_csv(ds_path/'train_labels.csv')
    is_valid = np.array([i%50==0 for i in range(len(df))])
    train_df = df.iloc[~is_valid]
    valid_df = df.iloc[is_valid]

    sp.SentencePieceProcessor()
    subwords_path = ds_path/'subwords'/f'bpe_{bpe_num}.model'
    swp = sp.SentencePieceProcessor(str(subwords_path))

    val_ds = DS(imgs_path, img_size, valid_df, swp, train=False)
    val_sampler = SplitterSampler(val_ds, shuffle=False)
    val_dl = DataLoader(val_ds, batch_sampler=val_sampler, pin_memory=True, num_workers=8, prefetch_factor=2)
    val_dl.dataset.build_new_split(bs, randomize=False, drop_last=False)

    trn_ds = DS(imgs_path, img_size, train_df, swp, max_len, train=True)
    tqdm.pandas()
    weights = train_df.progress_apply(lambda x: len(x[1])**1, axis=1).to_list()
    trn_sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_df), replacement=True)
    trn_dl = DataLoader(trn_ds, batch_size=bs, sampler=trn_sampler,
                        drop_last=True,
                        pin_memory=True, num_workers=8, prefetch_factor=4)

    # Model
    N, n = 32, 64
    enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers = 512, 8, 4*512, 6#24
    dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers = 512, 8, 4*512, 6#12
    model = Model(bpe_num, N, n,
                  enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers,
                  dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers,
                  max_len).to(device)

    # Record
    open_mode = "w" if start_epoch_num==0 else "a"
    w_trn = open('training_stats/train_stats.csv', open_mode, 1)
    w_val = open('training_stats/valid_stats.csv', open_mode, 1)

    # Train params
    total_steps = epochs_num*len(trn_dl)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters())
    lr_sched = lr_scheduler.OneCycleLR(optimizer,lr,total_steps,
                                       div_factor=1e3,pct_start=1/epochs_num,final_div_factor=1.)
    scaler = torch.cuda.amp.GradScaler()

    if start_epoch_num>0:
        model.load_state_dict(torch.load(f'model_weights/model_{start_epoch_num-1}.pth', map_location=device))
        for _ in range(start_epoch_num*len(trn_dl)): lr_sched.step()

    # Train
    N = 0
    m = 0
    prev_sched_step_i = 0
    for epoch_num in range(start_epoch_num,epochs_num):
        model.train()
        for i, batch in enumerate(tqdm(trn_dl)):
            imgs_tensor, lbls_tensor, lbls_len = batch
            n = (lbls_len-1).sum().item()
            lbls_tensor = lbls_tensor[:,:lbls_len.max()]
            imgs_tensor, lbls_tensor = imgs_tensor.to(device), lbls_tensor.to(device)

            history_tensor = lbls_tensor[:, :-1]
            predict_tensor = lbls_tensor[:, 1:]
            predict_mask = (predict_tensor == 0)


            with torch.cuda.amp.autocast(enabled=True):
                dec_out = model(imgs_tensor, history_tensor)
                loss = loss_fn(dec_out.flatten(0,1), predict_tensor.flatten())
                loss = (loss*(~predict_mask.flatten())).sum()/(n if BS is None else BS)

            scaler.scale(loss).backward()

            N += n
            if BS is None or N>=BS:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                for _ in range(i-prev_sched_step_i):
                    lr_sched.step()
                prev_sched_step_i = i

                # Record
                if m%100==0:
                    loss = loss.item() * (1 if BS is None else BS / n)
                    w_trn.write(f'{i},{loss},{lr_sched.get_last_lr()[0]}\n')
                # if m % 2500 == 0:
                #     torch.save(model.state_dict(), f'model_weights/model_partial.pth')
                m += 1

                N = 0

        # save model
        torch.save(model.state_dict(), f'model_weights/model_{epoch_num}.pth')
        # validate
        w_val.write(f'{epoch_num}')

        print('validate')
        val_loss = validate(model, val_dl, device)
        w_val.write(f',{val_loss}')
        print('levenshtein')
        val_lev = levenshtein(model, val_dl, swp, device)
        w_val.write(f',{val_lev}')

        w_val.write('\n')

    w_trn.close()
    w_val.close()
