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
from valid_utils import validate, levenshtein

def imgs_mixup(imgs_tensor, alpha=0.4):
    bs = len(imgs_tensor)
    device = imgs_tensor.device
    lam = torch.tensor(np.random.beta(alpha,alpha, size=(bs//2,1,1,1)), dtype=torch.float32).to(device)
    imgs_tensor = imgs_tensor[:bs//2] *lam + (1-lam)* imgs_tensor[bs//2:]
    return imgs_tensor, lam


if __name__ == '__main__':
    #
    start_epoch_num = 0
    #
    img_size = 768
    bpe_num = 4096
    max_len = 256
    lr = 3e-4
    bs = 64
    BS = None
    epochs_num = 96
    # model
    N, n, ff, first_k, first_s, last_s = 64, 128, 128, 7,4,1
    enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers = 512, 8, 2048, 6
    dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers = 512, 8, 2048, 6
    #
    div_factor = lr / 3e-7
    pct_start = 1 / epochs_num
    final_div_factor = (lr/div_factor) / 3e-6
    # clip grad
    max_norm = 1.0
    #
    weight_decay = 0.
    dropout_p = 0.072
    dropout_h_base = 0.063
    dropout_dec_emb = 0.1
    #
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    # Dataset
    ds_path = Path('/media/nofreewill/Datasets_nvme/kaggle/bms-data/').absolute()
    imgs_path = ds_path/'images/resized'/str(img_size)/'train'
    imgs_path = imgs_path if imgs_path.exists() else ds_path / 'images/train'
    df = pd.read_csv(ds_path/'train_labels.csv')
    is_valid = np.array([i%50==0 for i in range(len(df))])  # Split
    train_df = df.iloc[~is_valid]
    valid_df = df.iloc[is_valid]
    weights_df = pd.read_csv(ds_path / 'train_labels_weights.csv')[~is_valid]
    weights_df.iloc[:,1] = np.power(weights_df.iloc[:,1], 1.)  # Complexity
    weights_df.iloc[:,2] = np.power(weights_df.iloc[:,2], 1.)  # Atom count
    weights_df.iloc[:,3] = np.power(weights_df.iloc[:,3], 1.)  # Atom rarity
    weights_df.iloc[:,4] = np.power(weights_df.iloc[:,4], 1.)  # Layer rarity
    weights_df.iloc[:,1:] = weights_df.iloc[:,1:]/weights_df.iloc[:,1:].sum(axis=0)
    weights = (weights_df.iloc[:,1:] * np.array([1.4,1.,0.7,0.5])).sum(axis=1).astype(np.float32).values

    sp.SentencePieceProcessor()
    subwords_path = ds_path/'subwords'/f'bpe_{bpe_num}.model'
    swp = sp.SentencePieceProcessor(str(subwords_path))

    val_ds = DS(imgs_path, img_size, valid_df, swp, train=False)
    val_sampler = SplitterSampler(val_ds, shuffle=False)
    val_dl = DataLoader(val_ds, batch_sampler=val_sampler, pin_memory=True, num_workers=8, prefetch_factor=2)
    val_dl.dataset.build_new_split(bs, randomize=False, drop_last=False)

    trn_ds = DS(imgs_path, img_size, train_df, swp, max_len, train=True)
    tqdm.pandas()
    trn_sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_df), replacement=True)
    trn_dl = DataLoader(trn_ds, batch_size=bs, sampler=trn_sampler,
                        drop_last=True,
                        pin_memory=True, num_workers=8, prefetch_factor=4)

    # Model
    model = Model(bpe_num, N, n, ff, first_k, first_s, last_s,
                  enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers,
                  dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers,
                  dropout_p, dropout_dec_emb,
                  max_len).to(device)

    # Record
    open_mode = "w" if start_epoch_num==0 else "a"
    w_trn = open('training_stats/train_stats.csv', open_mode, 1)
    w_val = open('training_stats/valid_stats.csv', open_mode, 1)

    # Train params
    total_steps = epochs_num*len(trn_dl)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(),weight_decay=weight_decay)
    lr_sched = lr_scheduler.OneCycleLR(optimizer,lr,total_steps,
                                       div_factor=div_factor,pct_start=pct_start,final_div_factor=final_div_factor)
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

            # drop head
            if epoch_num == 0:
                dropout_h = dropout_h_base * 2 * (len(trn_dl)-i)/len(trn_dl)
            else:
                dropout_h = dropout_h_base * 2 * ((epoch_num-1)*len(trn_dl) + i)/((epochs_num-1)*len(trn_dl))
            # forward
            with torch.cuda.amp.autocast(enabled=True):
                dec_out = model(imgs_tensor, history_tensor, dropout_h=dropout_h)
                loss = loss_fn(dec_out.flatten(0,1), predict_tensor.flatten())
                loss = (loss*(~predict_mask.flatten())).sum()/(n if BS is None else BS)

            scaler.scale(loss).backward()

            N += n
            if BS is None or N>=BS:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

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
                m += 1

                N = 0

        # save model
        torch.save(model.state_dict(), f'model_weights/model_{epoch_num}.pth')
        # validate
        w_val.write(f'{epoch_num}')

        print('validate')
        val_loss = validate(model, val_dl, device)
        w_val.write(f',{val_loss}')
        if epoch_num%5==0 or epoch_num>=epochs_num-5:
            print('levenshtein')
            val_lev = levenshtein(model, val_dl, swp, device)
            w_val.write(f',{val_lev}')

        w_val.write('\n')

    w_trn.close()
    w_val.close()
