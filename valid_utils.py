import torch
import edlib
from tqdm import tqdm


def validate(model, val_dl, device='cpu'):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # even if trained with Label Smoothing
    model.eval()
    N = 0
    val_loss = 0
    for j, batch in enumerate(tqdm(val_dl)):
        imgs_tensor, lbls_tensor, lbls_len = batch
        N += (lbls_len-1).sum().item()
        lbls_tensor = lbls_tensor[:,:lbls_len.max()]
        imgs_tensor, lbls_tensor = imgs_tensor.to(device), lbls_tensor.to(device)

        history_tensor = lbls_tensor[:,:-1]
        predict_tensor = lbls_tensor[:,1:]
        predict_mask = (predict_tensor==0)

        with torch.no_grad():
            outs_tensor = model(imgs_tensor, history_tensor)
            loss = loss_fn(outs_tensor.flatten(0,1), predict_tensor.flatten())
            loss = (loss*(~predict_mask.flatten())).sum()

        val_loss += loss.item()
    model.train()

    return val_loss/N

def levenshtein(model, val_dl, swp, device='cpu', w=None):
    model.eval()
    n = 0
    lev_sum = 0
    for j, batch in enumerate(tqdm(val_dl)):
        imgs_tensor, lbls_tensor, lbls_len = batch
        lbls_tensor = lbls_tensor[:,:lbls_len.max()]
        imgs_tensor, lbls_tensor = imgs_tensor.to(device), lbls_tensor.to(device)

        with torch.no_grad():
            lbl_ids, lens = model.predict(imgs_tensor, 114)
            lbl_ids = torch.stack(lbl_ids).T if not isinstance(lbl_ids, torch.Tensor) else lbl_ids
        if (lens==val_dl.dataset.max_len).sum()==len(lens):
            return float('inf')  # it is repeating itself in hole batch, most likely would on all data..

        for i in range(len(lbl_ids)):
            pred = lbl_ids[i].tolist()[:lens[i]]
            lbl = lbls_tensor[i].tolist()[:(lbls_tensor[i]!=0).sum()]
            pred_text = swp.decode(pred)
            lbl_text = swp.decode(lbl)
            lev_dist = edlib.align(pred_text, lbl_text)['editDistance']
            lev_sum += lev_dist
            n += 1

            if w is not None:
                w.write(f'"{pred_text}","{lbl_text}","{val_dl.dataset.samples[n-1][0]}",{lev_dist}\n')

    model.train()
    return lev_sum/n