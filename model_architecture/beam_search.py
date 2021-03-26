import torch
from torchvision import transforms as T

from pathlib import Path
import sentencepiece as sp

from model_architecture.model import Model


class BeamSearcher:
    def __init__(self, models, weights, tfmss, bpe_num, beam_width=4, max_len=160):
        self.models = models
        self.weights = weights
        self.tfmss = tfmss
        self.bw = beam_width
        self.max_len = max_len
        self.bpe_num = bpe_num

    def eval(self):
        for model in self.models:
            model.eval()
    def train(self):
        for model in self.models:
            model.train()

    def predict(self, imgs_tensor, max_pred_len=114):
        bs = len(imgs_tensor)
        device = imgs_tensor.device
        bw = self.bw
        max_len = max_pred_len
        bpe_num = self.bpe_num
        eos_id = 2

        # Initialize

        #enc_outs = [model.encoder_output(self.tfmss[i](imgs_tensor)) for i, model in enumerate(self.models)]
        enc_outs = [model.encoder_output(imgs_tensor) for i, model in enumerate(self.models)]

        start_tokens = torch.ones(bs, 1, device=device, dtype=torch.long)
        caches = [None]*len(self.models)
        route_probs = torch.zeros(bs * bw, 1, device=device)

        beam_ids   = torch.zeros(bs, max_len, device=device, dtype=torch.long)
        beam_probs = torch.full((bs,), float('-inf'), device=device)
        beam_lens  = torch.ones(bs, device=device, dtype=torch.long)
        done_beams = torch.zeros(bs, device=device, dtype=torch.bool)

        # First decoder output

        bpe_probses, caches = list(zip(*[self.models[i].decoder_output(enc_outs[i], caches[i], start_tokens)
                                         for i in range(len(self.models))]))
        bpe_probs = sum([self.weights[i] * bpe_probses[i] for i in range(len(self.models))])
        topk_probs, topk_idxs = bpe_probs.topk(bw, dim=1)
        route_probs = topk_probs.reshape(bs * bw, 1)

        enc_outs = [enc_outs[i].unsqueeze(2).repeat(1,1,bw,1).reshape(-1,bs*bw,self.models[i].dec_d_model)
                    for i in range(len(self.models))]
        caches = [caches[i].unsqueeze(3).repeat(1,1,1,bw,1)\
                      .reshape(self.models[i].num_layers,-1,bs*bw,self.models[i].dec_d_model)
                  for i in range(len(self.models))]
        decoded_tokens = torch.cat([torch.ones(bs * bw, 1).long().to('cuda:0'), topk_idxs.reshape(bs * bw, 1)], dim=1)

        # Loop
        while True:
            route_len = decoded_tokens.shape[1]
            if route_len == max_len:
                break

            bpe_probses, caches = list(zip(*[self.models[i].decoder_output(enc_outs[i], caches[i], decoded_tokens)
                                             for i in range(len(self.models))]))
            bpe_probs = sum([self.weights[i] * bpe_probses[i] for i in range(len(self.models))])
            route_bpe_probs = route_probs + bpe_probs

            # check dones
            best_bpe_probs = route_bpe_probs.reshape(bs, bw * bpe_num).max(dim=1)[0]
            done_beams = beam_probs > best_bpe_probs

            if done_beams.all():
                break

            # step
            topk_probs, topk_idxs = route_bpe_probs.reshape(bs, bw * bpe_num).topk(bw, dim=1)

            index_helper = torch.arange(bs, device=device)
            shift_idxs = (index_helper * bw).reshape(bs, 1).repeat(1, bw).reshape(bs * bw)
            is_eos, eos_idxs = ((topk_idxs%bpe_num)==eos_id).max(dim=1)
            is_better = (topk_probs[index_helper,eos_idxs] > beam_probs) * is_eos
            beam_probs[is_better] = topk_probs[is_better,eos_idxs[is_better]]
            beam_lens[is_better] = route_len
            beam_ids[is_better,:route_len] = decoded_tokens[index_helper*bw+topk_idxs[index_helper,eos_idxs]//bpe_num][is_better]

            # route_probs
            route_probs = topk_probs.reshape(bs * bw, 1)
            keep_idxs = shift_idxs + (topk_idxs // bpe_num).reshape(bs * bw)
            caches = [cache[:,:,keep_idxs] for cache in caches]
            decoded_tokens = decoded_tokens[keep_idxs]
            decoded_tokens = torch.cat([decoded_tokens, (topk_idxs % bpe_num).reshape(bs * bw, 1)], dim=1)

        return beam_ids, beam_lens


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img_size = 192
    interpolation_type = '_BICUBIC'
    bpe_num = 2**15
    bs = 64
    torch.backends.cudnn.benchmark = True

    import pickle
    from torch.utils.data import DataLoader
    from data_loaders.data_loader import DS, SplitterSampler
    ds_path = Path("/media/nofreewill/Datasets_nvme/kaggle/bms/data")
    swp = sp.SentencePieceProcessor(f'{ds_path}/subwords/bpe_{bpe_num}.model')
    imgs_path = ds_path/'resized'/(str(img_size)+interpolation_type)/'train'
    valid_samples_path = ds_path/'valid_samples.pkl'
    #valid_samples_path = ds_path/'test_samples.pkl'
    valid_samples = pickle.load(valid_samples_path.open('rb'))
    val_ds = DS(imgs_path, img_size, valid_samples, train=False)
    val_sampler = SplitterSampler(val_ds)
    val_dl = DataLoader(val_ds, batch_sampler=val_sampler, pin_memory=True, num_workers=4)
    val_dl.dataset.build_new_split(bs, randomize=False, drop_last=False)

    # eval
    import kornia as K
    from torch import nn
    tfms1 = T.Compose([K.Resize(int(img_size*1.25),'nearest'), K.Resize(img_size,'bicubic')])#nn.Identity()#
    tfms2 = K.Rotate(torch.tensor(-1.5).to(device))
    # Model
    N, n = 32, 96
    enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers = 512,  8, 4*512, 6#16
    dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers = 512,  8, 4*512, 6#768, 12, 4*768,  6
    model = Model(bpe_num, N, n,
                  enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers,
                  dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers,
                  114, tta=tfms2).to(device)
    model.load_state_dict(torch.load(f'/media/nofreewill/Datasets_nvme/kaggle/bms/code/model_weights/model_0.pth', map_location=device))
    model.eval()

    # models = [model]*2
    # weights = [1.,1.]
    # tfmss = [tfms1, tfms2]
    models = [model]#*2
    weights = [1.]#,1.]
    tfmss = [tfms1]#, tfms2]

    from valid_utils import levenshtein, validate
    w = (ds_path/'beam_search_output.csv').open('a')
    for bw in [0]:#[0,1,2,3,4,6,8,12,16,32,96]:  # 3
        print(f'{bw}: ', end='')
        w.write(f',,,,{bw}\n')
        if (bw == 0):
            m = model
            val_score = validate(m, val_dl, device)
            print(val_score)
        else:
            m = BeamSearcher(models, weights, tfmss, bpe_num, bw)
        lev_score = levenshtein(m, val_dl, swp, device, w)
        print(lev_score)
        w.write(f',,,,{bw},{lev_score}\n')
    w.close()

