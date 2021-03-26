import torch
from torchvision import transforms as T
import kornia as K

from pathlib import Path
import sentencepiece as sp

from model_architecture.model import Model
from model_architecture.beam_search import BeamSearcher


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img_size = 384#192
    bpe_num = 2**15
    bs = 64
    torch.backends.cudnn.benchmark = True

    import pickle
    from torch.utils.data import DataLoader
    from data_loaders.data_loader import DS, SplitterSampler
    ds_path = Path("/home/nofreewill/Documents/kaggle/bms/bms-data")

    swp = sp.SentencePieceProcessor(f'{ds_path}/subwords/bpe_{bpe_num}.model')
    imgs_path = ds_path/'resized'/str(img_size)/'train'
    imgs_path = imgs_path if imgs_path.exists() else ds_path / 'images/train'
    valid_samples_path = ds_path/'samples/valid_samples.pkl'

    valid_samples = pickle.load(valid_samples_path.open('rb'))
    val_ds = DS(imgs_path, img_size, valid_samples, train=False)
    val_sampler = SplitterSampler(val_ds)
    val_dl = DataLoader(val_ds, batch_sampler=val_sampler, pin_memory=True, num_workers=4)
    val_dl.dataset.build_new_split(bs, randomize=False, drop_last=False)

    # eval
    tfms1 = T.Compose([K.Resize(int(img_size*1.25),'nearest'), K.Resize(img_size,'bicubic')])#nn.Identity()#
    tfms2 = K.Rotate(torch.tensor(-1.5).to(device))
    # Model
    N, n = 32, 128
    enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers = 512,  8, 4*512, 6#16
    dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers = 512,  8, 4*512, 6#768, 12, 4*768,  6
    model = Model(bpe_num, N, n,
                  enc_d_model, enc_nhead, enc_dim_feedforward, enc_num_layers,
                  dec_d_model, dec_nhead, dec_dim_feedforward, dec_num_layers,
                  256, tta=None).to(device)#114, tta=tfms2).to(device)
    model.load_state_dict(torch.load(f'/home/nofreewill/Documents/kaggle/bms/bms-code/model_weights/model_6_384.pth', map_location=device))
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
            # val_score = validate(m, val_dl, device)
            # print(val_score)
        else:
            m = BeamSearcher(models, weights, tfmss, bpe_num, bw)
        lev_score = levenshtein(m, val_dl, swp, device, w)
        print(lev_score)
        w.write(f',,,,{bw},{lev_score}\n')
    w.close()
