from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import random
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from kornia import augmentation as KA
import kornia as K


class DS(Dataset):
    def __init__(self, imgs_path, img_size, df, swp=None, max_len=256, train=False, dropout_bpe=0.1):
        self.imgs_path = imgs_path
        self.img_size = img_size
        self.max_len = max_len
        self.train = train
        self.swp = swp
        self.df = df
        self.dropout_bpe = dropout_bpe

        # Fill with with indices before each epoch
        self.batches = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # Input
        image_id = row[0]
        # Target
        inchi_str = 'InChI=1S/' + '/'.join(row[13:])
        inchi_str = re.sub(r'/+', '/', inchi_str)
        inchi_str = inchi_str[:-1] if inchi_str.endswith('/') else inchi_str
        # # Additional Targets
        # atom_nums = np.array(row[1:13], dtype=np.int16)
        # h_present = int(row[15] != '')
        # b_plus = row[16].count('+')
        # b_minus = max(0, (row[16].count('-') - b_plus)//2)
        # t_plus = row[17].count('+')
        # t_minus = row[17].count('-')
        # m = 0 if row[18]=='' else int(row[18][1:]) + 1
        # s = 0 if row[19]=='' else int(row[19][1:]) + 1
        # i_present = int(row[20] != '')
        # ih_present = int(row[21] != '')
        # ib_plus = row[22].count('+')
        # ib_minus = max(0, (row[22].count('-') - b_plus)//2)
        # it_plus = row[23].count('+')
        # it_minus = row[23].count('-')
        # im_ = 0 if row[24]=='' else int(row[24][1:]) + 1
        # is_ = 0 if row[25]=='' else int(row[25][1:]) + 1
        # additional_target = torch.tensor([*atom_nums, h_present, b_plus, b_minus, t_plus, t_minus, m, s,
        #                                   i_present, ih_present, ib_plus, ib_minus, it_plus, it_minus, im_, is_]).long()
        # "Real" data
        c1,c2,c3 = image_id[:3]
        img_path = self.imgs_path/c1/c2/c3/(image_id+'.png')
        img_pil = Image.open(img_path)
        # Preprocess
        w,h = img_pil.size
        # Horizontalize
        img_pil = img_pil if w >= h else img_pil.rotate(90, expand=True)
        w, h = img_pil.size
        img_pil_orig = img_pil
        # Augment
        if self.train:
            # Random -90 Rotate
            if 1.33*h > w > 0.75*h:  # Changeable
                img_pil = img_pil if random.random()<0.97 else img_pil.rotate(-90, expand=True)  # Changeable
                w, h = img_pil.size
            # Add & Remove Points
            img_tensor = T.ToTensor()(img_pil)*-1+1
            max_val = img_tensor.max()
            r_add = np.random.choice([0.01,0.003,0.001,0.0001,0.],p=[0.02,0.04,0.18,0.26,0.5])*(1+random.random()*0.5-0.25)
            r_rem = np.random.choice([0.3,0.2,0.1,0.01,0.],p=[0.1,0.2,0.4,0.27,0.03])*(1+random.random()*0.5-0.25)
            add_noise = (torch.rand_like(img_tensor)<r_add)
            rem_noise = (torch.rand_like(img_tensor)>r_rem)
            img_tensor = (img_tensor + add_noise) * rem_noise
            img_tensor.clamp_(max=max(max_val, max_val+np.random.uniform(-0.1,0.1)))
            img_pil = T.ToPILImage()(img_tensor*-1+1)
            # Resize/Rescale
            w,h = self.get_new_sizes(w,h)
            img_pil = img_pil.resize((w,h), resample=Image.BICUBIC)
        # Resize if needed
        if max(w,h) > self.img_size:
            ratio = min(self.img_size/w,self.img_size/h)
            w,h = int(ratio*w), int(ratio*h)
            img_pil = img_pil.resize((w,h), resample=Image.BICUBIC)
        # Augment
        if self.train:
            # Positioning
            rh,rw = random.random()*0.2-0.1, random.random()*0.2-0.1  # Changeable
            dh,dw = int(self.img_size-h*(1+rh)) // 2, int(self.img_size-w*(1+rw)) // 2
            dh,dw = min(self.img_size-h, max(0, dh)), min(self.img_size-w, max(0, dw))
        else:
            dh, dw = (self.img_size - h) // 2, (self.img_size - w) // 2
        img_tensor = T.ToTensor()(img_pil)*-1 + 1
        # Augment
        if self.train:
            # Add & Remove Points
            max_val = img_tensor.max()
            r_add = np.random.choice([0.01,0.003,0.001,0.0001,0.],p=[0.02,0.04,0.18,0.26,0.5])*(1+random.random()*0.5-0.25)
            r_rem = np.random.choice([0.3,0.2,0.1,0.01,0.],p=[0.1,0.2,0.4,0.27,0.03])*(1+random.random()*0.5-0.25)
            add_noise = (torch.rand_like(img_tensor)<r_add)
            rem_noise = (torch.rand_like(img_tensor)>r_rem)
            img_tensor = (img_tensor + add_noise) * rem_noise
            img_tensor.clamp_(max=max(max_val, max_val+np.random.uniform(-0.1,0.1)))
            # Erase
            img_tensor = self.erase_tfms(img_tensor)
        zero_tensor = torch.zeros(1, self.img_size, self.img_size)
        zero_tensor[:, dh:dh + h, dw:dw + w] = img_tensor
        img_tensor = zero_tensor
        img_tensor = (img_tensor - 0.0044) / 0.0327

        # Label
        inchi_str = inchi_str[10:]
        bpe_ids = [1]+self.swp.encode(inchi_str,enable_sampling=self.train,alpha=self.dropout_bpe)+[2]
        bpe_ids = bpe_ids if len(bpe_ids)-1 < self.max_len else [1]+self.swp.encode(inchi_str)+[2]
        bpe_len = len(bpe_ids)
        bpe_tensor = torch.zeros(self.max_len, dtype=torch.long)
        bpe_tensor[:bpe_len] = torch.tensor(bpe_ids, dtype=torch.long)
        return img_tensor, bpe_tensor, bpe_len

    def get_new_sizes(self, w,h):#,W,H):
        rs = 0.08  # Changeable
        rs = random.uniform(1-rs, 1)
        rrm = 0.87#0.67  # Changeable
        #rrM = min(1/rrm, self.img_size/max(w*rs,h))  # max out at img_size
        rrM = 1/rrm  # Changeable ver2
        rrm, rrM = np.log(rrm), np.log(rrM)
        rr = random.uniform(rrm, rrM)
        #rr = np.exp(rr)
        rr = min(np.exp(rr), self.img_size/max(w*rs,h))  # Changeable ver2
        w, h = (rr * w * rs, rr * h) if random.random() > 0.5 else (rr * w, rr * h * rs)
        return int(w), int(h)

    def erase_tfms(self, img_tensor, n=4, r=0.04):  # Changeable
        _,h,w = img_tensor.shape
        max_cut_size = max(h,w)
        h_idxs = torch.arange(h).reshape(-1,1)
        w_idxs = torch.arange(w).reshape(1,-1)
        h_cut_start = torch.randint(h, size=(1,n))
        w_cut_start = torch.randint(w, size=(n,1))
        h_cut_size = torch.randint(int(max_cut_size*r)+1, size=(1,n))
        w_cut_size = torch.randint(int(max_cut_size*r)+1, size=(n,1))
        h_cuts = (h_cut_start<h_idxs)*(h_idxs<h_cut_start+h_cut_size)
        w_cuts = (w_cut_start<w_idxs)*(w_idxs<w_cut_start+w_cut_size)
        cuts = (h_cuts.float() @ w_cuts.float()).clamp(max=1.)*-1+1
        img_tensor = img_tensor * cuts
        return img_tensor

    def build_new_split(self, bs, randomize=False, drop_last=False):
        # Sort
        tqdm.pandas()
        samples = self.df.progress_apply(lambda x: [1]+self.swp.encode('/'.join(x[13:]))+[2], axis=1)
        samples = sorted(samples, key=lambda x: (len(x)-1)*(1+randomize*(random.random()*0.4-0.2)))
        # Construct batches
        self.batches = [list(range(i,min(len(samples),i+bs))) for i in range(0,len(samples),bs)]
        if drop_last and len(self.batches[-1])<bs: self.batches = self.batches[:-1]


from torch.utils.data import Sampler
class SplitterSampler(Sampler):
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.dataset.batches)
        return iter(self.dataset.batches)

    def __len__(self):
        return len(self.dataset.batches)

