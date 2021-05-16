from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class TestDS(Dataset):
    def __init__(self, imgs_path, img_stems, img_size):
        self.imgs_path = imgs_path
        self.img_stems = img_stems
        self.w_max, self.h_max = img_size[1], img_size[0]

    def __len__(self):
        return len(self.img_stems)

    def __getitem__(self, i):
        img_stem = self.img_stems[i]
        c1,c2,c3 = img_stem[:3]
        img_path = self.imgs_path/c1/c2/c3/(img_stem+'.png')
        img_pil = Image.open(img_path)
        w,h = img_pil.size
        # Horizontalize
        img_pil = img_pil if w >= h else img_pil.rotate(90, expand=True)  # TODO: CNN decides
        w, h = img_pil.size
        # Resize if needed
        ratio = max(w/self.w_max, h/self.h_max)
        if ratio > 1:
            w,h = int(w/ratio), int(h/ratio)
            img_pil = img_pil.resize((w,h), resample=Image.BICUBIC)
        dh, dw = (self.h_max - h) // 2, (self.w_max - w) // 2
        img_tensor = T.ToTensor()(img_pil)*-1 + 1
        zero_tensor = torch.zeros(1, self.h_max, self.w_max)
        zero_tensor[:, dh:dh + h, dw:dw + w] = img_tensor
        img_tensor = zero_tensor
        img_tensor = (img_tensor - 0.0044) / 0.0327
        return img_tensor, img_stem