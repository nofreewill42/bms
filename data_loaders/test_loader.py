from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class TestDS(Dataset):
    def __init__(self, imgs_path, img_stems, img_size):
        self.imgs_path = imgs_path
        self.img_stems = img_stems
        self.img_size = img_size

    def __len__(self):
        return len(self.img_stems)

    def __getitem__(self, i):
        img_stem = self.img_stems[i]
        c1,c2,c3 = img_stem[:3]
        img_path = self.imgs_path/c1/c2/c3/(img_stem+'.png')
        img_pil = Image.open(img_path)
        img_tensor = T.ToTensor()(img_pil)*-1 + 1

        h,w = img_tensor.shape[1:]
        if h>w: img_tensor = img_tensor.transpose(1, 2).flip(2)  # TODO: CNN decides
        h, w = img_tensor.shape[1:]
        dh, dw = (self.img_size - h) // 2, (self.img_size - w) // 2
        zero_tensor = torch.zeros(1, self.img_size, self.img_size)
        zero_tensor[0, dh:dh + h, dw:dw + w] = img_tensor
        img_tensor = zero_tensor

        return img_tensor, img_stem