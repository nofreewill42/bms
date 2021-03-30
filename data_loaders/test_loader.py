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
        w,h = img_pil.size
        # Horizontalize
        img_pil = img_pil if w >= h else img_pil.rotate(90, expand=True)  # TODO: CNN decides
        w, h = img_pil.size
        # Resize if needed
        if max(w,h) > self.img_size:
            ratio = min(self.img_size/w,self.img_size/h)
            w,h = int(ratio*w), int(ratio*h)
            img_pil = img_pil.resize((w,h), resample=Image.BICUBIC)
        dh, dw = (self.img_size - h) // 2, (self.img_size - w) // 2
        img_tensor = T.ToTensor()(img_pil)*-1 + 1
        zero_tensor = torch.zeros(1, self.img_size, self.img_size)
        zero_tensor[:, dh:dh + h, dw:dw + w] = img_tensor
        img_tensor = zero_tensor
        img_tensor = (img_tensor - 0.0044) / 0.0327
        return img_tensor, img_stem