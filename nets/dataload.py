import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import h5py
import random

class GetRGB(Dataset):
    def __init__(self, path, crop_size, transform, img_size=[2700, 3600], img_name=False):
        self.path = path
        self.img_list = os.listdir(path)
        self.crop_size = crop_size
        self.transform = transform
        self.img_name = img_name
        self.max_W = img_size[1]
        self.max_H = img_size[0]
    
    def __getitem__(self, idx):
        rand_h = random.randint(0, self.max_H - self.crop_size)
        rand_w = random.randint(0, self.max_W - self.crop_size)
        with h5py.File(os.path.join(self.path, self.img_list[idx]), 'r') as f:
            img = f['img'][rand_h:rand_h+self.crop_size, rand_w:rand_w+self.crop_size, :] # (H, W, C)
        img = Image.fromarray(img)
        img = self.transform(img)
        
        if self.img_name:
            return img, self.img_list[idx]
        return img
    
    def __len__(self):
        return len(self.img_list)


def train_rgbloader(img_folder, crop_size, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
        ])
    dataloader = DataLoader(GetRGB(img_folder, crop_size, transform, img_name=False),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True
                            )
    return dataloader


def valid_rgbloader(img_folder, crop_size, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
    dataloader = DataLoader(GetRGB(img_folder, crop_size, transform, img_size=[3600, 4800], img_name=False),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=True
                            )
    return dataloader


def rgb_generator(dataloader:torch.utils.data.DataLoader, device, dtype):
    with torch.device('cpu'):
        for index, data in enumerate(dataloader):
            with torch.device(device):
                yield index, data.to(device).to(dtype)
                

class GetRAW(Dataset):
    def __init__(self, path, crop_size, img_size=[2160, 3840], img_name=False):
        self.path = path
        self.img_list = os.listdir(path)
        self.crop_size = crop_size
        self.img_name = img_name
        self.max_W = img_size[1]
        self.max_H = img_size[0]
    
    def __getitem__(self, idx):
        rand_h = random.randint(0, self.max_H - self.crop_size)
        rand_w = random.randint(0, self.max_W - self.crop_size)
        with h5py.File(os.path.join(self.path, self.img_list[idx]), 'r') as f:
            raw_img = torch.from_numpy(f['raw_img'][rand_h:rand_h+self.crop_size, rand_w:rand_w+self.crop_size]) # [H, W]
            raw_color = torch.from_numpy(f['raw_color'][rand_h:rand_h+self.crop_size, rand_w:rand_w+self.crop_size]) # [H, W]
            raw_wb = torch.from_numpy(f['raw_wb_matrix'][:])
            raw_cm = torch.from_numpy(f['raw_color_matrix'][:])
        
        if self.img_name:
            return raw_img, raw_color, raw_wb, raw_cm, self.img_list[idx]
        return raw_img, raw_color, raw_wb, raw_cm
    
    def __len__(self):
        return len(self.img_list)
    
    
def train_rawloader(img_folder, crop_size, batch_size, num_workers):
    dataloader = DataLoader(GetRAW(img_folder, crop_size, img_name=False),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True
                            )
    return dataloader


def valid_rawloader(img_folder, crop_size, batch_size, num_workers):
    dataloader = DataLoader(GetRAW(img_folder, crop_size, img_size=[2700, 4800], img_name=False),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=True
                            )
    return dataloader


def raw_generator(dataloader:torch.utils.data.DataLoader, device, dtype):
    with torch.device('cpu'):
        for index, datas in enumerate(dataloader):
            raw_img, raw_color, raw_wb, raw_cm = datas
            with torch.device(device):
                yield index, (raw_img.to(device).to(dtype), raw_color.to(device), raw_wb.to(device).to(dtype), raw_cm.to(device).to(dtype))