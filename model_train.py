# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:02:51 2022

@author: yangzhen
"""
import torch.nn as nn

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
import cv2
import pickle
from tqdm import tqdm 


k1=256
k2=1024
k3=4
train_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([k1,k1]),
    # transforms.Normalize((0.5), (0.5))
]) 
def ARLoader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((k1,k1))
    # img_pil = img_pil.convert('RGB')
    img_tensor = train_preprocess(img_pil)
    return img_tensor


class trainset(Dataset):
    def __init__(self, loader=ARLoader):
        #定义好 image 的路径
        self.images = train_list
        self.loader = loader
        self.target = train_label
    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target
    
    def __len__(self):
        return len(self.images)




class ResDown(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResDown, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x


class ResUp(nn.Module):

    def __init__(self, channel_in, channel_out, scale=2):
        super(ResUp, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x
    

class Encoder(nn.Module):
    def __init__(self, channels, ch=64, z=64, dim=32):
        super(Encoder, self).__init__()
        self.conv1 = ResDown(channels, ch)  # 64
        self.conv2 = ResDown(ch, 2*ch)  # 32
        self.conv3 = ResDown(2*ch, 4*ch)  # 16
        self.conv4 = ResDown(4*ch, 8*ch)  # 8
        self.conv5 = ResDown(8*ch, 8*ch)  # 4
        self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)  # 2
        self.conv_log_var = nn.Conv2d(8*ch, z, 2, 2)  # 2
        self.fc1 = nn.Linear(int(k1/64)**2*z, dim)
        self.fc2 = nn.Linear(int(k1/64)**2*z, dim)
    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv_mu(x)
        # print(x.shape)
        mu = self.fc1(x.view(x.size(0), -1))
        log_var = self.fc2(x.view(x.size(0), -1))
        x = self.sample(mu, log_var)


        return x, mu, log_var


class Decoder(nn.Module):
    def __init__(self, channels, ch=64, z=64, dim=32):
        super(Decoder, self).__init__()
        self.conv1 = ResUp(z, ch*8)
        self.conv2 = ResUp(ch*8, ch*8)
        self.conv3 = ResUp(ch*8, ch*4)
        self.conv4 = ResUp(ch*4, ch*2)
        self.conv5 = ResUp(ch*2, ch)
        self.conv6 = ResUp(ch, ch//2)
        self.conv7 = nn.Conv2d(ch//2, channels, 3, 1, 1)
        self.fc1 = nn.Linear(dim, int(k1/64)**2*z)
        self.z = z
    def forward(self, x):
        x = self.fc1(x).view(x.size(0), self.z , int(k1/64), int(k1/64))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x 


class VAE(nn.Module):
    def __init__(self, channel_in, ch=64, z=64, dim=8):
        super(VAE, self).__init__()

        self.encoder = Encoder(channel_in, ch=ch, z=z, dim=dim)
        self.decoder = Decoder(channel_in, ch=ch, z=z, dim=dim)

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon = self.decoder(encoding)
        return recon, mu, log_var




reconstruction_function = nn.MSELoss(size_average=False)
# reconstruction_function = FocalLoss2d()

def loss_function(recon_x, x, mu, logvar):

    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD
# model = VAE(channel_in=1,ch=128, z=64, dim=16)
#%%
num_epochs = 10
batch_size = 32
path='D:/data/auditory/USV_MP4-toXiongweiGroup/VAE/training/'
train_list=glob.glob(path+'*.png')
train_label=list(np.zeros(len(train_list)))
dataloader = DataLoader(trainset(), batch_size=batch_size, shuffle=True)
res=[]
dim_idx=[32] 
for k in range(len(dim_idx)):
    print('processing embedding dimension: '+str(dim_idx[k]))
    model = VAE(channel_in=1,ch=128, z=64, dim=dim_idx[k])
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1, verbose=False) 
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        with tqdm(total=len(dataloader)) as t: 
            for batch_idx, data in enumerate(dataloader):
                img, _ = data
                # img = img.view(img.size(0), -1)
                img = Variable(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(img)
                loss = loss_function(recon_batch, img, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                t.set_description('Processing epoch: '+str(batch_idx+1)+' train loss: '+str(train_loss/(batch_idx+1)))
                t.update(1)
            loss_record= (loss.item() / len(img))
            scheduler.step()

    #%
    torch.save(model, './models/cvae_model_1201_d'+str(dim_idx[k])+'.pth')