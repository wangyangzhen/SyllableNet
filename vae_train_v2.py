# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:02:51 2022

@author: yangzhen
"""

import argparse
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


def get_args_parser():
    parser = argparse.ArgumentParser('VAE training', add_help=False)
    parser.add_argument('--data_path', default='/home/sonic/as13000_calExt/WYZ/imaging/bm/training/', type=str,
                        help='dataset path')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--ch', default=128, type=int)
    parser.add_argument('--z', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser


def main(args):
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
        # img_pil = img_pil.resize((k1,k1))
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
        """
        Residual down sampling block for the encoder
        """
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
        """
        Residual up sampling block for the decoder
        """
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
        """
        Encoder block
        Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
        As the network is fully convolutional it will work for images LARGER than 64
        For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n
        When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
        and log_var will be None
        """
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
            if self.training:
                mu = self.fc1(x.view(x.size(0), -1))
                log_var = self.fc2(x.view(x.size(0), -1))
                x = self.sample(mu, log_var)
            else:
                mu = self.conv_mu(x)
                x = mu
                log_var = None
    
            return x, mu, log_var
    
    
    class Decoder(nn.Module):
        """
        Decoder block
        Built to be a mirror of the encoder block
        """
    
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
        """
        VAE network, uses the above encoder and decoder blocks
        """
        def __init__(self, channel_in, ch=64, z=64, dim=8):
            super(VAE, self).__init__()
            """Res VAE Network
            channel_in  = number of channels of the image 
            z = the number of channels of the latent representation
            (for a 64x64 image this is the size of the latent vector)
            """
            
            self.encoder = Encoder(channel_in, ch=ch, z=z, dim=dim)
            self.decoder = Decoder(channel_in, ch=ch, z=z, dim=dim)
    
        def forward(self, x):
            encoding, mu, log_var = self.encoder(x)
            recon = self.decoder(encoding)
            return recon, mu, log_var
    
    
    
    
    reconstruction_function = nn.MSELoss(size_average=False)
    # reconstruction_function = FocalLoss2d()
    
    def loss_function(recon_x, x, mu, logvar):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        BCE = reconstruction_function(recon_x, x)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return BCE + KLD
    # model = VAE(channel_in=1,ch=128, z=64, dim=16)

    num_epochs = args.epochs 
    batch_size = args.batch_size
    # path='F:/data/auditory/USV_dataset/training/vae/'
    path=args.data_path
    # path='F:/data/auditory/USV_dataset/training/trainannot/'
    train_list=glob.glob(os.path.join(path,'*.png'))
    train_label=list(np.zeros(len(train_list)))
    
    dataloader = DataLoader(trainset(), batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    dim_idx=[8] 
    # dim_idx=[16] 
    
    for k in range(len(dim_idx)):
        print('processing embedding dimension: '+str(dim_idx[k]))
        model = VAE(channel_in=1,ch=args.ch, z=args.z, dim=dim_idx[k])
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
       #%
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)
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
                    # if batch_idx % 100 == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch,
                #     batch_idx * len(img),
                #     len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                #     loss.item() / len(img)))
                    t.set_description('Processing dim '+str(dim_idx[k])+' epoch: '+str(batch_idx+1)+' train loss: '+str(train_loss/(batch_idx+1)))
                    t.update(1)
                # print('====> Epoch: {} Average loss: {:.4f}, lr: {:.7f}'.format(
                #     epoch, loss.item() / len(img), optimizer.state_dict()['param_groups'][0]['lr']))
                # print(loss.item() / len(img))
                # if (loss.item() / len(img))<loss_record:
                #     torch.save(model, './cvae_model_0224_d2.pth')
                loss_record= (loss.item() / len(img))
                scheduler.step()
            torch.save(model.module.state_dict(), './models/cvae_model_1213_d'+str(dim_idx[k])+'.pth')
    

            
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)