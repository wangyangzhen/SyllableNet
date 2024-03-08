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
from sklearn.decomposition import PCA
from tqdm import tqdm

k1=256
k2=1024
k3=4

train_preprocess = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize([k1,k1]),
    # transforms.Normalize((0.5), (0.5))
]) 

def ARLoader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((k1,k1))
    img_tensor = train_preprocess(img_pil)
    return img_tensor


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x





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
        mu = self.fc1(x.view(x.size(0), -1))
        log_var = self.fc2(x.view(x.size(0), -1))
        x = self.sample(mu, log_var)

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
        x = self.fc1(x)
        x1=x.view(x.size(0), self.z , int(k1/64), int(k1/64))
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)
        x7 = self.conv6(x6)
        x8 = self.conv7(x7)

        return x8, x1, x2, x4


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
        recon,f1,f2,f3 = self.decoder(encoding)
        return recon, mu, log_var,f1,f2,f3




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


def VAE_emb(model='',out_seg='',out_recon=''):

    test_path=out_seg
    output=out_recon
    c=np.empty([0])
    f1_res={};
    f2_res={};
    emb_res={};
    f_list=os.listdir(test_path)
    with tqdm(total=len(f_list)) as tq: 
        for fn in f_list:
            tq.set_description('Embedding USV: '+fn)
            try:
                os.mkdir(os.path.join(out_recon,fn))
            except:
                pass
            test_list=glob.glob(os.path.join(test_path,fn,'*.png'))
            embedding=np.zeros([len(test_list),32],dtype='float32')
            f1_sum=np.zeros([len(test_list),1024],dtype='float32')
            f2_sum=np.zeros([len(test_list),65536],dtype='float32')
            c=0
            test_label=list(np.zeros(len(test_list)))
            
            
            class testset(Dataset):
                def __init__(self, loader=ARLoader):
                    self.images = test_list
                    self.loader = loader
                    self.target = test_label
                def __getitem__(self, index):
                    fn = self.images[index]
                    img = self.loader(fn)
                    target = self.target[index]
                    return img,target
                
                def __len__(self):
                    return len(self.images)
         #%         
            testloader = DataLoader(testset(), batch_size=1, shuffle=False)
            for batch_idx, data in enumerate(testloader):
                img, _ = data
                img = Variable(img)
                if torch.cuda.is_available():
                    img = img.cuda()
                recon_batch, mu, logvar,f1, f2, f3 = model(img)
                x = recon_batch.cpu().data
                x = x.clamp(0.2, 1)*2-1
                x = x.view(x.size(0), 1, k1, k1)
                n=batch_idx+1
                save_image(x, os.path.join(output,fn,('%06d' % n)+'.png'))
                encoder, mu, log_var = model.encoder(img)
                embedding[c,:]=encoder.cpu().data.numpy()
                f1_sum[c,:]=f1.cpu().detach().numpy().flatten()
                f2_sum[c,:]=f2.cpu().detach().numpy().flatten()
                
                c+=1
            emb_res[fn]=embedding;
            f1_res[fn]=f1_sum;
            f2_res[fn]=f2_sum;
            tq.update(1)

    #%
    
    # pickle.dump(f1_res, file=open('f1_res.pkl', 'wb+'))
    # pickle.dump(f2_res, file=open('f2_res.pkl', 'wb+'))
    # pickle.dump(emb_res, file=open('emb_res.pkl', 'wb+'))
    
    
    
    pickle.dump(emb_res, file=open(os.path.join(output,'L0_res.pkl'), 'wb+'))
    
    #%%
    f0_sum=np.empty([0,1024],dtype='float16')
    # f0_sum=np.empty([0,65536])
    for fn in f1_res:
        f0_sum=np.append(f0_sum,np.float16(f1_res[fn]),0)
    pca = PCA(n_components=64)   
    pca.fit(f0_sum)
    newX=pca.fit_transform(f0_sum)
    L1_res={}
    c=0
    for fn in f2_res:
        L1_res[fn]=newX[c:c+f1_res[fn].shape[0],:]
        c+=f1_res[fn].shape[0]
    pickle.dump(L1_res, file=open(os.path.join(output,'L1_res.pkl'), 'wb+'))
    
    
    #%%             
    # f0_sum=np.empty([0,65536],dtype='float16')
    # for fn in f2_res:
    #     f0_sum=np.append(f0_sum,np.float16(f2_res[fn]),0)
    # pca = PCA(n_components=64)   
    # pca.fit(f0_sum)
    # newX=pca.fit_transform(f0_sum)
    # L2_res={}
    # c=0
    # for fn in f2_res:
    #     L2_res[fn]=newX[c:c+f2_res[fn].shape[0],:]
    #     c+=f2_res[fn].shape[0]
    # pickle.dump(L2_res, file=open(os.path.join(output,'L2_res.pkl'), 'wb+'))