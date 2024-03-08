# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:02:51 2022

@author: yangzhen
"""

import argparse
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm 
from vae_pred import *

def get_args_parser():
    parser = argparse.ArgumentParser('VAE training', add_help=False)
    parser.add_argument('--data_path', default='D:/data/auditory/USV_MP4-toXiongweiGroup/VAE/training', type=str,
                        help='dataset path')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--model_path', default='./models/cvae_model_0316_d16.pth', type=str,
                        help='dataset path')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--ch', default=128, type=int)
    parser.add_argument('--z', default=64, type=int)
    parser.add_argument('--dim', default=32, type=int)
    return parser

def main(args):
    k1=256

    train_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([k1,k1]),
        # transforms.Normalize((0.5), (0.5))
    ]) 
    def ARLoader(path):
        img_pil =  Image.open(path)
        img_tensor = train_preprocess(img_pil)
        return img_tensor
    
    
    class testset(Dataset):
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

    reconstruction_function = nn.MSELoss(size_average=False)
    

    def loss_function(recon_x, x, mu, logvar):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        BCE = reconstruction_function(recon_x, x)  # mse loss
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return BCE + KLD
 
    batch_size = args.batch_size
    path=args.data_path
    train_list=glob.glob(os.path.join(path,'*.png'))
    train_label=list(np.zeros(len(train_list)))
    
    dataloader = DataLoader(testset(), batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    model = torch.load(args.model_path) 
    # model = VAE(channel_in=1, ch=args.ch, z=args.z, dim=args.dim)
    # state_dict = torch.load(args.model_path)
    # model.load_state_dict(state_dict)
    if torch.cuda.is_available():
          model.cuda()
    model.eval()
    train_loss = 0
    with tqdm(total=len(dataloader)) as t: 
        for batch_idx, data in enumerate(dataloader):
            img, _ = data
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
            recon_batch, mu, logvar = model(img)
            loss = loss_function(recon_batch, img, mu, logvar)
            train_loss += loss.item()
            t.set_description('Processing '+ args.model_path +': train loss: '+str(train_loss/(batch_idx+1)/batch_size))
            t.update(1)

            
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)