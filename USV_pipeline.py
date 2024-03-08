# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:15:05 2023

@author: Admion
"""
import os
from syl_pred import detect_USV
from vae_pred import *
# from vae_pred import VAE_emb
import time
import torch




data_path='D:/data/auditory/USV_MP4-toXiongweiGroup/low_fq/'
out_path='D:/data/auditory/USV_MP4-toXiongweiGroup/low_fq_out/'

try:
    os.mkdir(out_path)
    os.mkdir(os.path.join(out_path,'detect'))
    os.mkdir(os.path.join(out_path,'seg'))
    os.mkdir(os.path.join(out_path,'recon'))
except:
    pass

model_seg= torch.load('./res_20231129.pth')
model_recon= torch.load('D:/data/auditory/vae/model/cvae_model_1201_d32.pth')
if torch.cuda.is_available():
    model_recon.cuda()
    
out_detect=os.path.join(out_path,'detect')
out_seg=os.path.join(out_path,'seg')
out_recon=os.path.join(out_path+'recon')
try:
    os.mkdir(out_detect)
    os.mkdir(out_seg)
    os.mkdir(out_recon)        
except:
    pass

detect_USV(model=model_seg, fn_path=data_path,out_detect=out_detect,out_seg=out_seg,usv_interval=0.05)
# VAE_emb(model=model_recon,out_seg=out_seg,out_recon=out_recon)
print('Done.')
