# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:27:00 2023

@author: yangzhen
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
from scipy import signal
import wave  
import csv
from scipy.signal import savgol_filter
from tqdm import tqdm

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()



def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(256, 256,interpolation = 1,always_apply = False,p = 1 ),
        # albu.RandomCrop(height=512, width=512, always_apply=True)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def detect_USV(model='', fn_path='',out_detect='',out_seg='',usv_interval=0.05):
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['syllable']
    ACTIVATION = 'syllable' # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing=get_preprocessing(preprocessing_fn)

    threshold=0.5
    noverlap=256
    nperseg=512
    overlap_t=0.15
    f_list=os.listdir(fn_path)
    with tqdm(total=len(f_list)) as tq: 
        for file in f_list:
            tq.set_description('Detecting USV: '+file)
            fn,fmat=os.path.splitext(file)
            if fmat!='.WAV':
                continue
            try:
                os.mkdir(os.path.join(out_detect,fn))
                os.mkdir(os.path.join(out_seg,fn))
            except:
                pass
            f = wave.open(os.path.join(fn_path,file),"rb")
            params = f.getparams()
            nchannels, sampwidth, fs, nframes = params[:4]
            str_data  = f.readframes(nframes)  
            f.close()
            wave_data = np.fromstring(str_data,dtype = np.short)
            spec_n=np.floor(nframes/fs/overlap_t/2)-1
            audio_sig_r=np.zeros(int((spec_n+1)*noverlap))
            for i in range(int(spec_n)):
                f, t, Zxx = signal.stft(wave_data[round(i*overlap_t*2*fs):round((i*overlap_t*2+overlap_t*2)*fs)], fs, nperseg=nperseg,noverlap=noverlap, nfft=768)
                P=np.flip(Zxx,0)
                P=abs(P[40:335,:])
                P=np.where(P==0,1,P)
                A = 10*np.log(P);
                A=(A-A.min())/(A.max()-A.min())
                A=np.uint8(A*255)
                image = np.expand_dims(A, axis=2) 
                image=np.concatenate((image,image,image),2)
                augmentation=get_validation_augmentation()
                sample = augmentation(image=image)
                image= sample['image']
                sample = preprocessing(image=image)
                image = sample['image']
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                pr_mask = model.predict(x_tensor)
                pr_mask = (pr_mask.squeeze().cpu().numpy())
                mask_scan=np.max(pr_mask,0)
                pr_mask=np.where(pr_mask>threshold,1,0)
                a=round(i*noverlap)
                b=round(i*noverlap)+256
                if a>0 and b<len(audio_sig_r):
                    audio_sig_r[a:b]=audio_sig_r[a:b]+mask_scan
            audio_sig=np.where(audio_sig_r>threshold,1,0)
            USV_t_raw=[]
            st3=0
            st2=0
            interval_th=usv_interval/overlap_t/2*295
            for i in range(len(audio_sig)-1):
                if audio_sig[i]==0 and audio_sig[i+1]==1 and i+1-st2>interval_th:
                    if st2==0:
                        st1=st3
                        st3=i+1
                        continue
                    st1=st3
                    st3=i+1
                    if st1>512:
                        USV_t_raw.append([st1,st2])
                if audio_sig[i]==1 and audio_sig[i+1]==0:
                    st2=i+1
            USV_t_raw.append([st3,st2])
            USV_t=np.array(USV_t_raw)/256*overlap_t*2
            USV_res=[]
            syl_idx=1
            for i in range(len(USV_t)):
                ut=int(np.mean(USV_t[i,:])*fs)
                # usv_sig=wave_data.copy()
                f, t, Zxx = signal.stft(wave_data[ut-round(overlap_t*fs):ut+round(overlap_t*fs)], fs, nperseg=nperseg,noverlap=noverlap, nfft=768)
                P=np.flip(Zxx,0)
                P=abs(P[40:335,:])
                P=np.where(P==0,1,P)
                A = 10*np.log(P);
                A=(A-A.min())/(A.max()-A.min())
                A=np.uint8(A*255)
                image = np.expand_dims(A, axis=2) 
                image=np.concatenate((image,image,image),2)
                augmentation=get_validation_augmentation()
                sample = augmentation(image=image)
                image= sample['image']
                sample = preprocessing(image=image)
                image = sample['image']
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                pr_mask = model.predict(x_tensor)
                pr_mask = (pr_mask.squeeze().cpu().numpy())
                pr_mask=np.where(pr_mask>threshold,1,0)
                USV_inten=np.max(pr_mask,0)
                if USV_inten.sum()==0:
                    continue
                USV_idx=np.where(USV_inten>0)[0]
                USV_cen=USV_idx[np.argmin(abs(USV_idx-127))]
                interv=0
                for k in range(USV_cen):
                    if USV_inten[USV_cen-k]==0:
                        interv+=1
                    else:
                        interv=0
                    if interv>interval_th:
                        pr_mask[:,0:USV_cen-k]=0
                for k in range(255-USV_cen):
                    if USV_inten[USV_cen+k]==0:
                        interv+=1
                    else:
                        interv=0
                    if interv>interval_th:
                        pr_mask[:,USV_cen+k:]=0
                if pr_mask.sum()<10 or np.mean(pr_mask[:,108:148])==0 or USV_t[i][1]-USV_t[i][0]<0.005:
                    continue
                pr_mask=np.uint8(pr_mask*255)
                fq_idx = savgol_filter(np.sum(pr_mask,1), window_length=5, polyorder=3)
                fq_idx_max=np.argmax(fq_idx)
                fq_dis=np.sum(pr_mask,1)/np.sum(pr_mask)
                fq_dis=np.array([np.sum(-fq_dis*np.log2(fq_dis+1e-10))])
                mf=((256-fq_idx_max)/256*235+80)/385*125
                if not(len(USV_res)):
                    c=0
                    USV_res=np.concatenate((np.array([c]),USV_t[i],np.array([mf]),np.array([USV_t[i,1]-USV_t[i,0]]),fq_dis),0)
                    USV_res=np.expand_dims(USV_res,0)
                else:
                    c+=1
                    USV_s=np.concatenate((np.array([c]),USV_t[i],np.array([mf]),np.array([USV_t[i,1]-USV_t[i,0]]),fq_dis),0)
                    USV_res=np.concatenate((USV_res,np.expand_dims(USV_s,0)),0)
                cv2.imwrite(os.path.join(out_detect,fn,str('%06d' %syl_idx)+'.png'),A)
                cv2.imwrite(os.path.join(out_detect,fn,str('%06d' %syl_idx)+'_m.png'),pr_mask)
                cv2.imwrite(os.path.join(out_seg,fn,str('%06d' %syl_idx)+'_m.png'),pr_mask)
                # cv2.imwrite(os.path.join('D:/WYZ/USV/VAE/training',fn+str('%05d' %syl_idx)+'.png'),pr_mask)
                syl_idx+=1
            headers = ['idx','start','end','main_freq','duration','distribution']
            rows = np.float32(USV_res)
            with open(os.path.join(out_detect,os.path.splitext(file)[0]+'.csv'),'w',newline='')as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                f_csv.writerows(rows)
            tq.update(1)
