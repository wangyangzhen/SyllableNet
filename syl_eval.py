# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:27:00 2020

@author: yangzhen
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import scipy.io as scio


# helper function for data visualization
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


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['back','cell']
    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
    #        'tree', 'signsymbol', 'fence', 'car', 
    #        'pedestrian', 'bicyclist', 'unlabelled']
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        # extract certain classes from mask (e.g. mitos)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)



def get_training_augmentation():
    train_transform = [
        albu.Resize(height=512, width=512, p=1),

    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(512, 512,interpolation = 1,always_apply = False,p = 1 ),
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

best_model = torch.load('./res_20220216.pth')
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['cell']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing=get_preprocessing(preprocessing_fn)
#%%
from scipy import signal
import wave  
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import csv

data_folder='D:/data/auditory/USVSEG_dataset/evulate/'
output_path='D:/data/auditory/USVSEG_dataset/evulate/image4/'
threshold=0.9
noverlap=128
nperseg=256
overlap_t=0.11
for file in os.listdir(data_folder):
    fn,fmat=os.path.splitext(file)
    if fmat!='.wav':
        continue
    try:
        os.mkdir(os.path.join(output_path,fn))
    except:
        pass
    f = wave.open(data_folder+file,"rb")
    params = f.getparams()  
    nchannels, sampwidth, fs, nframes = params[:4] 
    str_data  = f.readframes(nframes)  
    f.close()
    wave_data = np.fromstring(str_data,dtype = np.short)
    #%
    # nframes=fs*3
    spec_n=np.floor(nframes/fs/overlap_t)-1
    audio_sig_r=np.zeros(int((spec_n+1)*nperseg))
    for i in range(int(spec_n)):
        f, t, Zxx = signal.stft(wave_data[round(i*overlap_t*fs):round((i*overlap_t+overlap_t*2)*fs)], fs, nperseg=nperseg,noverlap=noverlap, nfft=1024)
        P=np.flip(Zxx,0)
        P=abs(P[0:430])
        P=np.where(P==0,1,P)
        A = 10*np.log(P);
        A=(A-A.min())/(A.max()-A.min())
        A=np.uint8(A*255)
        # cv2.imwrite(os.path.join(output_path,fn,str('%05d' %i)+'.png'),A)
        image = np.expand_dims(A, axis=2) 
        image=np.concatenate((image,image,image),2)
        # image=np.stack(image,image,2)
        augmentation=get_validation_augmentation()
        sample = augmentation(image=image)
        image= sample['image']
        sample = preprocessing(image=image)
        image = sample['image']
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy())
        pr_mask[-100:-1,:]=0
        mask_scan=np.max(pr_mask,0)
        # print(pr_mask.max())
        # pr_mask=np.where(pr_mask>0.5,1,0)
        # if np.max(pr_mask)>0:
        #     cv2.imwrite('D:/data/auditory/USVSEG_dataset/evulate/image4/'+str('%05d' %i)+'.png',A)
        #     cv2.imwrite('D:/data/auditory/USVSEG_dataset/evulate/image4/'+str('%05d' %i)+'_m.png',pr_mask*255)
        a=round(i*nperseg)
        b=round(i*nperseg)+512
        if a>0 and b<len(audio_sig_r):
            audio_sig_r[a:b]=audio_sig_r[a:b]+mask_scan
    
    audio_sig=np.where(audio_sig_r>threshold,1,0)
    USV_t=[]
    st3=0
    st2=0
    for i in range(len(audio_sig)-1):
        if audio_sig[i]==0 and audio_sig[i+1]==1 and i+1-st2>50:
            if st2==0:
                st1=st3
                st3=i+1
                continue
            st1=st3
            st3=i+1
            if st1>512:
                USV_t.append([st1,st2])
        if audio_sig[i]==1 and audio_sig[i+1]==0:
            st2=i
    USV_t.append([st3,st2])
    USV_t=np.array(USV_t)/512*overlap_t*2
    USV_res=[]
    for i in range(len(USV_t)):
        ut=int(np.mean(USV_t[i,:])*fs)
        f, t, Zxx = signal.stft(wave_data[ut-round(overlap_t*fs):ut+round(overlap_t*fs)], fs, nperseg=nperseg,noverlap=noverlap, nfft=1024)
        P=np.flip(Zxx,0)
        P=abs(P[0:430])
        P=np.where(P==0,1,P)
        A = 10*np.log(P);
        A=(A-A.min())/(A.max()-A.min())
        A=np.uint8(A*255)
        
        image = np.expand_dims(A, axis=2) 
        image=np.concatenate((image,image,image),2)
        # image=np.stack(image,image,2)
        augmentation=get_validation_augmentation()
        sample = augmentation(image=image)
        image= sample['image']
        sample = preprocessing(image=image)
        image = sample['image']
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy())
        
        # pr_mask=(pr_mask-pr_mask.min())/(pr_mask.max()-pr_mask.min())
        pr_mask=np.where(pr_mask>threshold,1,0)
        if np.max(pr_mask)>threshold:
            if not(len(USV_res)):
                USV_res=USV_t[i]
                USV_res=np.expand_dims(USV_res,0)
            else:
                USV_res=np.concatenate((USV_res,np.expand_dims(USV_t[i],0)),0)
            pr_mask=np.uint8(pr_mask*255)
            cv2.imwrite(os.path.join(output_path,fn,str('%05d' %i)+'.png'),A)
            cv2.imwrite(os.path.join(output_path,fn,str('%05d' %i)+'_m.png'),pr_mask)
    headers = ['start','end']
    rows = np.float32(USV_res)
    with open(os.path.join(output_path,fn,'output.csv'),'w',newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
    # break
    #%%
plt.plot(audio_sig_r)
plt.figure()
plt.plot(wave_data)