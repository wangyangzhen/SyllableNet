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
#%%
torch.cuda.set_device(0)
num_epochs = 4
batch_size = 32
# path='F:/data/auditory/USV_dataset/training/vae/'
path='D:/WYZ/USV/VAE/training/'
# path='F:/data/auditory/USV_dataset/training/trainannot/'
train_list=glob.glob(path+'*.png')
train_label=list(np.zeros(len(train_list)))

dataloader = DataLoader(trainset(), batch_size=batch_size, shuffle=True)
loss_record=1000
res=[]
dim_idx=[64,16,32,64,128] 
# dim_idx=[16] 

for k in range(len(dim_idx)):
    print('processing embedding dimension: '+str(dim_idx[k]))
    model = VAE(channel_in=1,ch=128, z=64, dim=dim_idx[k])
    if torch.cuda.is_available():
        model.cuda()
   #%
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1, last_epoch=-1)
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
                t.set_description('Processing epoch: '+str(batch_idx+1)+' train loss: '+str(train_loss/(batch_idx+1)))
                t.update(1)
            # print('====> Epoch: {} Average loss: {:.4f}, lr: {:.7f}'.format(
            #     epoch, loss.item() / len(img), optimizer.state_dict()['param_groups'][0]['lr']))
            # print(loss.item() / len(img))
            # if (loss.item() / len(img))<loss_record:
            #     torch.save(model, './cvae_model_0224_d2.pth')
            loss_record= (loss.item() / len(img))
            scheduler.step()
    # torch.save(model, './cvae_model_'+str(loss_record)[0:3]+'.pth')
        # print('====> Epoch: {} Average loss: {:.4f}'.format(
        #     epoch, train_loss / len(dataloader.dataset)))
    #     if epoch % 10 == 0:
    #         save = to_img(recon_batch.cpu().data)
    #         save_image(save, './vae_img/image_{}.png'.format(epoch))


    #%
    torch.save(model, 'D:/WYZ/USV/VAE/models/cvae_model_1201_d'+str(dim_idx[k])+'.pth')
    #%%
    test_path='H:/data/bone_marrow/vae_dataset/'
    # test_path='D:/data/auditory/vae/syl_stim/'
    output='H:/data/bone_marrow/vae_output/'
    # test_path='D:/data/miniscope/USV/SYL/T0000020/seg/'
    # output='D:/data/miniscope/USV/SYL/T0000020/output/'
    test_list=glob.glob(test_path+'*.png')
    # model  = torch.load('D:/data/auditory/vae/model/cvae_model_0316_d16.pth')  
    embedding=np.array([])
    # model.load_state_dict(pretained_model)
    # if torch.cuda.is_available():
    #     model.cuda()
    test_label=list(np.zeros(len(test_list)))
    
    def ARLoader(path=test_path):
        img_pil =  Image.open(path)
        img_pil = img_pil.resize((k1,k1))
        img_tensor = train_preprocess(img_pil)
        return img_tensor
    
    
    class testset(Dataset):
        def __init__(self, loader=ARLoader):
            #定义好 image 的路径
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
        if batch_idx>200:
            break
        print(batch_idx)
        img, _ = data
        # img = img.view(img.size(0), -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        recon_batch, mu, logvar = model(img)
        encoder, mu, log_var = model.encoder(img)
        encoding=encoder.cpu().data.numpy()
        if embedding.shape[0]==0:
            embedding=np.append(embedding,encoding) 
            embedding=np.expand_dims(embedding, 0)
        else:
            embedding=np.concatenate([embedding,encoding],0) 
            # embedding=[embedding]
        if batch_idx==0:
            dimen=mu.cpu().data.numpy()
        else:
            dimen=np.append(dimen,mu.cpu().data.numpy(),0)
        x = recon_batch.cpu().data
        # x = 2 * (x - 0.5)
        x = x.clamp(0.2, 1)*2-1
        x = x.view(x.size(0), 3, k1, k1)
        n=batch_idx+1
        save_image(x, output+('%06d' % n)+'.png')

   #%% evulate results
    import cv2
    import glob
    test_path='H:/data/bone_marrow/vae_output/'
    output='H:/data/bone_marrow/vae_output/'
    # test_path='D:/data/miniscope/USV/SYL/T0000020/seg'
    # output='D:/data/miniscope/USV/SYL/T0000020/output'
    test_list=os.listdir((test_path))

    iou=[]
    mse=[]
    # image_s=np.zeros((260,32,32))
    for kk in range(len(test_list)):
        # k=44
        test_im=cv2.imread(test_path+test_list[kk])
        test_im=cv2.resize(test_im,(k1,k1))
        test_im = cv2.cvtColor(test_im, cv2.COLOR_GRAY2RGB)
        n= kk+1
        # cv2.imwrite(output+('%06d' %n+'_gt.png'),test_im)
        # test_im=cv2.cvtColor(test_im,cv2.COLOR_BGR2GRAY)/255
        # image_s[k,:,:]=cv2.resize(test_im,[32,32])
        pred_im=np.float32(cv2.imread(output+test_list[kk]))
        # pred_im=cv2.cvtColor(pred_im,cv2.COLOR_BGR2GRAY)/255
        
        pred_im=np.where(pred_im>0.3,1.,0.)
        d=(test_im*pred_im).sum()/((pred_im.sum()+test_im.sum())-(test_im*pred_im).sum())
        iou.append(d)
        mse.append( 0.5 * np.sum((test_im - pred_im)**2))
        # print(d)
        # if k==s:
        # break
    # print(k)
    print(np.mean(iou))
    print(np.mean(mse))
    # res.append([np.mean(iou),np.mean(mse)])
    # torch.cuda.empty_cache()
# cv2.imshow('asd',test_im)
# cv2.imshow('azxcsd',pred_im)

#%% 
import cv2
w = 128
r0=(-2, 2)
r1=(-2, 2)
n=10
img = np.zeros((n*w, n*w))
for i, y in enumerate(np.linspace(*r1, n)):
    for j, x in enumerate(np.linspace(*r0, n)):
        # z = torch.Tensor([[0, x, 0,0,0,0,0,0,0,0,0,0,0,0,0,0]]).cuda()
        ax = torch.Tensor([[x, y]]).cuda()
        # z = torch.Tensor([[0.0435,  0.3410, -0.2570, -0.4460,  0.1255,  0.1289,  0.3699, -0.3270,
        #         -0.1905, -0.2376, -0.3197, -0.5030, -0.5003,  0.3507,  0.0474,  0.1159]]).cuda()
        x_hat = model.decoder(ax)
        x_hat= x_hat.clamp(0.05, 1)
        x_hat= x_hat.to('cpu').detach().numpy()[0,0,:,:]
        
        x_hat = cv2.resize(x_hat,[w, w])
        img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
plt.imshow(img, extent=[*r0, *r1])

#%%
import cv2
w = 256
r1=(-2, 2)
n=10
k=6
for k in range(16):
    img = np.zeros((1*w, n*w))
    for i, x in enumerate(np.linspace(*r1, n)):
        z = torch.Tensor([[0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0]]).cuda()
        z[0][k]=x
        # z = torch.Tensor([[0.0435,  0.3410, -0.2570, -0.4460,  0.1255,  0.1289,  0.3699, -0.3270,
        #         -0.1905, -0.2376, -0.3197, -0.5030, -0.5003,  0.3507,  0.0474,  0.1159]]).cuda()
        x_hat = model.decoder(z)
        x_hat= x_hat.clamp(0.1, 1)
        x_hat= x_hat.to('cpu').detach().numpy()[0,0,:,:]
        
        x_hat = cv2.resize(x_hat,[w, w])
        img[0:w, i*w:(i+1)*w] = x_hat
    img=np.where(img<0.2,0.,img)
    
    plt.figure(figsize=(20,8))
    plt.imshow(img,extent=[*r1,*(0, 1)])
    plt.axis('off')
    plt.savefig(str(k)+'_d.png',bbox_inches='tight',pad_inches = -0.1)
    plt.close()
#%% image embedding

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
test_list=[]
data_path='D:/data/auditory/vae/cal_dis/'
for ff in os.listdir(data_path):
    test_list.extend(glob.glob(data_path+ff+'/*.png'))
#%%
from tensorboardX import SummaryWriter
import torchvision
import umap
emb_U = umap.UMAP(n_neighbors=5,
                      n_components=2,
                      min_dist=0.5,
                      metric='correlation',
                      random_state=16).fit_transform(embedding)
#%%
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
# plt.axis('off')   
plt.scatter(emb_U[:,0],emb_U[:,1],s=1)
#%% plot 
# c57=[3079,3081,3085,3087,3157,3166,3251]
# dba=[3070,3168,3249,3172,3248,3244]
c57=[]
dba=['c']
g=[]
r=[]
b=[]
for i in range(51800):
    if len(test_list[i])>1002:
        if i>2100 and i<2500:
            g.append(i)
        continue
    if test_list[i][29] in c57:
        r.append(i)
    elif test_list[i][29] in dba:
        b.append(i)
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
plt.scatter(emb_U[:,0],emb_U[:,1],s=1,color=[0.75,0.75,0.75])
plt.scatter(emb_U[r,0],emb_U[r,1],s=1,color=[1,0,0]) 
plt.scatter(emb_U[b,0],emb_U[b,1],s=1,color=[0,0,1])
# plt.scatter(emb_U[g,0],emb_U[g,1],s=1,color=[0,1,0])
plt.axis('off')   

    
#%% patch

def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        im = 255-cv2.imread(image)
        im = cv2.resize(im, (128, 128))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_f = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
imgs=[]

ax_0=[-5,-2.5]
step1=0.5
step2=0.4
emb_range=[ax_0[0],ax_0[0]+step1,ax_0[1],ax_0[1]+step2]
r_patch=np.array([])
for i in range(emb_U.shape[0]-5):
    if emb_U[i,0]<emb_range[1] and emb_U[i,0]>emb_range[0] and emb_U[i,1]>emb_range[2] and emb_U[i,1]<emb_range[3]:
        imgs.append(test_list[i])
        encoding=[emb_U[i,:]]
        if r_patch.shape[0]==0:
            r_patch=np.append(r_patch,encoding) 
            r_patch=np.expand_dims(r_patch, 0)
        else:
            r_patch=np.concatenate([r_patch,encoding],0) 
            
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
plt.axis('off')            
imscatter(r_patch[:, 0], r_patch[:, 1], imgs, zoom=0.5, ax=ax)  
#%% transparent
def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        im = 255-cv2.imread(image)
        # im = cv2.imread('19.png')
        height, width, channels = im.shape
        new_im = np.ones((height, width, 4)) * 255
        new_im[:, :, :3] = im
        for i in range(height):
            for j in range(width):
                if new_im[i, j, :3].tolist() == [255.0, 255.0, 255.0]:
                    new_im[i, j, :] = np.array([255.0, 255.0, 255.0, 0])

        im = cv2.resize(new_im, (128, 128))
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_f = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
imgs=[]

ax_0=[8.5,5.5]
step1=0.5
step2=0.5
emb_range=[ax_0[0],ax_0[0]+step1,ax_0[1],ax_0[1]+step2]
r_patch=np.array([])
for i in range(emb_U.shape[0]-5):
    if emb_U[i,0]<emb_range[1] and emb_U[i,0]>emb_range[0] and emb_U[i,1]>emb_range[2] and emb_U[i,1]<emb_range[3]:
        imgs.append(test_list[i])
        encoding=[emb_U[i,:]]
        if r_patch.shape[0]==0:
            r_patch=np.append(r_patch,encoding) 
            r_patch=np.expand_dims(r_patch, 0)
        else:
            r_patch=np.concatenate([r_patch,encoding],0) 
            
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
plt.axis('off')            
imscatter(r_patch[:, 0], r_patch[:, 1], imgs, zoom=0.75, ax=ax)  
#%%
filename = 'embedding_20220317.data'

f = open(filename, 'wb')
# 将变量存储到目标文件中区
pickle.dump({'embedding':embedding,'emb_U':emb_U}, f)
# 关闭文件
f.close() 
#%%
f=open('D:/data/auditory/vae/visualiza_20220316/embedding_20220317.data',"rb")  
data=pickle.load(f)  
f.close()
embedding=data['embedding']
emb_U=data['emb_U']
