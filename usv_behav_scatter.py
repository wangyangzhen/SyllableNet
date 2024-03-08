# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:16:29 2023

@author: Admion
"""
import pickle
import umap
import numpy as np
import matplotlib.pyplot as plt
import csv

pc = open(r'./FWPCA0.00_P100_en3_hid30_epoch230_svm2allAcc0.93_kmeansK2use-44_fromK1-20_K100_sequences.pkl','rb')
data = pickle.load(pc)
#%%
usv_emb={}
import os
ls=os.listdir('./')
for fn in ls:
    f=fn.split(".")[0]
    pc= open(fn,'rb')
    usv_emb[f]=pickle.load(pc)
#%%
c=0
emb_sum=np.empty([0,16])
USV_white_sum=np.empty([0])
USV_black_sum=np.empty([0])
for fn_r in usv_emb.keys():
    emb_sum=np.append(emb_sum,usv_emb[fn_r],0)
    fn=fn_r.replace("-", "_")
    for behv_fn in data.keys():
        if behv_fn[6:30]==fn and behv_fn[-6]=='e':
            ut=np.empty([0])
            behav_data=data[behv_fn]
            with open(fn_r+'.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                c=0
                for row in reader:
                    if c==0:
                        c+=1
                        continue
                    if float(row[1])*30<len(behav_data):
                        USV_white_sum=np.append(USV_white_sum,behav_data[np.int16(float(row[1])*30)])
                    else:
                        USV_white_sum=np.append(USV_white_sum,behav_data[-1])
        if behv_fn[6:30]==fn and behv_fn[-6]=='k':
            ut=np.empty([0])
            behav_data=data[behv_fn]
            with open(fn_r+'.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                c=0
                for row in reader:
                    if c==0:
                        c+=1
                        continue
                    if float(row[1])*30<len(behav_data):
                        USV_black_sum=np.append(USV_black_sum,behav_data[np.int16(float(row[1])*30)])
                    else:
                        USV_black_sum=np.append(USV_black_sum,behav_data[-1])



#%%
usv_t={}
ls=os.listdir('./')
for fn in ls:
    if fn[-3:]=='csv':
        ut=np.empty([0])
        with open(fn, 'r') as csvfile:
            reader = csv.reader(csvfile)
            c=0
            for row in reader:
                if c==0:
                    c+=1
                    continue
                ut=np.append(ut,np.array(float(row[1])))
        usv_t[fn[:-4]]=ut
#%%
emb_U = umap.UMAP(n_neighbors=5,
                      n_components=2,
                      min_dist=0.5,
                      metric='correlation',
                      random_state=16).fit_transform(emb_sum)
#%%  USV_white_sum
k=44
cm=USV_white_sum
cm=np.where(cm==k,0,1)
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
# plt.axis('off')   
plt.scatter(emb_U[:,0],emb_U[:,1],c=cm,s=2)
plt.colorbar()

#%% USV_black_sum
k=42
cm=USV_black_sum
cm=np.where(cm==k,0,1)
cm[1634:]=1
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
# plt.axis('off')   
plt.scatter(emb_U[:,0],emb_U[:,1],c=cm,s=2)
plt.colorbar()
