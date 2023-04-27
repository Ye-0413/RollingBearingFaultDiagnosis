#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:16:46 2022

@author: youngj
"""

import matplotlib.pyplot as plt
import pywt
import numpy as np
import scipy.io as scio
import os

filepath = '/home/zy/jy_form_OsX/machine/CaseWesternReserveUniversityData-master/normal_2_99.mat'
dataFile = filepath
data = scio.loadmat(dataFile)
dict_data=scio.loadmat(filepath)
data_ = np.array(data['X099_DE_time'])

name=j=int(0)
for n in range(238):
    a = []
    name=int(name)
    for i in range(1024):
       a.append(data_[j])
       j+=1;
       name+=1;
    t=np.arange(len(a))/12000;
    scales = np.arange(1,5)
    coef,freqs = pywt.cwt(a,scales,'gaus1')

   #Ploting Sclogram

    plt.figure(figsize=(15,10));
    plt.imshow(abs(coef),interpolation='nearest',cmap='rainbow_r',
    aspect='auto',vmax=0.1,vmin=-0.1)
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(0,3,0.1))
    plt.xticks()
    del a[:];
    figure_save_path = "/home/zy/jy_form_OsX/machine/CWT_picture/normal_2_99"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
    name=str(name)
    plt.savefig(os.path.join(figure_save_path, "normal_99_"+name))
    plt.show()
    print(j);


 