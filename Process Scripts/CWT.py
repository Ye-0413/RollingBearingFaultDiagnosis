#!/usr/bin/env python3
#-*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pywt
import numpy as np
import scipy.io as scio
import os

filepath = '/Users/youngj/Local/Project/2022_Rolling_Bearing_Fault_Diagnosis/CaseWesternReserveUniversityData-master/48k_Drive_End_OR021@12_3_265.mat'
dataFile = filepath
data = scio.loadmat(dataFile)
dict_data=scio.loadmat(filepath)
data_ = np.array(data['X265_DE_time'])

sig = data['X265_DE_time'][:1024]
scales = np.arange(1,5)
wavelet='morl'
coef,freqs = pywt.cwt(sig,scales,wavelet)

   #Ploting Sclogram

plt.figure(figsize=(15,10));
mag = np.abs(coef)
plt.imshow(mag,interpolation='nearest',cmap='rainbow_r',
aspect='auto',vmax=0.1,vmin=-0.1)
plt.gca().invert_yaxis()
plt.yticks(np.arange(0,3,0.1))
plt.xticks()
figure_save_path = "/Users/youngj/Local/Project/2022_Rolling_Bearing_Fault_Diagnosis/CWT"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建

plt.savefig(os.path.join(figure_save_path, wavelet + '.png'))
plt.show()

'''
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dataFile = r'/home/zy/jy_form_OsX/machine/CaseWesternReserveUniversityData-master/48k_Drive_End_B007_2_124.mat' 
data = scio.loadmat(dataFile)
print(type(data))


a=data['X124_DE_time']

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

new_im = MatrixToImage(a)
plt.imshow(a, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()
new_im.save('data_2.bmp') 
'''
'''
import numpy as np
import pywt
import scipy.io
import matplotlib.pyplot as plt

# Load signal data from a .mat file
data = scipy.io.loadmat('/Users/youngj/Local/Project/2022_Rolling_Bearing_Fault_Diagnosis/CaseWesternReserveUniversityData-master/48k_Drive_End_OR021@12_3_265.mat')
sig = data['X265_DE_time'][:1024]

# Define parameters for the CWT
scales = np.arange(1, 100)
wavelet = 'morl'

# Perform the CWT
coeffs, freqs = pywt.cwt(sig, scales, wavelet)

# Plot the results
plt.imshow(coeffs, extent=[0, len(sig), 1, 100], cmap=plt.cm.gray, aspect='auto', vmax=abs(coeffs).max(), vmin=-abs(coeffs).max())
plt.show()
plt.imshow(coeffs, extent=[0, len(sig), 1, 100], cmap='plt.cm.gray', aspect='auto', vmax=abs(coeffs).max(), vmin=-abs(coeffs).max())
plt.show()

'''