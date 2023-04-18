#!/usr/bin/env python3
#-*- coding: utf-8 -*-

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

