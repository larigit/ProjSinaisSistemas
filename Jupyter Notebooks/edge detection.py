#!/usr/bin/env python
# coding: utf-8

def generate_fir():
    fir =np.zeros(16)
    for i in range(-8,9):
        if i==0:
            fir[8] = 1
        else:
            if i==8:
                fir[15]=(1/i)
            else:
                fir[i+8]= (1/i)
    return fir

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('2.jpg',0)

fir = generate_fir() #gerando kernel

img1, img2 = filtro_separavel(fir,img) #aplicando filtro na linha e na coluna

mag_grad = np.sqrt(np.square(img1) + np.square(img2)) #obtendo o modulo dos vetores de derivada
mag_grad *= 255 / mag_grad.max()

plt.imshow(mag_grad.astype(int), cmap = "gray")
