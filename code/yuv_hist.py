#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:25:01 2021

@author: chenyuchun
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


img = cv2.imread("DSC_9463.JPG")
nr, nc = img.shape[:2]
ycrcb = cv2.cvtColor( img, cv2.COLOR_BGR2YCrCb )

y = ycrcb[:,:,0]
Cr = ycrcb[:,:,1]
Cb = ycrcb[:,:,2]


y_flatten = y.flatten()

ymin = y.min()
ymax = y.max()


light = int((y_flatten.size) * 0.1) #10%有幾個
y_sorted = sorted(y_flatten, reverse=True)

light_num = y_sorted[light]  #10%的那個亮度值
for i in range(y_flatten.size):
    if y_flatten[i] >= light_num:
        y_flatten[i] = ymax

s = pd.Series(y_flatten)
x = s.value_counts().head(20)
print(x)


y_flatten_img = y_flatten.reshape(3264, 4928)

#Normalize
y_norm = ((((y_flatten_img - ymin)/(ymax - ymin))**1.5)*255)
y_normint = y_norm.astype('uint8')


ycrcb[:,:,0] = y_normint

img1 = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imwrite("aa.JPG", img1)


'''

#Normalize
y_norm = (((y - ymin)/(ymax - ymin))*255)
y_normint = y_norm.astype('uint8')

yk = y_normint.flatten()

# print(yk)
# print(yk.max())





yk_img = yk.reshape(3264, 4928)


def histogram( f ):
    hist = cv2.calcHist( [f], [0], None, [256], [0,256])
    plt.bar(range(1,257), hist.flatten() )
# 	else:
# 		color = ( 'b', 'g', 'r' )
# 		for i, col in enumerate( color ):
# 			hist = cv2.calcHist( f, [i], None, [256], [0,256] )
# 			plt.plot( hist, color = col )
    plt.xlim( [0, 256] )
    plt.xlabel( "Intensity" )
    plt.ylabel( "#Intensities" )
    plt.show( )

histogram(yk_img)



ycrcb[:,:,0] = yk_img

img1 = cv2.cvtColor( ycrcb, cv2.COLOR_YCrCb2BGR )

cv2.imwrite("aa.JPG", img1)
'''