# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:24:20 2021

@author: User
"""

import numpy as np
import cv2


def RGB_gamma_correction(f, gamma):
    g = f.copy()
    nr, nc = f.shape[:2]
    c = 255.0/(255.0 ** gamma)
    table = np.zeros(256)
    #print(table.shape)
    for i in range(256):
        table[i] = round(i ** gamma *c, 0)
    for x in range(nr):
        for y in range(nc):
            g[x,y, 0]  = table[f[x,y,0]]
            
    print(f[3,2,0])
    print(g[3,2,0])
    return g


def gamma_correction(img, gamma):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    g = yuv.copy()
    nr, nc = yuv.shape[:2]
    c = 255.0/(255.0 ** gamma)
    table = np.zeros(256)
    #print(table.shape)
    for i in range(256):
        table[i] = round(i ** gamma *c, 0)
    for x in range(nr):
        for y in range(nc):
            g[x,y, 0]  = table[yuv[x,y,0]]
            
    print(yuv[3,2,0])
    print(g[3,2,0])
    
    img_gamma = cv2.cvtColor(g, cv2.COLOR_YUV2BGR)
    return img_gamma


if __name__ == '__main__':
    img = cv2.imread("DSC_9463.JPG")
    
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    gammacor = RGB_gamma_correction(yuv, 1.2)
    img_gamma = cv2.cvtColor(gammacor, cv2.COLOR_YUV2BGR)
    cv2.imwrite("DSC_9463_gamma_12.JPG", img_gamma)
    
    img_gamma_2 = gamma_correction(img, 1.2)
    cv2.imwrite("DSC_9463_gamma_13.JPG", img_gamma_2)
    
    
    
    