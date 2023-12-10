# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:48:19 2021

@author: vicky
"""

import cv2
import os
import string
from sympy import *
import pandas as pd


category = []
f = open('D:\\category.txt','r', encoding='utf-8')
for line in f.readlines():
    s = line.strip()
    category.append(s)

df = pd.DataFrame(category)
#print(df)


def yolo2voc(img_w, img_h, gt):
    xmin = Symbol('xmin')
    xmax = Symbol('xmax')
    ymin = Symbol('ymin')
    ymax = Symbol('ymax')
    
    f1 = ((xmin + (xmax-xmin)/2) / img_w) - float(gt[0])
    f2 = ((ymin + (ymax-ymin)/2) / img_h) - float(gt[1])
    f3 = ((xmax-xmin) / img_w) - float(gt[2])
    f4 = ((ymax-ymin) / img_h) - float(gt[3])
    
    sol = solve((f1, f2, f3, f4), xmin, xmax, ymin, ymax)
    return sol[xmin], sol[xmax], sol[ymin], sol[ymax]


pic_path = "D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016\\"
pic_list=[]
all_list_name = os.listdir(pic_path)
for x in range(len(all_list_name)):
    if ".JPG"in all_list_name[x] or ".jpg" in all_list_name[x]:
        pic_list.append(all_list_name[x])

gt_path = 'D:\\new_label\\'
gt_list = os.listdir(gt_path)


for i in range(len(pic_list)):
    print("img："+pic_list[i])
    img = cv2.imread(pic_path + pic_list[i])
    gt = pic_list[i][:-4]+".txt"
    with open(gt_path + gt, 'r', encoding="utf-8") as f:
        for j, line in enumerate(f.readlines()):
            s = line.strip().split(' ')
            xmin, xmax, ymin, ymax = yolo2voc(img.shape[1], img.shape[0], s[1:])
            ROI = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            cv2.imencode('.jpg', ROI)[1].tofile("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_already_del_rotate\\"+df[0][int(s[0])]+"_"+str(j)+"_"+pic_list[i][:-4]+".jpg")  #蜚蠊瘦蜂_0_DSC_9682.jpg




#s = line.lstrip(string.digits).strip().split(' ')

#cv2.imwrite("我//h.jpg", frame) #該方法不成功 
#cv2.imencode('.jpg', frame)[1].tofile('我/9.jpg') #正確方法 



    
    
    