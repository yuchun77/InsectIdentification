# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:45:12 2021

@author: User
"""

import cv2
import os
import string
from sympy import *
import pandas as pd
import numpy as np

df = pd.read_csv('D:\\code\\count_insect_already_del_rotate.csv', header=None)
# print(df)


pic_path = "D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_cut_clahe3\\"
pic_list=[]
all_list_name = os.listdir(pic_path)
for x in range(len(all_list_name)):
    if ".JPG"in all_list_name[x] or ".jpg" in all_list_name[x]:
        pic_list.append(all_list_name[x])
        # print(pic_list)
# print(df)


img = []
for i in range(len(pic_list)):
    outfile=cv2.imdecode(np.fromfile("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_cut_clahe3\\"+pic_list[i],dtype=np.uint8),cv2.IMREAD_COLOR)
    img.append(outfile)

# print(img)


label = []
for i in range(len(pic_list)):
#   取得img_label
    for j in range(len(pic_list[i])):
        if '_' == pic_list[i][j]:
            temp = pic_list[i][0:j]
            label.append(temp)
            break


d = {'label': label, 'img': img, 'name':pic_list}
img_df = pd.DataFrame(data = d)
# print(img_df)
# print(df)
    

for i in range(len(df[0])):
    img_filter = (img_df["label"] == df[0][i])
    img_f_df = img_df[img_filter]
    img_f_df.index = range(len(img_f_df)) #從0開始編號
    # label_name = df[0][i]
    num = df[1][i]
    train_num = int(num*0.8)
    # print(train_num)
    test_num = num - train_num
    # print(img_f_df)
    for j in range(0, train_num):
        cv2.imencode('.jpg', img_f_df["img"][j])[1].tofile("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_cut_clahe3_train\\"+img_f_df["name"][j])
        print(img_f_df["name"][j])
    for k in range(num-1, train_num-1, -1):
        cv2.imencode('.jpg', img_f_df["img"][k])[1].tofile("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_cut_clahe3_test\\"+img_f_df["name"][k])
        print(img_f_df["name"][k])
        
# print(len(df[1]))