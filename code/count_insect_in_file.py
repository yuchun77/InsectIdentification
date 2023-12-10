# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:40:35 2021

@author: User
"""
import cv2
import os
import string
from sympy import *
import pandas as pd


category = []
f = open('D:\\category_bee.txt','r', encoding='utf-8')
for line in f.readlines():
    s = line.strip()
    category.append(s)

category_df = pd.DataFrame(category)
category_df[1] = 0
print(category_df)


pic_path = "D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_rgb_normalize_bee\\"
pic_list=[]
all_list_name = os.listdir(pic_path)
for x in range(len(all_list_name)):
    if ".JPG"in all_list_name[x] or ".jpg" in all_list_name[x]:
        pic_list.append(all_list_name[x])



for i in range(len(pic_list)):
    print("img："+pic_list[i])
    #   取得img_label
    for j in range(len(pic_list[i])):
        if '_' == pic_list[i][j]:
            label = pic_list[i][0:j]
            break
        
    index = category.index(label)
    category_df[1][index] += 1
    
    

print(category_df)


category_df.to_csv("count_rgb_normalize_bee.csv", encoding ="utf-8-sig", index = False, header = False)


