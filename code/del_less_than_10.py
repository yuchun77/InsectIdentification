# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:15:05 2021

@author: User
"""

import numpy as np
import cv2
import os
import pandas as pd


statistic_df = pd.read_csv('D:\\code\\count_insect_org.csv', header=None)

df_filter = statistic_df[1]<10
to_be_del_df = statistic_df[df_filter]
to_be_del_df.index = range(len(to_be_del_df))
print(to_be_del_df)

to_be_del_list = list(to_be_del_df[0])
to_be_del_list.append('無法辨識')


pic_path = "D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_already_del_rotate\\"
pic_list=[]
all_list_name = os.listdir(pic_path)
for x in range(len(all_list_name)):
    if ".JPG"in all_list_name[x] or ".jpg" in all_list_name[x]:
        pic_list.append(all_list_name[x])
        

for i in range(len(pic_list)):
    img = cv2.imdecode(np.fromfile(pic_path + pic_list[i],dtype=np.uint8),-1)
    label = ""
#   取得img_label
    for j in range(len(pic_list[i])):
        if '_' == pic_list[i][j]:
            label = pic_list[i][0:j]
            break
    
    if label in to_be_del_list:
        print("del_label:", label)
        print("del_photo:", pic_list[i])
        os.remove("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_already_del_rotate\\"+pic_list[i])

