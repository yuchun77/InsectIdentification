# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:58:13 2021

@author: User
"""

import numpy as np
import cv2
import os
import pandas as pd


#statistic = [["cat", 7], ["dog", 7], ["potato", 4], ["paint2", 2], ["rabbit", 1]]
#statistic_df = pd.DataFrame(statistic)
statistic_df = pd.read_csv('D:\\code\\count_insect_already_del.csv', header=None)
df_filter = statistic_df[1]<40
statistic_f_df = statistic_df[df_filter]
statistic_f_df.index = range(len(statistic_f_df))
print(statistic_f_df)
to_do_list = list(statistic_f_df[0])


def rotate_image(mat, angle, name): 
    height, width = img.shape[:2]
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape 

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.) 

    #多留黑
    # rotation calculates the cos and sin, taking absolutes of those. 
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1]) 

    # find the new width and height bounds 
    bound_w = int(height * abs_sin + width * abs_cos) 
    bound_h = int(height * abs_cos + width * abs_sin) 

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates 
    rotation_mat[0, 2] += bound_w/2 - image_center[0] 
    rotation_mat[1, 2] += bound_h/2 - image_center[1] 

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))   
    # cv2.imencode('.jpg', rotated_mat)[1].tofile("C:\\Users\\vicky\\Desktop\\picture_w\\"+name[:-4]+"_rot_"+str(angle)+".jpg")
    cv2.imencode('.jpg', rotated_mat)[1].tofile("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_already_del_rotate\\"+name[:-4]+"_rot_"+str(angle)+".jpg")


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
    
    if label in to_do_list:
        print("rot_label:", label)
        index = to_do_list.index(label)
        single_rot = int(np.ceil(40/statistic_f_df[1][index])) #一張照片要轉幾次
        
        for j in range(1, single_rot):
            rotate_image(img, 90*j, pic_list[i])
            # rotate_image(img, int(np.floor(360/single_rot))*j, pic_list[i])
            
