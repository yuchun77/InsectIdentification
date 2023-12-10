# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:14:09 2021

@author: User
"""
import cv2
import glob
import os.path
import time
import numpy as np


def rgb_normalize(img):
    (b, g, r) = cv2.split(img)
    
    B = cv2.normalize(b, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    G = cv2.normalize(g, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    R = cv2.normalize(r, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)    
    
    img = cv2.merge((B ,G ,R ))
    
    return img



if __name__ == '__main__':
    
    path = 'D:\test'
    
    all_thing_path = "D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_cut_very_little\\"
    all_thing_list = os.listdir(all_thing_path)
    pic_list=[]
    for x in range(len(all_thing_list)):
        if ".JPG"in all_thing_list[x] or ".jpg" in all_thing_list[x]:
            pic_list.append(all_thing_list[x])
            
    # print(pic_list)
    
    
    
    
    for i in range(len(pic_list)):
        print("img："+ pic_list[i])
        # img = cv2.imread(all_thing_path + pic_list[i], encoding='utf-8') #讀檔
        img = cv2.imdecode(np.fromfile(all_thing_path + pic_list[i],dtype=np.uint8),-1)
        (b, g, r) = cv2.split(img)
        
        B = cv2.normalize(b, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)      
        G = cv2.normalize(g, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)  
        R = cv2.normalize(r, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

        img2 = cv2.merge((B ,G ,R ))
        cv2.imencode('.jpg', img2)[1].tofile("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_norm\\"+pic_list[i])
        # cv2.imwrite("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_cut_clahe\\"+pic_list[i], img)


