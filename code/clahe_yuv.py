import cv2
import glob
import os.path
import time
import numpy as np
from PIL import Image, ImageTk, ImageSequence

def clahe_yuv(img):
    (b, g, r) = cv2.split(img)
#    Y = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
#     b = clahe.apply(b)
#     g = clahe.apply(g)
#     r = clahe.apply(r)
#     Y = cv2.merge([b,g,r])
    
    Y = 0.299 * r + 0.587 * g + 0.114 * b #灰階
    #Y = 0.55 * r + 0.91 * g + 0.167 * b
    U = -0.147 * r - 0.289 * g + 0.436 * b 
    V = 0.615 * r - 0.515 * g - 0.100 * b 
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    equal_img_v = clahe.apply(img)
    
    R = equal_img_v + 0 * U + 1.140 * V
    G = equal_img_v - 0.395 * U - 0.581 * V
    B = equal_img_v + 2.032 * U + 0 * V

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
    #    Y = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    #     b = clahe.apply(b)
    #     g = clahe.apply(g)
    #     r = clahe.apply(r)
    #     Y = cv2.merge([b,g,r])
        
        Y = 0.299 * r + 0.587 * g + 0.114 * b #灰階
        #Y = 0.55 * r + 0.91 * g + 0.167 * b
        U = -0.147 * r - 0.289 * g + 0.436 * b 
        V = 0.615 * r - 0.515 * g - 0.100 * b 
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        equal_img_v = clahe.apply(img)
        
        R = equal_img_v + 0 * U + 1.140 * V
        G = equal_img_v - 0.395 * U - 0.581 * V
        B = equal_img_v + 2.032 * U + 0 * V
    
        img = cv2.merge((B ,G ,R ))
        cv2.imencode('.jpg', img)[1].tofile("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_clahe\\"+pic_list[i])
        
        
        
        #先存起來再用cv2讀檔
        img3 = cv2.imdecode(np.fromfile("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_clahe\\"+pic_list[i], dtype=np.uint8),-1)
        cv2.imencode('.jpg', img3)[1].tofile("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_clahe\\"+"save"+pic_list[i])
    
        # cv2.imwrite("D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\1016_cut_clahe\\"+pic_list[i], img)
