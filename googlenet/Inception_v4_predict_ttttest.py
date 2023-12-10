# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:48:31 2021

@author: User
"""
#!/usr/bin/env python
# coding: utf-8

# In[1]:
    

import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append(r"..\code")
import rgb_normalize
import clahe_yuv


# In[2]:

#預測類別轉換
def label_decode(pre, class_path=r"..\\insect_category_txt\\category_new_en.txt"):
    # class_path=r"D:\\category_new.txt"
    label=[]
    with open(class_path, 'r',encoding="utf-8") as f:
        for line in f.readlines():
            label.append(line.strip())
    return label[pre]


def get_category():
    class_path = "..\\insect_category_txt\\\\category_new_en.txt"
    label=[]
    with open(class_path, 'r',encoding="utf-8") as f:
        for line in f.readlines():
            label.append(line.strip())
    return label


def count_and_save(all_img_decoded_label, big_img_list, upload_dir):
    category = get_category()
    result = pd.DataFrame(category, columns=["Species"])
    dict_all_img_num = {'Number':[0]*len(category)}
    dict_one_img_num = {'Number':[0]*len(category)}
    
    for i in range(len(all_img_decoded_label)):
        for insect in all_img_decoded_label[i]:
            index = category.index(insect)
            dict_one_img_num["Number"][index] += 1
            dict_all_img_num["Number"][index] += 1
        result[big_img_list[i]] = dict_one_img_num["Number"]
        dict_one_img_num = {'Number':[0]*len(category)}  #歸零
            
    result["Accumulated Number"] = dict_all_img_num["Number"]
    # result.to_csv(upload_dir+"result.csv", encoding ="utf-8-sig", index = False, sep=",")
    result.to_excel(upload_dir+"result.xls", index = False)
    
    # #畫表格並將表格存成圖片
    # fig, ax = plt.subplots(figsize=(2, 6), dpi=150)
    # ax.axis('off')
    # ax.axis('tight')
    # plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
    
    # tb = ax.table(cellText=result.values, colLabels=result.columns, bbox=[0, 0, 1, 1])
    
    # tb[0, 0].set_facecolor('#363636')
    # tb[0, 1].set_facecolor('#363636')
    # tb[0, 0].set_text_props(color='w', fontsize=8)
    # tb[0, 1].set_text_props(color='w', fontsize=8)
    
    # plt.savefig(upload_dir+"result.png", dpi=300, bbox_inches = 'tight', pad_inches = 0)


def googlenet_predict(all_img_final_boxes, all_cut_img_list, big_img_list, dirs, model, model_moth, model_bee):
    upload_img_dir, yolo_draw_dir, cut_dir, final_dir, upload_dir = dirs
    
    # model_path = r"D:\\googlenet\\googlenet\\model\\googlenetV4_classification_Origin_run16_0428_1016_cut_gamma1_5_shuffle.h5"
    # img_path = 'D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_predict_cut\\'
    img_path = cut_dir
    
    # model_path_moth=r"D:\\googlenet\\googlenet\\model\\googlenetV4_classification_Origin_run17_0502_rgb_normalize_moth_shuffle.h5"
    
    # print("載入model...")
    # model = load_model(model_path)
    # print("載入完成")
    
    # print("載入model_moth...")
    # model_moth = load_model(model_path_moth)
    # print("載入完成")
    
    all_img_decoded_label = []
    for cut_imgs in all_cut_img_list:
        img_decoded_label = []
        for i in range(len(cut_imgs)):
            print("predicting:", cut_imgs[i])
            img=cv2.imdecode(np.fromfile(img_path+cut_imgs[i],dtype=np.uint8),cv2.IMREAD_COLOR) #等同=cv2.imread
            img_clahe = clahe_yuv.clahe_yuv(img)
            img_clahe = cv2.resize(img_clahe,(299,299))
            img_clahe = img_clahe.reshape(-1, 299, 299, 3)
            predict_prob = model.predict(img_clahe)
            predict = np.argmax(predict_prob) #最大值的index  axis=0 行 axis=1 列  ex: np.argmax(predict,axis=1)
            print("insect index:", predict)
            decoded_label = label_decode(predict)
            print("大model辨識結果:", decoded_label)
            

            #辨識成moth的再用合併model辨識第二次(rgb_normalize+org)
            # if decoded_label == ("蛾"or"麥蛾"or"螟蛾"or"蕈蛾"):
            if decoded_label == "moth" or decoded_label == "Sitotroga cerealella" or decoded_label == "Pyralidae" or decoded_label == "Tineidae":
                print("我是蛾類")
                moth=[11, 23, 15, 9]
                
                img_moth = rgb_normalize.rgb_normalize(img)
                img_moth = cv2.resize(img_moth,(299,299))
                img_moth = img_moth.reshape(-1, 299, 299, 3)
            
                print("predict before")
                predict_prob_moth = model_moth.predict(img_moth)
                print("predict after")
                predict_prob_moth = predict_prob_moth.reshape(-1)
                add = sum(predict_prob_moth)
                final_moth = [(predict_prob_moth[k]/add)*100 for k in range(0, 4)]
                
                predict_prob = predict_prob.reshape(-1)
                #9蕈蛾, 11蛾, 15螟蛾, 23麥蛾
                prob_org=[]
                for j in moth:
                    prob_org.append(predict_prob[j])
                add = sum(prob_org)
                final_org = [(prob_org[k]/add)*100 for k in range(0, 4)]
                
                temp_1 = final_org[0]*0.1+final_moth[0]*(1-0.1)
                temp_2 = final_org[1]*0.1+final_moth[1]*(1-0.1)
                temp_3 = final_org[2]*0.1+final_moth[2]*(1-0.1)
                temp_4 = final_org[3]*0.1+final_moth[3]*(1-0.1)
                big = max(temp_1, temp_2, temp_3, temp_4)
                if big == temp_1:
                    predict = "moth"
                elif big == temp_2:
                    predict = "Sitotroga cerealella"
                elif big == temp_3:
                    predict = "Pyralidae"
                else:
                    predict = "Tineidae"
                decoded_label = predict
                print(decoded_label)
                
            #辨識成bee類的再用合併model辨識第二次(rgb_normalize+org)
            # elif decoded_label == "小蜂"or"蟻"or"蠅"or"隱翅蟲":
            elif decoded_label == "Chalcidoidea" or decoded_label == "Formicidae" or decoded_label == "Muscoidea" or decoded_label == "Staphylinidae":
                print("我是小蜂類")
                category_num=[4, 16, 17, 21]
                
                img_bee = rgb_normalize.rgb_normalize(img)
                img_bee = cv2.resize(img_bee,(299,299))
                img_bee = img_bee.reshape(-1, 299, 299, 3)
                predict_prob_bee = model_bee.predict(img_bee)
                predict_prob_bee = predict_prob_bee.reshape(-1)
                add = sum(predict_prob_bee)
                final_bee = [(predict_prob_bee[k]/add)*100 for k in range(0, 4)]
                
                predict_prob = predict_prob.reshape(-1)
                #4小蜂, 16蟻, 17蠅, 21隱翅蟲
                prob_org=[]
                for j in category_num:
                    prob_org.append(predict_prob[j])
                add = sum(prob_org)
                final_org = [(prob_org[k]/add)*100 for k in range(0, 4)]
                
                temp_1 = final_org[0]*0.1+final_bee[0]*(1-0.1)
                temp_2 = final_org[1]*0.1+final_bee[1]*(1-0.1)
                temp_3 = final_org[2]*0.1+final_bee[2]*(1-0.1)
                temp_4 = final_org[3]*0.1+final_bee[3]*(1-0.1)
                big = max(temp_1, temp_2, temp_3, temp_4)
                if big == temp_1:
                    predict = "Chalcidoidea"
                elif big == temp_2:
                    predict = "Formicidae"
                elif big == temp_3:
                    predict = "Muscoidea"
                else:
                    predict = "Staphylinidae"
                decoded_label = predict
            else:
                print("other")
                
            print("最終辨識結果:", decoded_label)
            img_decoded_label.append(decoded_label)
    
        all_img_decoded_label.append(img_decoded_label)
    # print(all_img_decoded_label)
    draw(all_img_final_boxes, all_img_decoded_label, big_img_list, upload_img_dir, final_dir)
    count_and_save(all_img_decoded_label, big_img_list, upload_dir)

    return all_img_decoded_label, all_img_final_boxes
        

def draw(all_img_final_boxes, all_img_decoded_label, big_img_list, upload_img_dir, final_dir):
    # img_path = 'D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one\\'
    img_path = upload_img_dir
    # out_path = 'D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_final\\'
    out_path = final_dir

    for i in range(len(big_img_list)):
        img = cv2.imdecode(np.fromfile(img_path+big_img_list[i],dtype=np.uint8),cv2.IMREAD_COLOR) #等同=cv2.imread
        print("drawing:", big_img_list[i])       
        for k, box in enumerate(all_img_final_boxes[i]):
            x, y, w, h = box
            #畫框
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 6)
            #標上文字
            if y > 100:
                img = cv2ImgAddText(img, all_img_decoded_label[i][k] ,x, y-100, (255, 0, 0), 100)
            else:
                img = cv2ImgAddText(img, all_img_decoded_label[i][k] ,x, y+h+50, (255, 0, 0), 100)
        cv2.imencode('.jpg', img)[1].tofile(out_path+big_img_list[i]) 


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):  #照片加上中文字
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV圖片類型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  #OpenCV圖片轉換為PIL圖片格式
    
    #使用PIL繪製文字
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
   
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  #PIL圖片格式轉換成OpenCV的圖片格式


##為了分步辨識新增的函數
def model(model, img):
    predict_prob = model.predict(img)
    predict = np.argmax(predict_prob) #最大值的index  axis=0 行 axis=1 列  ex: np.argmax(predict,axis=1)
    # print("insect index:", predict)
    decoded_label = label_decode(predict)
    print(decoded_label)
    return decoded_label, predict_prob

def model_moth(model, img):
    predict_prob = model.predict(img)
    predict = np.argmax(predict_prob) #最大值的index  axis=0 行 axis=1 列  ex: np.argmax(predict,axis=1)
    # print("insect index:", predict)
    class_path_moth = r"..\\insect_category_txt\\\\category_moth_en.txt"
    decoded_label = label_decode(predict, class_path_moth)
    print(decoded_label)
    return decoded_label, predict_prob

def model_bee(model, img):
    predict_prob = model.predict(img)
    predict = np.argmax(predict_prob) #最大值的index  axis=0 行 axis=1 列  ex: np.argmax(predict,axis=1)
    # print("insect index:", predict)
    class_path_bee = r"..\\insect_category_txt\\\category_bee_en.txt"
    decoded_label = label_decode(predict, class_path_bee)
    print(decoded_label)
    return decoded_label, predict_prob


def merge_model_and_model_moth(predict_prob, predict_prob_moth):
    moth=[11, 23, 15, 9]
    
    predict_prob_moth = predict_prob_moth.reshape(-1)
    add = sum(predict_prob_moth)
    final_moth = [(predict_prob_moth[k]/add)*100 for k in range(0, 4)]
    
    predict_prob = predict_prob.reshape(-1)
    #9蕈蛾, 11蛾, 15螟蛾, 23麥蛾
    prob_org=[]
    for j in moth:
        prob_org.append(predict_prob[j])
    add = sum(prob_org)
    final_org = [(prob_org[k]/add)*100 for k in range(0, 4)]
    
    temp_1 = final_org[0]*0.1+final_moth[0]*(1-0.1)
    temp_2 = final_org[1]*0.1+final_moth[1]*(1-0.1)
    temp_3 = final_org[2]*0.1+final_moth[2]*(1-0.1)
    temp_4 = final_org[3]*0.1+final_moth[3]*(1-0.1)
    big = max(temp_1, temp_2, temp_3, temp_4)
    if big == temp_1:
        predict = "moth"
    elif big == temp_2:
        predict = "Sitotroga cerealella"
    elif big == temp_3:
        predict = "Pyralidae"
    else:
        predict = "Tineidae"
    decoded_label = predict
    print(decoded_label)
    return decoded_label


def merge_model_and_model_bee(predict_prob, predict_prob_bee):
    category_num=[4, 16, 17, 21]

    predict_prob_bee = predict_prob_bee.reshape(-1)
    add = sum(predict_prob_bee)
    final_bee = [(predict_prob_bee[k]/add)*100 for k in range(0, 4)]
    
    predict_prob = predict_prob.reshape(-1)
    #4小蜂, 16蟻, 17蠅, 21隱翅蟲
    prob_org=[]
    for j in category_num:
        prob_org.append(predict_prob[j])
    add = sum(prob_org)
    final_org = [(prob_org[k]/add)*100 for k in range(0, 4)]
    
    temp_1 = final_org[0]*0.1+final_bee[0]*(1-0.1)
    temp_2 = final_org[1]*0.1+final_bee[1]*(1-0.1)
    temp_3 = final_org[2]*0.1+final_bee[2]*(1-0.1)
    temp_4 = final_org[3]*0.1+final_bee[3]*(1-0.1)
    big = max(temp_1, temp_2, temp_3, temp_4)
    if big == temp_1:
        predict = "Chalcidoidea"
    elif big == temp_2:
        predict = "Formicidae"
    elif big == temp_3:
        predict = "Muscoidea"
    else:
        predict = "Staphylinidae"
    decoded_label = predict
    return decoded_label



# In[3]:


if __name__ == '__main__':
    
    pass

    # model_path=r"D:\\googlenet\\googlenet\\model\\googlenetV4_classification_Origin_run8_0407_gamma1_5_shuffle.h5"
    # img_path='D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_predict_cut\\'
    
    # model = load_model(model_path)

    # img_name=[]
    # all_list_name = os.listdir(img_path)
    # for x in range(len(all_list_name)):
    #     if ".JPG"in all_list_name[x] or ".jpg" in all_list_name[x]:
    #         img_name.append(all_list_name[x])
    
    # for i in range(len(img_name)):
    #     img=cv2.imdecode(np.fromfile(img_path+img_name[i],dtype=np.uint8),cv2.IMREAD_COLOR) #等同=cv2.imread
    #     img=cv2.resize(img,(299,299))
    #     img=img.reshape(-1, 299, 299, 3)
    #     predict_prob = model.predict(img)
    #     predict = np.argmax(predict_prob) #最大值的index  axis=0 行 axis=1 列  ex: np.argmax(predict,axis=1)
    #     decoded_label= label_decode(predict)
    #     print(decoded_label)


# In[ ]:

