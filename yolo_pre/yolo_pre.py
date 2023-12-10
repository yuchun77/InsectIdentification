#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import sys
sys.path.append(r"..\googlenet\googlenet")
import Inception_v4_predict_ttttest
sys.path.append(r"..\code")
import gamma_correction


model_path=r"..\\darknet\\build\\darknet\\x64\\backup\\yolo-obj_best.weights"
cfg_path=r"..\\darknet\\build\\darknet\\x64\\cfg\\yolov4-obj.cfg"
classes_path=r"..\\darknet\\build\\darknet\\x64\\data\\obj.names"


net = cv2.dnn.readNetFromDarknet(cfg_path,model_path) #載入模型
layer_names = net.getLayerNames() #神經網路模型架構
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] #輸出層
classes = [line.strip() for line in open(classes_path)] #所有分類label
colors = [(0,0,255),(255,0,0),(0,255,0)] #框框顏色


def yolo_detect(image):
    # forward propogation
    img = cv2.resize(image, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape 
    
    #cv讀取圖片並做資料標準化
    blob = cv2.dnn.blobFromImage(img, #圖片
                                 1/255.0, #pixel0-1之間
                                 (416, 416), #input影像大小,
                                 (0, 0, 0), #光照調整參數
                                 True, #BGR轉RGB
                                 crop=False) #是否裁切圖片
    net.setInput(blob) #輸入神經網路
    outs = net.forward(output_layers) #輸出結果

    # get detection boxes
    class_ids = [] #各偵測物件分類
    confidences = [] #各偵測物件信賴度confidences
    boxes = [] #各偵測物件座標
    ratio_boxes = []   #各偵測物件座標比例  #自己加的
    
    for out in outs:
        for detection in out:
            tx, ty, tw, th, confidence = detection[0:5]
            scores = detection[5:] #各分類機率
            class_id = np.argmax(scores)  
            if confidence > 0.3:   #信賴度
                center_x = int(tx * width)
                center_y = int(ty * height)
                w = int(tw * width)
                h = int(th * height)

                # 取得箱子方框座標
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                ratio_boxes.append([tx, ty, tw, th])   #自己加的
                
    
    #移除低confidences 去除重複框選
    indexes = cv2.dnn.NMSBoxes(boxes, #各偵測物件座標
                               confidences, #各偵測物件confidences
                               0.25, #confidences threshold
                               0.1) #Non-max suppression threshold
    # draw boxes
    font = cv2.FONT_HERSHEY_SIMPLEX
    final=[]
    final_ratio_boxes = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            
            if w>=24 and h>=24:   #60*0.4=24
                final.append(boxes[i])
                final_ratio_boxes.append(ratio_boxes[i])  #自己加的

                label = str(classes[class_ids[i]])
                print(label,x,y,w,h)
                color = colors[class_ids[i]]
                #畫框
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 6)
                #標上文字
                if y >40:
                    cv2.putText(img, label ,(x,y-6),font,2,color,3)
                else:
                    cv2.putText(img, label ,(x,y + h+50),font,2,color,3)

    return img, final, final_ratio_boxes



def cut_photo(img, final_ratio_boxes, cut_path, big_img_list, i):
    final_boxes = []
    cut_img_list = []
    for j, box in enumerate(final_ratio_boxes):
        height, width, _ = img.shape
        center_x_ratio, center_y_ratio, w_ratio, h_ratio = box
        x = int((center_x_ratio - w_ratio/2)*width)
        y = int((center_y_ratio - h_ratio/2)*height)
        w = int(w_ratio*width)
        h = int(h_ratio*height)
        if x<0:
            x = 0
        if y<0:   
            y = 0
        # if w>=60 and h>=60:
        final_boxes.append([x, y, w, h])
        ROI = img[y:y+h, x:x+w]
        check_file_exist(cut_path)
        cv2.imencode('.jpg', ROI)[1].tofile(cut_path+str(j)+"_"+big_img_list[i]) #=cv2.imwrite
        cut_img_list.append(str(j)+"_"+big_img_list[i])
    return final_boxes, cut_img_list


def check_file_exist(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def yolo_predict(dirs, obj_state, model, model_moth, model_bee):
    upload_img_dir, yolo_draw_dir, cut_dir, final_dir, upload_dir = dirs
    
    img_path = upload_img_dir
    out_path = yolo_draw_dir
    cut_path = cut_dir
    
    big_img_list=[]
    all_list_name = os.listdir(img_path)
    for x in range(len(all_list_name)):
        if ".JPG"in all_list_name[x] or ".jpg" in all_list_name[x]:
            big_img_list.append(all_list_name[x])

    all_img_final_boxes = []
    all_cut_img_list = []
    for i in range(len(big_img_list)):
        img=cv2.imdecode(np.fromfile(img_path+big_img_list[i],dtype=np.uint8),cv2.IMREAD_COLOR) #等同=cv2.imread
        print(big_img_list[i])
        img_gamma = gamma_correction.gamma_correction(img, 1.5)
        img_draw, final, final_ratio_boxes = yolo_detect(img_gamma)
        
        #儲存圖檔
        check_file_exist(out_path)
        cv2.imencode('.jpg', img_draw)[1].tofile(out_path+big_img_list[i]) #=cv2.imwrite

        final_boxes, cut_img_list = cut_photo(img, final_ratio_boxes, cut_path, big_img_list, i)
        all_img_final_boxes.append(final_boxes)
        all_cut_img_list.append(cut_img_list)

    all_img_decoded_label, all_img_final_boxes = Inception_v4_predict_ttttest.googlenet_predict(all_img_final_boxes, all_cut_img_list, big_img_list, dirs, model, model_moth, model_bee)
    print(all_img_decoded_label)
    print(all_img_final_boxes)
    obj_state.set_value("finish")
    return big_img_list, all_img_decoded_label
        

if __name__=='__main__':
    # img_path=r"D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one\\"
    # # img_name='DSC_9463.jpg'
    # out_path='D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_predict\\'
    # cut_path='D:\\darknet2\\darknet\\build\\darknet\\x64\\data\\img\\one_predict_cut\\'
    
    # big_img_list=[]
    # all_list_name = os.listdir(img_path)
    # for x in range(len(all_list_name)):
    #     if ".JPG"in all_list_name[x] or ".jpg" in all_list_name[x]:
    #         big_img_list.append(all_list_name[x])
    
    
    # all_img_final_boxes = []
    # all_cut_img_list = []
    # for i in range(len(big_img_list)):
    #     img=cv2.imdecode(np.fromfile(img_path+big_img_list[i],dtype=np.uint8),cv2.IMREAD_COLOR) #等同=cv2.imread
    #     print(big_img_list[i])
    #     img_draw, final, final_ratio_boxes = yolo_detect(img)
        
    #     #儲存圖檔
    #     check_file_exist(out_path)
    #     cv2.imencode('.jpg', img_draw)[1].tofile(out_path+big_img_list[i]) #=cv2.imwrite
        
        
    #     # #用resize的圖片
    #     # for j, box in enumerate(final):
    #     #     x, y, w, h = box
    #     #     ROI = im[y:y+h, x:x+w]
    #     #     check_file_exist(cut_path)
    #     #     cv2.imencode('.jpg', ROI)[1].tofile(cut_path+str(j)+"__"+big_img_list[i]) #=cv2.imwrite
    
    
    #     # #用原圖
    #     # for j, box in enumerate(final_ratio_boxes):
    #     #     height, width, _ = img.shape
    #     #     center_x_ratio, center_y_ratio, w_ratio, h_ratio = box
    #     #     x = int((center_x_ratio - w_ratio/2)*width)
    #     #     y = int((center_y_ratio - h_ratio/2)*height)
    #     #     w = int(w_ratio*width)
    #     #     h = int(h_ratio*height)
    #     #     ROI = img[y:y+h, x:x+w]
    #     #     check_file_exist(cut_path)
    #     #     cv2.imencode('.jpg', ROI)[1].tofile(cut_path+str(j)+"_"+big_img_list[i]) #=cv2.imwrite

    #     final_boxes, cut_img_list = cut_photo(img, final_ratio_boxes, cut_path, big_img_list, i)
    #     all_img_final_boxes.append(final_boxes)
    #     all_cut_img_list.append(cut_img_list)

    
    # #圖片預覽
    # # cv2.namedWindow('detection', 0);
    # # cv2.resizeWindow('detection', 1024, 768);
    # # cv2.imshow('detection', im)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    

    
    # all_img_decoded_label = Inception_v4_predict_ttttest.predict(all_img_final_boxes, all_cut_img_list, big_img_list)
    # print(all_img_decoded_label)
    
    
    
    pass

