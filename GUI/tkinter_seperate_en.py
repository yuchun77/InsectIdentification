# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:30:27 2021

@author: User
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk, ImageSequence
import os
import shutil
import psutil
import gc
import cv2
import numpy as np
import pandas as pd
import prettytable as pt
from tensorflow.keras.models import load_model
from datetime import datetime
import time
import threading

import sys
sys.path.append(r"..\code")
import gamma_correction
import clahe_yuv
import rgb_normalize
sys.path.append(r"..\yolo_pre")
import yolo_pre
sys.path.append(r"..\googlenet\googlenet")
import Inception_v4_predict_ttttest
import global_global


#Insect_Species_Identification_System資料夾位置
ISIS_path = os.path.dirname(os.path.abspath('..'))
ISIS_path = ISIS_path.replace('\\', '/')+"/"
# ISIS_path = r"C:/Users/vicky/Desktop/"

#定義global變數
big_img_list = []
all_img_decoded_label = []
obj_state = global_global.init("")

upload_img_dir = ""
yolo_draw_dir = ""
cut_dir =""
final_dir = ""
upload_dir = ""

step_region_dir = ""
step_region_clahe_dir = ""
step_dir = ""

selectFilePaths = ""

category = Inception_v4_predict_ttttest.get_category()

#分步辨識圖片的寬跟高
img_width = 0
img_height = 0


# 顯示當前 python 程式佔用的記憶體大小
def show_memory_info(hint):
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    memory = info.rss / 1024. / 1024
    print('{} memory used: {} MB'.format(hint, memory))
    

#視窗置中
def get_screen_size(window):  
    return window.winfo_screenwidth(),window.winfo_screenheight()  

def center_window(root, width, height):  
    screenwidth, screenheight = get_screen_size(root)
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width)/2-8, (screenheight - height)/2-35) 
    # print(size)
    root.geometry(size) 
    root.update()

def get_window_corordinate(window):  
    return window.winfo_x(),window.winfo_y() 

def center_toplevel(toplevel):
    x, y = get_window_corordinate(root)
    toplevel.geometry("+%d+%d" % (x+200, y+35))
    

#程式結束關閉視窗
def closewindow():
    ans = messagebox.askyesno(title='Warning' ,message='Close the window?')
    if ans:
        root.destroy()
    else:
        return


def put_img_specify_width(img_open, img_size, label):   #img要為PIL格式
    scale = img_open.size[0]/img_size    #找出要縮小或放大幾倍
    img_open = img_open.resize((img_size, int(img_open.size[1]/scale)), Image.ANTIALIAS)  #指定寬度
    img = ImageTk.PhotoImage(img_open)  
    label.config(image=img)  
    label.image = img  #keep a reference
    return int(img_open.size[0]), int(img_open.size[1])


def put_img_specify_height(img_open, img_size, label):   #img要為PIL格式
    scale = img_open.size[1]/img_size    #找出要縮小或放大幾倍
    img_open = img_open.resize((int(img_open.size[0]/scale), img_size), Image.ANTIALIAS)  #指定高度
    img = ImageTk.PhotoImage(img_open)  
    label.config(image=img)
    label.image = img  #keep a reference
    return int(img_open.size[0]), int(img_open.size[1])


def show_img(use, page):
    if use == "preview":
        path = selectFilePaths[page-1]   #page-1才會是對應的index
    elif use == "recognize":
        path = final_dir+big_img_list[page-1]
    
    #填入照片
    img_open = Image.open(path)
    if img_open.size[0] >= img_open.size[1]:  #寬大於等於高
        put_img_specify_width(img_open, 750, label_for_img1)
    else:  #寬小於高
        put_img_specify_height(img_open, 510, label_for_img1)
    
    #填入照片名稱
    label_for_img_name.place(x=20, y=60)
    label_for_img_name.config(text=path)


def prev_img(use, current_page):
    if current_page > 1:  #頁數比1大才能上一張
        e_for_page.set(current_page-1)
        show_img(use, current_page-1)   #要show現在頁數的上一張
        if use == "recognize":
            show_one_img_result(current_page-1)
    else:
        messagebox.showinfo("hint", "This is the first photo.")


def next_img(use, current_page):  #頁數比總頁數小才能下一張
    if current_page < len(selectFilePaths):
        e_for_page.set(current_page+1)
        show_img(use, current_page+1)  #要show現在頁數的下一張
        if use == "recognize":
            show_one_img_result(current_page+1)
    else:
        messagebox.showinfo("hint", "This is the last photo.")



def choose_files():  #選擇照片並展示
    global selectFilePaths
    selectFilePaths = filedialog.askopenfilenames(title='select images', initialdir=os.getcwd(), filetypes=[('image files', '*.png *.jpeg *.jpg')])  #獲得圖片路徑
    global obj_state
    obj_state.set_value("")
    if selectFilePaths == "":   #按了選擇圖案又不選
        temp = path_entry.get()
        e_for_path.set(temp)   #填回原本的路徑
    else:
        #填入所選取的各個照片路徑
        e_for_path.set(selectFilePaths)
        
        #將開始辨識按鈕的狀態變成可以按
        button_for_recognize.config(state=tk.NORMAL)
        
        #填入預覽的照片
        label_for_img1.place(x=20, y=80)
        show_img("preview", 1)
                
        #這時才出現上一張、下一張的按鈕
        button_prev.place(anchor=tk.W, height=30, width=65, x=850, y=560)
        button_prev.config(command=lambda:prev_img("preview", int(e_for_page.get())))
        button_next.place(anchor=tk.W, height=30, width=65, x=960, y=560)
        button_next.config(command=lambda:next_img("preview", int(e_for_page.get())))
          
        #出現並設定目前頁數
        page_entry.place(anchor=tk.W, height=25, width=25, x=925, y=560)
        e_for_page.set(1)
        
        #填入共幾張
        label_for_num_page.place(x=990, y=575)
        label_for_num_page.config(text=str(len(selectFilePaths))+" in total")
        
        #再次選擇照片時，將原本的辨識結果清除
        label_for_result_title['text'] = ""
        label_for_line.place_forget()
        label_for_result_category['text'] = ""
        label_for_result_num['text'] = ""
        button_for_allresult.place_forget()



def choose_file():
    selectFilePath = filedialog.askopenfilename(title='select image', initialdir=os.getcwd(), filetypes=[('image files', '*.png *.jpeg *.jpg')])  #獲得圖片路徑
    if selectFilePath == "":   #按了選擇圖案又不選
        temp = path_entry.get()
        e_for_path.set(temp)   #填回原本的路徑
    else: 
        #隱藏版面上的所有物件
        for widget in root.place_slaves():
            for g in widget.grid_slaves():
                for p in g.pack_slaves():  #因為在frame_for_step_p1新增了一個frame用pack的方式放切割照片
                    p.pack_forget()
                g.grid_forget()
            widget.place_forget()
        
        button_for_step.config(text="Step1：Gamma correction", command=lambda:Threader(gamma, path_entry.get()), state=tk.NORMAL)
        button_for_step.place(anchor=tk.NW, height=35, width=270, x=535, y=10)   #放上分步辨識按鈕
        button_for_choose_file.place(anchor=tk.NW, height=35, width=150, x=380, y=10) #放上選擇一張圖片按鈕
        path_entry.place(anchor=tk.NW, height=35, width=350, x=20, y=10)
        button_for_rechoose.place(anchor=tk.NW, height=35, width=220, x=830, y=10)
        
        #填入圖片路徑
        e_for_path.set(selectFilePath)  
        #填入要辨識的照片
        label_for_step_img.place(x=100, y=60)
        img_open = Image.open(selectFilePath)
        if img_open.size[0] >= img_open.size[1]:  #寬大於等於高
            put_img_specify_width(img_open, 780, label_for_step_img)
        else:  #寬小於高
            put_img_specify_height(img_open, 520, label_for_step_img)



def take_dir_path(file_path):
    #取得dir_path
    for j in range(len(file_path)):
        last = len(file_path)-1
        if file_path[last-j]  == "/" or file_path[last-j]  == "\\":
            dir_path = file_path[:(last-j)+1]
            break
    print("dir_path", dir_path)
    return dir_path


def choose_region():
    selectFilePath = filedialog.askopenfilename(title='Select regional image', initialdir=step_region_dir, filetypes=[('image files', '*.png *.jpeg *.jpg')])  #獲得圖片路徑
    if selectFilePath != "":
        if take_dir_path(selectFilePath) == step_region_dir:
            Threader(clahe, selectFilePath)
        else:
            messagebox.showinfo("hint", "Please select one regional image which is just cut out.")
    


def remove_symbol(a):   #用來拆開 e_path_entry.get()
        a = a.split(" ")
        return a


def setup_dir(upload_dir, paths):  #建立接下來需要的各種資料夾
    abs_paths = []
    for path in paths:
        os.makedirs(upload_dir+path, exist_ok=True)  #創建upload系列資料夾  #可創建多層的資料夾，如果前一層資料夾不存在，程式會自動幫你建立
        abs_paths.append(upload_dir+path)
    abs_paths.append(upload_dir)
    return abs_paths


def take_img_name(file_path):
    #取得img_name
    for j in range(len(file_path)):
        last = len(file_path)-1
        if file_path[last-j]  == "/" or file_path[last-j]  == "\\":
            img_name = file_path[(last-j)+1:]
            break
    print("img_name", img_name)
    return img_name


def show_one_img_result(page):
    #統計各類別有幾隻
    result = pd.DataFrame(category, columns=["Species"])
    dict_num = {'Number':[0]*len(category)}
    
    for insect in all_img_decoded_label[page-1]:   ##page-1才會是對應的index
        index = category.index(insect)
        dict_num["Number"][index] += 1
        
    result["Number"] = dict_num["Number"]
    
    #種類table
    tb_cat = pt.PrettyTable()
    tb_cat.add_column("Species", result["Species"])
    tb_cat.set_style(pt.PLAIN_COLUMNS)
    tb_cat.align = 'l'
    
    #數量table
    tb_num = pt.PrettyTable()
    tb_num.add_column("Number", result["Number"])
    tb_num.set_style(pt.PLAIN_COLUMNS)
    tb_num.align = 'c'
    
    #填入辨識結果
    label_for_result_category.place(width=200, x=800, y=110)
    label_for_result_category.config(text=tb_cat, justify=tk.LEFT)
    label_for_result_num.place(x=1000, y=110)
    label_for_result_num.config(text=tb_num, justify=tk.CENTER)



def read(sfname):
    df = pd.read_excel(sfname, header=0)
    cols = list(df.columns)
    cols.insert(0, "")
    return df, cols

def showdata(frame, df, cols):
    tree = ttk.Treeview(frame, columns=cols, show="headings")
    tree.pack()
    for i in cols:
        tree.column(i, width=50, anchor="center")
        tree.heading(i, text=i, anchor='center')
    for index, row in df.iterrows():
        row = list(row)
        row.insert(0, index+1)
        tree.insert("", index, values=row)
    return tree

def show_all_result():
    toplevel = tk.Toplevel(root)
    toplevel.title("Identification result")
    toplevel.geometry("600x530")
    center_toplevel(toplevel)
    
    data, c = read(upload_dir+"result.xls")
    tree = showdata(toplevel, data, c)
    tree.place(relx=0,rely=0,relheight=1,relwidth=1)

    #創建滾動條
    scroll = tk.Scrollbar(toplevel, orient=tk.HORIZONTAL)
    #將滾動條填充
    scroll.pack(side=tk.BOTTOM, fill=tk.X) # side是滾動條放置的位置，上下左右。fill是將滾動條沿著y軸填充
    #將滾動條與文本框關聯
    scroll.config(command=tree.xview) #將tree關聯到滾動條上，滾動條滑動，tree跟隨滑動
    tree.config(xscrollcommand=scroll.set) #將滾動條關聯到tree

    

def recognize(file_paths):
    if file_paths == "":  #還沒選照片就按開始辨識
        messagebox.showinfo("hint", "Please select at least one picture.")
    else:
        #把上一張、下一張、頁數、總頁數先隱藏起來
        button_prev.place_forget()
        button_next.place_forget()
        page_entry.place_forget()
        label_for_num_page.config(text="")
        label_for_img_name.config(text="")
        
        #將開始辨識、選擇影像、重選展示方式按鈕的狀態變成不可按
        button_for_recognize.config(state=tk.DISABLED)
        button_for_choose_files.config(state=tk.DISABLED)
        button_for_rechoose.config(state=tk.DISABLED)
        
        #把黏在一起的各個檔案路徑分開
        file_paths_list = remove_symbol(file_paths)
        print(file_paths_list)
        
        #做出upload資料夾
        global upload_img_dir, yolo_draw_dir, cut_dir, final_dir, upload_dir
        time = datetime.now().strftime('%Y%m%d%H%M%S')  #取得現在時間
        upload_dir = ".\\upload\\upload_"+time+"\\"  #結尾有沒有加\\都可以
        paths = ["img\\", "draw\\", "cut\\", "final\\"]
        dirs = setup_dir(upload_dir, paths)
        upload_img_dir, yolo_draw_dir, cut_dir, final_dir, upload_dir = dirs
        
        #上傳檔案至upload資料夾
        for i in range(len(file_paths_list)):
            #取得img_name
            img_name = take_img_name(file_paths_list[i])
        
            upload_path = upload_img_dir+img_name
            shutil.copyfile(file_paths_list[i], upload_path)
            print("上傳至："+upload_path)
        
        #yolo切割
        obj_state.set_value("not finish")
        print(obj_state.get_value())
        show_memory_info('before yolo_pre')
        t = Threader(yolo_pre.yolo_predict, dirs, obj_state, model, model_moth, model_bee) #建立一個子執行緒做yolo切割
        
        # 顯示loading中
        loading_gif(obj_state)
        
        #等yolo子執行續做完
        t.join()
        print(obj_state.get_value())
        show_memory_info('after yolo_pre')
        
        global big_img_list, all_img_decoded_label
        big_img_list, all_img_decoded_label = t.get_result()
        
        #填入辨識完成的照片
        show_img("recognize", 1)  #第1張照片(從page1開始)
        
        # #出現上一張、下一張
        button_prev.place(anchor=tk.W, height=30, width=65, x=850, y=560)
        button_prev.config(command=lambda:prev_img("recognize", int(e_for_page.get())))
        button_next.place(anchor=tk.W, height=30, width=65, x=960, y=560)
        button_next.config(command=lambda:next_img("recognize", int(e_for_page.get())))
        
        #出現並設定目前頁數
        page_entry.place(anchor=tk.W, height=25, width=25, x=925, y=560)
        e_for_page.set(1)

        #填入辨識結果
        label_for_result_title.config(text="Identification result")
        label_for_result_title.place(x=850, y=80)
        img_open = Image.open('.\\img\\line.png')
        put_img_specify_width(img_open, 250, label_for_line)
        label_for_line.place(x=800, y=100)
        show_one_img_result(1)  #第1張照片的結果(從page1開始)
        
        #填入共幾張
        label_for_num_page.config(text=str(len(big_img_list))+" in total")

        #出現查看本次辨識結果按鈕
        button_for_allresult.place(anchor=tk.NW, height=35, width=160, x=610, y=10)
        
        #將選擇影像、重選展示方式的狀態變成可按
        button_for_choose_files.config(state=tk.NORMAL)
        button_for_rechoose.config(state=tk.NORMAL)


#分解gif並逐貞顯示
def loading_gif(obj_state):
    label_for_img1.config(image="")
    label_for_loading.place(x=480, y=240)
    label_for_loading_text.config(text="Identifying •••")
    label_for_loading_text.place(x=460, y=365)
    while obj_state.get_value() == "not finish":
        loading_img = Image.open('.\\img\\loading.gif')
        # GIF圖片流的迭代器
        iter = ImageSequence.Iterator(loading_img)
        #frame就是gif的每一帧，轉換下格式就能顯示了
        for frame in iter:
            frame = frame.resize((90, 90), Image.ANTIALIAS)  #指定高度
            img = ImageTk.PhotoImage(frame)
            label_for_loading.config(image=img)
            time.sleep(0.1)
            root.update_idletasks()  #刷新
            root.update()
    label_for_loading.place_forget()
    label_for_loading_text.place_forget()



# 用來放步驟圖片跟文字說明
def put_step_imgandtext(frame, row, col, img, img_size, description, ipaddingx):  #圖片格式為cv2格式
    label = tk.Label(frame)
    label.grid(row=row, column=col, ipadx=ipaddingx)
    img_open = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  #在這裡轉為PIL格式
    if img_open.size[0] >= img_open.size[1]:  #寬大於等於高
        width, height = put_img_specify_width(img_open, img_size, label)
    else:  #寬小於高
        width, height = put_img_specify_height(img_open, img_size-30, label)
    tk.Label(frame, text=description, pady=8, font=fontStyle_normal_text).grid(row=row+1, column=col, ipady=5)
    return width, height


def arrow(frame, row, col, size, angle, ipaddingx):
    label = tk.Label(frame)
    label.grid(row=row, column=col, ipadx=ipaddingx)
    img_open = Image.open('.\\img\\arrow'+str(angle)+'.png')
    put_img_specify_width(img_open, size, label)


def gamma(file_path):
    if file_path == "":  #還沒選照片就按gamma
        messagebox.showinfo("hint", "Please select one picture.")
    else:
        label_for_step_img.place_forget()
        
        #將gamma按鈕的狀態變成不可按
        button_for_step.config(state=tk.DISABLED)
        
        #填入原圖及說明文字
        global img_width, img_height
        frame_for_step_p1.place(x=45, y=70)
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        img_width, img_height = put_step_imgandtext(frame_for_step_p1, 0, 0, img, 310, "original image", 35)
        
        #箭頭
        arrow(frame_for_step_p1, 0, 1, 50, 90, 40)
        
        #原圖做gamma
        img_gamma = gamma_correction.gamma_correction(img, 1.5)
        
        #填入gamma1.5後的照片及說明文字
        put_step_imgandtext(frame_for_step_p1, 0, 2, img_gamma, 310, "gamma1.5", 35)
        
        #更改步驟button
        button_for_step.config(text="Step2：YOLO detection", command=lambda:Threader(yolo, img_gamma), state=tk.NORMAL)


def yolo(img_gamma):
    button_for_step.config(state=tk.DISABLED)
    #箭頭
    arrow(frame_for_step_p1, 1, 1, 50, 225, 40)
    
    #YOLO辨識
    img_draw, _, final_ratio_boxes = yolo_pre.yolo_detect(img_gamma)
    
    #填入YOLO辨識後的照片及說明文字
    put_step_imgandtext(frame_for_step_p1, 2, 0, img_draw, 310, "yolo detection result", 35)
    
    #更改步驟button
    button_for_step.config(text="Step3：Extract regional images", command=lambda:Threader(cut, final_ratio_boxes), state=tk.NORMAL)
    

def cut(final_ratio_boxes):
    button_for_step.config(state=tk.DISABLED)
    #箭頭
    arrow(frame_for_step_p1, 2, 1, 50, 90, 40)
    
    #做出step資料夾
    global step_region_dir, step_region_clahe_dir, step_dir
    time = datetime.now().strftime('%Y%m%d%H%M%S')  #取得現在時間
    step_dir = ISIS_path + r"Insect_Species_Identification_System/GUI/step/step_"+time+"/"  #結尾有沒有加\\都可以
    paths = ["region/", "region_clahe/"]
    dirs = setup_dir(step_dir, paths)
    step_region_dir, step_region_clahe_dir, step_dir = dirs
    print(step_region_dir)
    print(step_region_clahe_dir)
    print(step_dir)
    
    #切割原圖
    img = cv2.imdecode(np.fromfile(path_entry.get(), dtype=np.uint8), -1)
    cut_path = step_region_dir
    img_name = take_img_name(path_entry.get())
    big_img_list = [img_name]
    _, cut_img_list = yolo_pre.cut_photo(img, final_ratio_boxes, cut_path, big_img_list, 0)
    print(cut_img_list)
    
    #resize照片比較好排版
    imgs = []
    size = int(img_height/2-10)
    for cut_img in cut_img_list:
        img_open = Image.open(cut_path+cut_img)
        scale = img_open.size[1]/size   #將高變成50  #找出要縮小或放大幾倍
        img_open = img_open.resize((int(img_open.size[0]/scale), size), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img_open)  
        imgs.append(img)
    
    #填入切割照片
    frame_for_img_cut.grid(row=2, column=2)
    # frame_for_img_cut.place(x=420, y=380)
    if len(imgs) <=3:
        for i in range(len(imgs)):
            label = tk.Label(frame_for_img_cut, image=imgs[i])
            label.pack(side=tk.LEFT)
            label.image = imgs[i]
    else:
        for i in range(3):
            label = tk.Label(frame_for_img_cut, image=imgs[i])
            label.pack(side=tk.LEFT)
            label.image = imgs[i]
        tk.Label(frame_for_img_cut, text="•••").pack(side=tk.LEFT)
        
    tk.Label(frame_for_step_p1, text="regional image", pady=8, font=fontStyle_normal_text).grid(row=3, column=2)

    #更改步驟button
    button_for_step.config(text="Step4：Select one regional image", command=choose_region, state=tk.NORMAL)
    

def clahe(img_path):
    button_for_step.config(state=tk.DISABLED)
    #隱藏版面上的分步展示物件
    for widget in frame_for_step_p1.grid_slaves():
        for w in widget.pack_slaves():  #因為在frame_for_step_p1新增了一個frame用pack的方式放切割照片
            w.pack_forget()
        widget.grid_forget()
    frame_for_step_p1.place_forget()

    #填入切割的原圖及說明文字
    frame_for_step_p2.place(x=60, y=80)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    put_step_imgandtext(frame_for_step_p2, 0, 0, img, 200, "regional image", 20)
    
    #箭頭
    arrow(frame_for_step_p2, 0, 1, 50, 90, 10)

    #做clahe和填入圖片及說明文字
    img_clahe = clahe_yuv.clahe_yuv(img)
    img_clahe_name = take_img_name(img_path)
    clahe_path = step_region_clahe_dir+img_clahe_name
    cv2.imencode('.jpg', img_clahe)[1].tofile(clahe_path)
    img_clahe_uint8 = cv2.imdecode(np.fromfile(clahe_path, dtype=np.uint8),-1)
    put_step_imgandtext(frame_for_step_p2, 0, 2, img_clahe_uint8, 200, "clahe", 20)
     
    #更改步驟button
    button_for_step.config(text="Step5：Identify by main model", command=lambda:Threader(big_model_recognize, img_clahe, img), state=tk.NORMAL)


def big_model_recognize(img_clahe, img):
    button_for_step.config(state=tk.DISABLED)
    #箭頭
    arrow(frame_for_step_p2, 0, 3, 50, 90, 10)
    
    #clahe圖片resize送入googlenet辨識
    img_clahe = cv2.resize(img_clahe,(299,299))
    img_clahe = img_clahe.reshape(-1, 299, 299, 3)
    decoded_label, predict_prob = Inception_v4_predict_ttttest.model(model, img_clahe)
    
    #填入googlenet辨識結果
    label = tk.Label(frame_for_step_p2, text=decoded_label, font=fontStyle_decoded_label)
    label.grid(row=0, column=4, ipadx=10)
    
    #小model再次辨識或完成辨識
    if decoded_label == "moth" or decoded_label == "Sitotroga cerealella" or decoded_label == "Pyralidae" or decoded_label == "Tineidae":
        #更改步驟button
        button_for_step.config(text="Step6：Identify by Model A", command=lambda:Threader(little_model_recognize, "moth",img, predict_prob), state=tk.NORMAL)
    elif decoded_label == "Chalcidoidea" or decoded_label == "Formicidae" or decoded_label == "Muscoidea" or decoded_label == "Staphylinidae":
        #更改步驟button
        button_for_step.config(text="Step6：Identify by ModelB", command=lambda:Threader(little_model_recognize, "bee", img, predict_prob), state=tk.NORMAL)
    else:
        #更改步驟button
        button_for_step.config(text="finish", state=tk.DISABLED)


def little_model_recognize(little_model_name, img, predict_prob):
    button_for_step.config(state=tk.DISABLED)
    #箭頭
    arrow(frame_for_step_p2, 2, 1, 50, 135, 10)
    
    #原圖做rgb normalize
    img_rgb_nor = rgb_normalize.rgb_normalize(img)
    #填入圖片及說明文字
    put_step_imgandtext(frame_for_step_p2, 3, 2, img_rgb_nor, 200, "rgb normalization", 20)
    
    #箭頭
    arrow(frame_for_step_p2, 3, 3, 50, 90, 10)
    
    #rgb normalize的圖resize送入little_model辨識
    img_rgb_nor = cv2.resize(img_rgb_nor,(299,299))
    img_rgb_nor = img_rgb_nor.reshape(-1, 299, 299, 3)
    if little_model_name == "moth":
        decoded_label, predict_prob_by_lmodel = Inception_v4_predict_ttttest.model_moth(model_moth, img_rgb_nor)
    elif little_model_name == "bee":
        decoded_label, predict_prob_by_lmodel = Inception_v4_predict_ttttest.model_bee(model_bee, img_rgb_nor)
    
    #填入little_model辨識結果
    label = tk.Label(frame_for_step_p2, text=decoded_label, font=fontStyle_decoded_label)
    label.grid(row=3, column=4)
    
    #更改步驟button
    button_for_step.config(text="Step7：Combined model", command=lambda:Threader(merge_two_model, predict_prob, predict_prob_by_lmodel, little_model_name), state=tk.NORMAL)


def merge_two_model(predict_prob, predict_prob_by_lmodel, little_model_name):
    button_for_step.config(state=tk.DISABLED)
    #箭頭(上)
    arrow(frame_for_step_p2, 0, 5, 50, 135, 10)
    #箭頭(下)
    arrow(frame_for_step_p2, 3, 5, 50, 45, 10)
    
    #合併結果
    if little_model_name == "moth":
        decoded_label = Inception_v4_predict_ttttest.merge_model_and_model_moth(predict_prob, predict_prob_by_lmodel)
    elif little_model_name == "bee":
        decoded_label = Inception_v4_predict_ttttest.merge_model_and_model_bee(predict_prob, predict_prob_by_lmodel)
    
    #填入合併結果
    label = tk.Label(frame_for_step_p2, text=decoded_label, font=fontStyle_decoded_label)
    label.grid(row=1, column=6, rowspan=2, ipadx=20)
    
    #更改步驟button
    button_for_step.config(text="Finish", state=tk.DISABLED)
        

def show(purpose):
    if purpose == "seperate":
        button_for_step.config(text="Step1：Gamma correction", command=lambda:Threader(gamma, path_entry.get()), state=tk.DISABLED)
        button_for_step.place(anchor=tk.NW, height=35, width=260, x=535, y=10)   #放上分步辨識按鈕
        button_for_choose_file.place(anchor=tk.NW, height=35, width=150, x=380, y=10) #放上選擇一張圖片按鈕
    elif purpose == "direct":
        button_for_recognize.config(state=tk.DISABLED)
        button_for_recognize.place(anchor=tk.NW, height=35, width=90, x=515, y=10)  #放上直接辨識按鈕
        button_for_choose_files.place(anchor=tk.NW, height=35, width=130, x=380, y=10) #放上選擇多張圖片按鈕
    path_entry.place(anchor=tk.NW, height=35, width=350, x=20, y=10)
    button_for_rechoose.place(anchor=tk.NW, height=35, width=220, x=830, y=10)
    button_for_seperate.place_forget()
    button_for_direct.place_forget()



def rechoose():
    #隱藏版面上的所有物件
    for widget in root.place_slaves():
        for g in widget.grid_slaves():
            for p in g.pack_slaves():  #因為在frame_for_step_p1新增了一個frame用pack的方式放切割照片
                p.pack_forget()
            g.grid_forget()
        widget.place_forget()
    
    #清除路徑
    e_for_path.set("")
    #出現分步辨識、直接辨識的按鈕
    button_for_seperate.place(anchor=tk.NW, height=100, width=350, x=120, y=250)
    button_for_direct.place(anchor=tk.NW, height=100, width=350, x=600, y=250)


class Threader(threading.Thread):
    def  __init__ (self, func, * args):
        super(). __init__ ()
        
        self.func = func
        self.args = args
        
        self.setDaemon(True)
        self.start()    #在這裡開始
        
    def run(self):
        self.result = self.func(*self.args)
    
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
        
        
#main thread
if __name__ == '__main__':
    
    show_memory_info('initial')
        
    model_path = r"..\\googlenet\\googlenet\\model\\googlenetV4_classification_Origin_run24_0508_cut_clahe3_lr001_shuffle.h5"
    model_path_moth=r"..\\googlenet\\googlenet\\model\\googlenetV4_classification_Origin_run17_0502_rgb_normalize_moth_shuffle.h5"
    model_path_bee = r"..\\googlenet\\googlenet\\model\\googlenetV4_classification_Origin_run21_0504_rgb_normalize_bee_shuffle.h5"
    
    print("載入model...")
    model = load_model(model_path)
    print("載入完成")
    
    show_memory_info('after load_model')
    
    print("載入model_moth...")
    model_moth = load_model(model_path_moth)
    print("載入完成")
    
    show_memory_info('after load_model_moth')
    
    print("載入model_bee...")
    model_bee = load_model(model_path_bee)
    print("載入完成")
    
    show_memory_info('after load_model_bee')

    
    
    root = tk.Tk()
    root.title("Insect Species Identification System")
    # root.geometry('1000x600')
    center_window(root, 1070, 600)
    
    
    #字體大小
    fontStyle_loading = tkFont.Font(family='Microsoft JhengHei UI', size=16, weight='bold')
    fontStyle_normal_text = tkFont.Font(family="Microsoft JhengHei UI", size=11)
    fontStyle_normal_text_bold = tkFont.Font(family="Microsoft JhengHei UI", size=11, weight='bold')
    fontStyle_decoded_label = tkFont.Font(family="Microsoft JhengHei UI", size=16)
    fontStyle_little = tkFont.Font(family="Microsoft JhengHei UI", size=10)
    
    s = ttk.Style()
    s.configure('start.TButton', font = ('Microsoft JhengHei UI','24', 'bold'))
    s.configure('normalbutton.TButton', font = ('Microsoft JhengHei UI','12'))
    s.configure('littlebutton.TButton', font = ('Microsoft JhengHei UI','11'))
    
    ##最一開始的兩個按鈕
    #分步辨識
    button_for_seperate = ttk.Button(root, text="Identify step by step", command=lambda:show("seperate"), style='start.TButton', takefocus=0)
    button_for_seperate.place(anchor=tk.NW, height=100, width=350, x=120, y=250)
    #直接辨識
    button_for_direct = ttk.Button(root, text="Identify directly", command=lambda:show("direct"), style='start.TButton', takefocus=0)
    button_for_direct.place(anchor=tk.NW, height=100, width=350, x=600, y=250)

    
    #檔案路徑 (先定義還沒擺上去)
    e_for_path = tk.StringVar()
    path_entry = ttk.Entry(root, textvariable=e_for_path, state='readonly', font=fontStyle_normal_text, takefocus=0)
    
    
    ##分步辨識
    #選擇一圖、分步辨識、重選展示方式 (先定義還沒擺上去)
    button_for_choose_file = ttk.Button(root, text="select one picture", command=choose_file, style='normalbutton.TButton', takefocus=0)
    button_for_step = ttk.Button(root, style='normalbutton.TButton', takefocus=0)
    button_for_rechoose = ttk.Button(root, text="rechoose identification way", command=rechoose, style='normalbutton.TButton', takefocus=0)
    
    #欲分步展示的圖片 (先定義還沒擺上去)
    label_for_step_img = tk.Label(root)
    #分步展示page1 (先定義還沒擺上去)
    frame_for_step_p1 = tk.Frame(root)
    #座標切割結果 (先定義還沒擺上去)
    frame_for_img_cut = tk.Frame(frame_for_step_p1)
    #分步展示page2 (先定義還沒擺上去)
    frame_for_step_p2 = tk.Frame(root)
    
    
    ##直接辨識
    #選擇圖片、開始辨識、查看本次辨識結果 (先定義還沒擺上去)
    button_for_choose_files = ttk.Button(root, text="Select pictures", command=choose_files, style='normalbutton.TButton', takefocus=0)
    button_for_recognize = ttk.Button(root, text="Identify", command=lambda:recognize(path_entry.get()), style='normalbutton.TButton', takefocus=0, state=tk.DISABLED)
    button_for_allresult = ttk.Button(root, text="Look up all result", command=show_all_result, style='normalbutton.TButton', takefocus=0)
    
    #要辨識的照片、辨識完成的照片 (先定義還沒擺上去)
    label_for_img1 = tk.Label(root)
    
    #辨識結果 (先定義還沒擺上去)
    label_for_result_title = tk.Label(root, font=fontStyle_normal_text_bold)
    label_for_line = tk.Label(root)
    label_for_result_category = tk.Label(root, font=fontStyle_little)
    label_for_result_num = tk.Label(root, font=fontStyle_little) 

    #目前照片名稱 (先定義還沒擺上去)
    label_for_img_name = tk.Label(root)
    
    #目前頁數 (先定義還沒擺上去)
    e_for_page = tk.StringVar()
    page_entry = ttk.Entry(root, textvariable=e_for_page, state='readonly', takefocus=False, font=fontStyle_little)
    
    #總頁數 (先定義還沒擺上去)
    label_for_num_page = tk.Label(root, font=fontStyle_little)    

    #上一張、下一張 (先定義還沒擺上去)
    button_prev = ttk.Button(root, text="Prev", style='littlebutton.TButton', takefocus=0)
    button_next = ttk.Button(root, text="Next", style='littlebutton.TButton', takefocus=0)
    
    #loading的gif (先定義還沒擺上去)
    label_for_loading = tk.Label(root)
    #loading中的字 (先定義還沒擺上去)
    label_for_loading_text = tk.Label(root, font=fontStyle_loading)
    
    
    
    root.protocol('WM_DELETE_WINDOW', closewindow)
    root.mainloop()


    show_memory_info('finish')
    
    del model
    del model_moth
    del model_bee
    gc.collect()
    show_memory_info('finish')
