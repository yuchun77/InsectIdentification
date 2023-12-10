# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:48:46 2021

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:49:46 2021

@author: vicky
"""

from tensorflow.keras.models import load_model

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk, ImageSequence
import os
import shutil
import psutil
import gc


import sys
sys.path.append(r"C:\Users\vicky\Desktop\Insect_Species_Identification_System\yolo_pre")
import yolo_pre
sys.path.append(r"C:\Users\vicky\Desktop\Insect_Species_Identification_System\googlenet\googlenet")
import Inception_v4_predict_ttttest
import global_global

from datetime import datetime
import time
import threading
import pandas as pd
import prettytable as pt



#定義global變數
big_img_list = []
all_img_decoded_label = []
obj_state = global_global.init("")

upload_img_dir = ""
yolo_draw_dir = ""
cut_dir =""
final_dir = ""
upload_dir = ""

selectFilePaths = ""

category = Inception_v4_predict_ttttest.get_category()



# 顯示當前 python 程式佔用的記憶體大小
def show_memory_info(hint):
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    memory = info.rss / 1024. / 1024
    print('{} memory used: {} MB'.format(hint, memory))
    


def closewindow():
    ans = messagebox.askyesno(title='Warning' ,message='Close the window?')
    if ans:
        root.destroy()
    else:
        return


def put_img(img_open, img_size, label):   #img要為PIL格式
    scale = img_open.size[0]/img_size    #找出要縮小或放大幾倍
    img_open = img_open.resize((img_size, int(img_open.size[1]/scale)), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img_open)  
    label.config(image=img)  
    label.image = img  #keep a reference
    return int(img_open.size[1])


def show_img(use, page):
    if use == "preview":
        path = selectFilePaths[page-1]   #page-1才會是對應的index
    elif use == "recognize":
        path = final_dir+big_img_list[page-1]
    
    #填入照片
    img_open = Image.open(path)
    put_img(img_open, 750, label_for_img1)
    
    #填入照片名稱
    label_for_img_name.config(text=path)


def prev_img(use, current_page):
    if current_page > 1:  #頁數比1大才能上一張
        e_for_page.set(current_page-1)
        show_img(use, current_page-1)   #要show現在頁數的上一張
        if use == "recognize":
            show_one_img_result(current_page-1)
    else:
        messagebox.showinfo("提示", "此為第一張照片")


def next_img(use, current_page):  #頁數比總頁數小才能下一張
    if current_page < len(selectFilePaths):
        e_for_page.set(current_page+1)
        show_img(use, current_page+1)  #要show現在頁數的下一張
        if use == "recognize":
            show_one_img_result(current_page+1)
    else:
        messagebox.showinfo("提示", "此為最後一張照片")


    
def choose_file():  #選擇照片並展示
    global selectFilePaths
    selectFilePaths = filedialog.askopenfilenames(title='選擇圖片')  #獲得圖片路徑
    global obj_state
    obj_state.set_value("")
    if selectFilePaths == "":   #按了選擇圖案又不選
        temp = path_entry.get()
        e_for_path.set(temp)   #填回原本的路徑
    else:
        #填入所選取的各個照片路徑
        e_for_path.set(selectFilePaths) 
        
        #填入預覽的照片
        show_img("preview", 1)
                
        #這時才出現上一張、下一張的按鈕
        button_prev.place(anchor=tk.SW, height=35, width=80, x=800, y=560)
        button_prev.config(command=lambda:prev_img("preview", int(e_for_page.get())))
        button_next.place(anchor=tk.SW, height=35, width=80, x=900, y=560)
        button_next.config(command=lambda:next_img("preview", int(e_for_page.get())))
          
        #出現並設定目前頁數
        page_entry.place(anchor=tk.NW, height=20, width=20, x=880, y=570)
        e_for_page.set(1)
        
        #填入共幾張
        label_for_num_page.config(text="共"+str(len(selectFilePaths))+"張")
        
        #再次選擇照片時，將原本的辨識結果清除
        label_for_result_category['text'] = ""
        label_for_result_num['text'] = ""



def remove_symbol(a):   #用來拆開 e_path_entry.get()
        a = a.split(" ")
        return a


def setup_dir(upload_dir):  #建立接下來需要的各種資料夾
    paths = ["img\\", "draw\\", "cut\\", "final\\"]
    abs_paths = []
    for path in paths:
        os.makedirs(upload_dir+path, exist_ok=True)  #創建upload系列資料夾  #可創建多層的資料夾，如果前一層資料夾不存在，程式會自動幫你建立
        abs_paths.append(upload_dir+path)
    abs_paths.append(upload_dir)
    return abs_paths


def show_one_img_result(page):
    #統計各類別有幾隻
    result = pd.DataFrame(category, columns=["種類"])
    dict_num = {'數量':[0]*len(category)}
    
    for insect in all_img_decoded_label[page-1]:   ##page-1才會是對應的index
        index = category.index(insect)
        dict_num["數量"][index] += 1
        
    result["數量"] = dict_num["數量"]
    
    #種類table
    tb_cat = pt.PrettyTable()
    tb_cat.add_column("種類", result["種類"])
    tb_cat.set_style(pt.PLAIN_COLUMNS)
    tb_cat.align = 'l'
    
    #數量table
    tb_num = pt.PrettyTable()
    tb_num.add_column("數量", result["數量"])
    tb_num.set_style(pt.PLAIN_COLUMNS)
    tb_num.align = 'c'
    
    #填入辨識結果
    label_for_result_category.config(text=tb_cat)
    label_for_result_num.config(text=tb_num)
    

def recognize(file_paths, model, model_moth, model_bee):
    if file_paths == "":  #還沒選照片就按開始辨識
        messagebox.showinfo("提示", "請選擇照片")
    else:
        #把上一張、下一張、頁數、總頁數先隱藏起來
        button_prev.place_forget()
        button_next.place_forget()
        page_entry.place_forget()
        label_for_num_page.config(text="")
        label_for_img_name.config(text="")
        
        #把黏在一起的各個檔案路徑分開
        file_paths_list = remove_symbol(file_paths)
        print(file_paths_list)
        
        #做出upload資料夾
        global upload_img_dir, yolo_draw_dir, cut_dir, final_dir, upload_dir
        time = datetime.now().strftime('%Y%m%d%H%M%S')  #取得現在時間
        upload_dir = "D:\\GUI\\upload\\upload_"+time+"\\"  #結尾有沒有加\\都可以
        dirs = setup_dir(upload_dir)
        upload_img_dir, yolo_draw_dir, cut_dir, final_dir, upload_dir = dirs
        
        #上傳檔案至upload資料夾
        for i in range(len(file_paths_list)):
            #取得img_name
            for j in range(len(file_paths_list[i])):
                last = len(file_paths_list[i])-1
                if '/' == file_paths_list[i][last-j]:
                    img_name = file_paths_list[i][(last-j)+1:]
                    break
        
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
        button_prev.place(anchor=tk.SW, height=35, width=80, x=800, y=560)
        button_prev.config(command=lambda:prev_img("recognize", int(e_for_page.get())))
        button_next.place(anchor=tk.SW, height=35, width=80, x=900, y=560)
        button_next.config(command=lambda:next_img("recognize", int(e_for_page.get())))
        
        #出現並設定目前頁數
        page_entry.place(anchor=tk.NW, height=20, width=20, x=880, y=570)
        e_for_page.set(1)

        #填入辨識結果
        show_one_img_result(1)  #第1張照片的結果(從page1開始)
        
        #填入共幾張
        label_for_num_page.config(text="共"+str(len(big_img_list))+"頁")



#分解gif並逐貞顯示
def loading_gif(obj_state):
    label_for_img1.config(image="")
    label_for_loading.place(x=450, y=240)
    label_for_loading_text.config(text="辨 識 中 •••")
    label_for_loading_text.place(x=450, y=365)
    while obj_state.get_value() == "not finish":
        loading_img = Image.open('D:\\GUI\\img\\loading5.gif')
        # GIF圖片流的迭代器
        iter = ImageSequence.Iterator(loading_img)
        #frame就是gif的每一帧，轉換下格式就能顯示了
        for frame in iter:
            img = ImageTk.PhotoImage(frame)
            label_for_loading.config(image=img)
            time.sleep(0.1)
            root.update_idletasks()  #刷新
            root.update()
    label_for_loading.place_forget()
    label_for_loading_text.place_forget()



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
        
    model_path = r"C:\\Users\\vicky\\Desktop\\Insect_Species_Identification_System\\googlenet\\googlenet\\model\\googlenetV4_classification_Origin_run24_0508_cut_clahe3_lr001_shuffle.h5"
    model_path_moth=r"C:\\Users\\vicky\\Desktop\\Insect_Species_Identification_System\\googlenet\\googlenet\\model\\googlenetV4_classification_Origin_run17_0502_rgb_normalize_moth_shuffle.h5"
    model_path_bee = r"C:\\Users\\vicky\\Desktop\\Insect_Species_Identification_System\\googlenet\\googlenet\\model\\googlenetV4_classification_Origin_run21_0504_rgb_normalize_bee_shuffle.h5"
    
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
    root.title("蚊蟲辨識系統")
    root.geometry('1000x600')
    
    
    #檔案路徑
    e_for_path = tk.StringVar()
    path_entry = tk.Entry(root, textvariable=e_for_path, state='readonly')
    path_entry.place(anchor=tk.NW, height=35, width=350, x=20, y=10)
    
    
    #選擇圖片、開始辨識
    button_choose = tk.Button(root, text="選擇圖片", command=choose_file)
    button_choose.place(anchor=tk.NW, height=35, width=80, x=380, y=10)
    button_start = tk.Button(root, text="開始辨識", command=lambda:recognize(path_entry.get(), model, model_moth, model_bee))
    button_start.place(anchor=tk.NW, height=35, width=80, x=470, y=10)
    
    
    #要辨識的照片、辨識完成的照片
    label_for_img1 = tk.Label(root)
    label_for_img1.place(x=20, y=90)
    
    
    #辨識結果
    label_for_result_category = tk.Label(root)    
    label_for_result_category.place(width=120, x=800, y=90)
    label_for_result_num = tk.Label(root)    
    label_for_result_num.place(x=920, y=90)
    
    
    #目前照片名稱
    label_for_img_name = tk.Label(root)    
    label_for_img_name.place(x=20, y=60)

    
    #目前頁數 (先定義還沒擺上去)
    e_for_page = tk.StringVar()
    page_entry = tk.Entry(root, textvariable=e_for_page, state='readonly')
    
    
    #總頁數
    label_for_num_page = tk.Label(root)    
    label_for_num_page.place(x=910, y=570)

    
    #上一張、下一張 (先定義還沒擺上去)
    button_prev = tk.Button(root, text="上一張")
    button_next = tk.Button(root, text="下一張")
    
    #loading的gif (先定義還沒擺上去)
    label_for_loading = tk.Label(root)
    #loading中的字 (先定義還沒擺上去)
    fontStyle = tkFont.Font(family='microsoft yahei', size=16, weight='bold')
    label_for_loading_text = tk.Label(root, font=fontStyle)
    
    
    root.protocol('WM_DELETE_WINDOW', closewindow)
    root.mainloop()


    show_memory_info('finish')
    del model
    del model_moth
    del model_bee
    gc.collect()
    show_memory_info('finish')
