# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 01:55:21 2020

@author: vicky
"""

import os
import string

path1 = 'D:/標記完成(只有label)/'  #每張照片蚊蟲標記檔
path2 = 'D:/org_label/'  #原始yolo標記檔
path3 = 'D:/new_label/'  #更改後yolo標記檔
allFileList = os.listdir(path1)  #列出指定路徑底下所有檔案(包含資料夾)


category = []
f = open('category.txt','r', encoding='utf-8')
for line in f.readlines():
    s = line.strip()
    category.append(s)


for file in allFileList: 
    print(file)
    temp_index = []
    with open(path1 + file, 'r', encoding='utf-8') as f1:
#        print(file)
        for line in f1.readlines():
            s = line.lstrip(string.digits).strip()
            if s != "":   #避免s是空白行
                temp_index.append(category.index(s))

    with open(path2 + file, 'r', encoding='utf-8') as f2:
        with open(path3 + file, 'w') as f3:
            for i, line in enumerate(f2.readlines()):
                if len(temp_index) != 0:   #避免temp_index沒有任何東西
                    new = line.strip().replace(line[0], str(temp_index[i]), 1)
                    f3.write(new+"\n")
                else:
                    f3.write(line)
