# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:41:01 2020

@author: User
"""

import os

path = './'
txt_list = []
all_list_name = os.listdir(path)
for x in range(len(all_list_name)):
    if ".txt"in all_list_name[x]:
        txt_list.append(all_list_name[x])
#print(txt_list)

outfile = open("category.txt","w", encoding="utf-8")

CATEGORY = []
for k in txt_list:
    infile = open(k, "r", encoding = "utf-8")
    
    words = infile.read().splitlines()
    #print(words)
    lst = [ln.split(' ')[-1] for ln in words]
    print("lst:\n", lst)
    
    for i in lst:
        #print(i)
        if i not in CATEGORY:
            CATEGORY.append(i)
    
    print(CATEGORY)

for i in CATEGORY:
    outfile.writelines(i)
    outfile.writelines("\n")

infile.close()
outfile.close()






