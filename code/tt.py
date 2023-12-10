import os

#print(os.getcwd())
path_name = './' #'D:/Yolo_mark-master/x64/Release/data/img'
img_list_3=[]
all_list_name = os.listdir(path_name)
for x in range(len(all_list_name)):
    if ".JPG"in all_list_name[x] or ".jpg" in all_list_name[x]:
        img_list_3.append(all_list_name[x])
print(img_list_3)
print("Total:"+str(len(img_list_3)))

txt_path = './training_1016_clahe_cut_train.txt'#'D:/Yolo_mark-master/x64/Release/data/test.txt'
txt_open = open(txt_path,'w', encoding="utf-8")
for name in img_list_3:
    #txt_open.writelines("data"+"\\"+"img"+"\\"+"1016_gamma0_8_train_cut"+"\\")
    txt_open.writelines(name)
    txt_open.writelines("\n")    
txt_open.close()