# Insect Identification

這是我在大學期間所做的專題，蚊蟲種類辨識系統。

共使用兩種 model: YOLO-v4 & GoogLeNet Inception-v4. YOLO-v4 用來找出圖片中「是蚊蟲」的部分，GoogLeNet Inception-v4 用來分辨被框出的蚊蟲種類為何，一共要辨識 24 種蚊蟲。

### 資料前處理
在將圖片餵給 model 之前會做資料前處理：
1. Gamma-Correction: 

    `cd code`  
    `python3 gamma_correction.py`
2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    `python3 clahe_yuv.py`
    
### 使用者介面 GUI
`cd GUI`

`python3 tkinter_seperate.py`

