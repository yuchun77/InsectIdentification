# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:46:58 2021

@author: User
"""

#way2
class GlobalVar:
    # 建構式
    def __init__(self, value):
        self.value = value
    
    def set_value(self, value):
        self.value = value

    def get_value(self):
         return self.value
 

def init(value):
    variable = GlobalVar(value)
    return variable







