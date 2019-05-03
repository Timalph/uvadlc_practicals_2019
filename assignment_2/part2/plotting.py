# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:26:52 2019

@author: tim
"""

import ast
import matplotlib.pyplot as plt
import numpy as np

with open('PrideAndPrejudice.txt', 'r') as csvfile:
    #truthwriter = csv.reader(csvfile, delimiter ='')
    for row in csvfile:
        #print(row[1000])

        index1 = row.index('[')
        index2 = row.index(']')
        listi = (row[index1:index2+1])
        acc_list = ast.literal_eval(listi)
        for i in acc_list:
            i = float(i)
        lindex1 = row.index('L') + 5

        lindex2 = -1
        lossti = (row[lindex1:lindex2])
        loss_list = ast.literal_eval(lossti)
        for i in loss_list:
            i = float(i)



xx = []            
for i in range(len(acc_list)):
    xx.append(i)
#print(acc_dict)

plt.figure()
plt.plot(np.array(xx)*100,acc_list)

plt.xlabel('Step #')
plt.ylabel('Loss')

plt.legend()



plt.show

aa = []
for i in range(len(loss_list)):
    aa.append(i)
#print(acc_dict)

plt.figure()
plt.plot(np.array(aa)*100,loss_list, color='#FF7D33')
         
plt.xlabel('Step #')
plt.ylabel('Loss')
         
         
plt.legend()
plt.show