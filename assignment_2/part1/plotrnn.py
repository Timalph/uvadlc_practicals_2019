# -*- coding: utf-8 -*-
"""
Created on Fri May  3 01:33:15 2019

@author: tim
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:25:06 2019

@author: tim
"""

import csv
import ast
import matplotlib.pyplot as plt

acc_dict = {}

aa = [0.027604166666666666, 0.03854166666666667, 0.0359375, 0.035416666666666666, 0.024479166666666666, 0.027083333333333334, 0.028125, 0.0265625, 0.0296875, 0.022916666666666665, 0.026041666666666668, 0.025520833333333333, 0.028125, 0.0296875, 0.028125, 0.0765625, 0.0640625, 0.0734375, 0.09791666666666667, 0.10885416666666667, 0.1125, 0.1234375, 0.11458333333333333, 0.12135416666666667, 0.1109375, 0.10833333333333334, 0.09895833333333333, 0.10885416666666667, 0.09375, 0.11927083333333334, 0.1125, 0.11979166666666667, 0.11875, 0.11927083333333334, 0.1109375, 0.12239583333333333, 0.12083333333333333, 0.13020833333333334, 0.121875, 0.11510416666666666, 0.1140625, 0.13072916666666667, 0.12604166666666666, 0.1203125, 0.115625, 0.10364583333333334, 0.1203125, 0.121875, 0.1265625, 0.13020833333333334, 0.125, 0.0026041666666666665, 0.11354166666666667, 0.13229166666666667, 0.1234375, 0.12760416666666666, 0.13020833333333334, 0.11458333333333333, 0.1328125, 0.11875, 0.10833333333333334, 0.0015625, 0.13072916666666667, 0.13072916666666667, 0.005208333333333333, 0.12604166666666666, 0.0026041666666666665, 0.12395833333333334, 0.12552083333333333, 0.12083333333333333, 0.12135416666666667, 0.12864583333333332, 0.13177083333333334, 0.0026041666666666665, 0.13020833333333334, 0.12760416666666666, 0.0036458333333333334, 0.11510416666666666, 0.1296875, 0.0026041666666666665, 0.12604166666666666, 0.009895833333333333, 0.12708333333333333, 0.1234375, 0.01875, 0.13020833333333334, 0.007291666666666667, 0.0109375, 0.0, 0.0125, 0.0078125, 0.007291666666666667, 0.0067708333333333336, 0.0, 0.013541666666666667, 0.008854166666666666, 0.13177083333333334, 0.01875, 0.015104166666666667, 0.0036458333333333334, 0.0026041666666666665, 0.11666666666666667, 0.005208333333333333, 0.13125, 0.12395833333333334, 0.0036458333333333334]

RNN_dict = {}
with open('RNN.txt', 'r') as csvfile:
    #truthwriter = csv.reader(csvfile, delimiter ='')
    for row in csvfile:
        #print(row[1000])
        if row[0] == 'L':
            index = row[24]
        else:
            index = row[23:25]
            print(index)
        index1 = row.index('[')
        index2 = row.index(']')
        listi = (row[index1:index2+1])
        acc_list = ast.literal_eval(listi)
        for i in acc_list:
            i = float(i)
        try:
            RNN_dict[index].append(max(acc_list))
        except KeyError:
            RNN_dict[index] = [max(acc_list)]
        #acc_dict[index] = acc_list
        #indexl1 = row.index()

with open('RNN2.txt', 'r') as csvfile:
    #truthwriter = csv.reader(csvfile, delimiter ='')
    for row in csvfile:
        #print(row[1000])
        if row[0] == 'L':
            index = row[24]
        else:
            index = row[23:25]
            print(index)
        index1 = row.index('[')
        index2 = row.index(']')
        listi = (row[index1:index2+1])
        acc_list = ast.literal_eval(listi)
        for i in acc_list:
            i = float(i)
        try:
            RNN_dict[index].append(max(acc_list))
        except KeyError:
            RNN_dict[index] = [max(acc_list)]

with open('RNN3.txt', 'r') as csvfile:
    #truthwriter = csv.reader(csvfile, delimiter ='')
    for row in csvfile:
        #print(row[1000])
        if row[0] == 'L':
            index = row[24]
        else:
            index = row[23:25]
            print(index)
        index1 = row.index('[')
        index2 = row.index(']')
        listi = (row[index1:index2+1])
        acc_list = ast.literal_eval(listi)
        for i in acc_list:
            i = float(i)
        try:
            RNN_dict[index].append(max(acc_list))
        except KeyError:
            RNN_dict[index] = [max(acc_list)]
avg_dict = {}
for key in RNN_dict:
    #print(type(RNN_dict[key][0]))
    avg_dict[key] = sum(RNN_dict[key]) / len(RNN_dict[key])
    
        
# =============================================================================
# xx = []            
# for i in range(len(acc_list)):
#     xx.append(i)
# #print(acc_dict)
# for key in acc_dict.keys():
#     xx = []            
#     for i in range(len(acc_dict[key])):
#         xx.append(i)
#     plt.figure()
#     plt.plot(xx,acc_dict[key], label = key)
#     plt.legend()
#     plt.show
# 
# lenlist = []
# maxlist = []
# 
# for i in acc_dict.keys():
#     lenlist.append(int(i))
#     maxlist.append(max(acc_dict[i]))
# plt.figure()
# plt.plot(lenlist,maxlist)
# plt.legend()
# plt.show    
# =============================================================================

#print(len(xx))

# =============================================================================
bb =[]
aa = []
for key in avg_dict:
    aa.append(avg_dict[key])
    
for i in range(len(avg_dict.keys())):
    bb.append(i)
 
plt.figure()
plt.plot(bb,aa)
plt.show
# =============================================================================
