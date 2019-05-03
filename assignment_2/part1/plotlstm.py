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


RNN_dict = {}
for file in ['LSTMMMMMlisa.txt','LSTM_good.txt','LSTM.txt','LSTMMMMM']:
    
    with open('LSTMMMMMlisa.txt', 'r') as csvfile:
        #truthwriter = csv.reader(csvfile, delimiter ='')
        for row in csvfile:
            #print(row[1000])
            if row[0] == 'L':
                index = row[24:26]
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

# =============================================================================
# for filenumber in [2,3,4,5]:
#     file = 'RNN' + str(filenumber) + '.txt'
# 
#     with open(file, 'r') as csvfile:
#         #truthwriter = csv.reader(csvfile, delimiter ='')
#         for row in csvfile:
#             #print(row[1000])
#             if row[0] == 'L':
#                 index = row[24]
#             else:
#                 index = row[23:25]
#                 print(index)
#             index1 = row.index('[')
#             index2 = row.index(']')
#             listi = (row[index1:index2+1])
#             acc_list = ast.literal_eval(listi)
#             for i in acc_list:
#                 i = float(i)
#             try:
#                 RNN_dict[index].append(max(acc_list))
#             except KeyError:
#                 RNN_dict[index] = [max(acc_list)]
# 
# =============================================================================
# =============================================================================
# with open('RNN3.txt', 'r') as csvfile:
#     #truthwriter = csv.reader(csvfile, delimiter ='')
#     for row in csvfile:
#         #print(row[1000])
#         if row[0] == 'L':
#             index = row[24]
#         else:
#             index = row[23:25]
#             print(index)
#         index1 = row.index('[')
#         index2 = row.index(']')
#         listi = (row[index1:index2+1])
#         acc_list = ast.literal_eval(listi)
#         for i in acc_list:
#             i = float(i)
#         try:
#             RNN_dict[index].append(max(acc_list))
#         except KeyError:
#             RNN_dict[index] = [max(acc_list)]
#             
# =============================================================================
            
            
            
            
            
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
    
for i in avg_dict.keys():
    bb.append(int(i))
 
plt.figure()
plt.plot(bb,aa, label = 'Accuracy')
plt.xlabel('Length of Palindrome')
plt.ylabel('Max Accuracy')

#plt.legend()

plt.show
# =============================================================================
