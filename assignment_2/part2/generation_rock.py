# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:56:42 2019

@author: tim
"""
import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TextDataset
from model import TextGenerationModel
import sys

batch_size = 64
seq_length = 30




dataset = TextDataset('PP.txt',seq_length)
device = 'cpu'

model = TextGenerationModel(batch_size = batch_size,
                                seq_length = seq_length,
                                vocabulary_size = dataset.vocab_size,
                                lstm_num_hidden = 128,
                                lstm_num_layers = 2)

optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.005)




#filename = ''.join((str(epoch), str(step), '.pth'))
checkpoint = torch.load('model.pth')    
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()


def string_generator(pre,model):
   
    zerol = torch.zeros([1,30,dataset.vocab_size])
    one_hot = torch.tensor(pre).unsqueeze(-1).unsqueeze(-1)
    zerol.scatter_(2,one_hot,1)
    print(zerol.shape)
    
        
        
        if i == 0:
            output, h = model.forward(zerol)
        else:
            output, h = model.forward(zerol, h)
            
    return output, h

output,h = string_generator(pre,zerol,model)

soft = torch.nn.Softmax(dim=2)
 
zerol = torch.zeros([1,1,dataset.vocab_size])
one_hot_letter = torch.tensor(output).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
zerol.scatter_(2,one_hot_letter,1)
zerol = zerol.to(device)





    if i == 0:
        output, h = model.forward(zerol)
 
    else:
        output, h = model.forward(zerol, h)
     
    tempered = soft(output/temp_value)
    #print(tempered)
    output = int(torch.multinomial(tempered[0][0],1).detach())
    #print(output)
    letters.append(output)
the_string =  dataset.convert_to_string(letters)
abs_step = 1
 
line = ' '.join(('Step:',str(abs_step),'Temperature:'
                 ,str(temp_value), 'Text:',the_string))

with open('Finishers.txt', 'a') as file:
              file.write(line + '\n')
        
        
        
        
        