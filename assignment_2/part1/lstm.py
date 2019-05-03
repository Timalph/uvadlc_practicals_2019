################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_size = num_hidden
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.device = device
        
        tt = torch.tensor
        nrn = np.random.normal    

        self.Wgx = torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size, self.input_dim))))
        self.Wix = torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size, self.input_dim))))
        self.Wfx = torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size, self.input_dim))))
        self.Wox = torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size, self.input_dim))))
        
        self.Wgh =  torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size,self.hidden_size))))
        self.Wih =  torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size,self.hidden_size))))
        self.Wfh =  torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size,self.hidden_size))))
        self.Woh =  torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size,self.hidden_size))))
        
        self.bg = torch.nn.Parameter(torch.zeros(self.hidden_size, 1))
        self.bi = torch.nn.Parameter(torch.zeros(self.hidden_size, 1))
        self.bf = torch.nn.Parameter(torch.zeros(self.hidden_size, 1))
        self.bo = torch.nn.Parameter(torch.zeros(self.hidden_size, 1))
        
        
        self.Whinit = torch.nn.Parameter(torch.zeros(self.hidden_size,1)).to(device)
        self.Whinit = self.Whinit.double()
        
        self.Whx = torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size, self.input_dim))))
        #print(self.Whx.shape)
        #self.Whinit = torch.nn.Parameter(torch.zeros(self.hidden_size,1))
        #print(self.hidden_size)
        self.Whh =  torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size,self.hidden_size))))
        self.bh  = torch.nn.Parameter(torch.zeros(self.hidden_size,1))
        self.Wph = torch.nn.Parameter(torch.ones(self.num_classes,self.hidden_size))
        self.bp = torch.nn.Parameter(torch.zeros(self.num_classes,1))
        
        self.tan = torch.nn.Tanh()
        self.sig = torch.nn.Sigmoid()
        

    def forward(self, x):
        # Implementation here ...
        #print(x.shape)
        #x = x.double()
        for i in range(self.seq_length):
            #print(i)

            if i == 0:  
                self.g = (self.Wgx @x[i].unsqueeze(0)) + (self.Wgh @ self.Whinit) + self.bg
                self.i = (self.Wix @x[i].unsqueeze(0)) + (self.Wih @ self.Whinit) + self.bi
                self.f = (self.Wfx @x[i].unsqueeze(0)) + (self.Wfh @ self.Whinit) + self.bf
                self.o = (self.Wox @x[i].unsqueeze(0)) + (self.Woh @ self.Whinit) + self.bo
                
                self.g = self.tan(self.g)
                self.i = self.sig(self.i)
                self.f = self.sig(self.f)
                self.o = self.sig(self.o)

                self.c = torch.zeros(self.f.shape).double().to(self.device)
                
                
                self.c = (self.g * self.i) + (self.c * self.f)
                self.h = self.tan(self.c) * self.o

                self.h_prev = (self.Whx @ x[i].unsqueeze(0)) + (self.Whh @ self.Whinit) + self.bh
                #print(self.h_prev.shape)
                self.h_prev = self.tan(self.h_prev)
            else:
                self.g = (self.Wgx @x[i].unsqueeze(0)) + (self.Wgh @ self.h) + self.bg
                self.i = (self.Wix @x[i].unsqueeze(0)) + (self.Wih @ self.h) + self.bi
                self.f = (self.Wfx @x[i].unsqueeze(0)) + (self.Wfh @ self.h) + self.bf
                self.o = (self.Wox @x[i].unsqueeze(0)) + (self.Woh @ self.h) + self.bo
                
                self.g = self.tan(self.g)
                self.i = self.sig(self.i)
                self.f = self.sig(self.f)
                self.o = self.sig(self.o)
                
                
                self.c = (self.g * self.i) + (self.c * self.f)
                self.h = self.tan(self.c) * self.o
        #print(self.h_prev.shape)
        #print(self.Wph.shape)
        #print((self.Wph @ self.h_prev).transpose(0,1).shape)
        #print(self.bp.shape)
        p = (self.Wph @ self.h) + self.bp
        
        softy = torch.nn.Softmax()
        y = softy(p)
        
        return y.transpose(0,1)