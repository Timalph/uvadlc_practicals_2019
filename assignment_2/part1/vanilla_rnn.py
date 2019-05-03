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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_size = num_hidden
        self.num_classes = num_classes
        self.input_dim = input_dim

        tt = torch.tensor
        nrn = np.random.normal

        
        
        
        self.Whx = torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size, self.input_dim))))

        self.Whinit = torch.nn.Parameter(torch.zeros(self.hidden_size,1))

        self.Whh =  torch.nn.Parameter(tt(nrn(0,0.0001,(self.hidden_size,self.hidden_size))))
        self.bh  = torch.nn.Parameter(torch.zeros(self.hidden_size,1))
        self.Wph = torch.nn.Parameter(torch.ones(self.num_classes,self.hidden_size))
        self.bp = torch.nn.Parameter(torch.zeros(self.num_classes,1))
        
        self.tan = torch.nn.Tanh()
        
        

    def forward(self, x):
        for i in range(self.seq_length):
            if i == 0:
                self.h_prev = (self.Whx @ x[i].unsqueeze(0)) + (self.Whh @ self.Whinit) + self.bh
                self.h_prev = self.tan(self.h_prev)
            else:
                self.h_prev = (self.Whx @ x[i].unsqueeze(0)) + (self.Whh @ self.h_prev) + self.bh
                self.h_prev = self.tan(self.h_prev)

        p = (self.Wph @ self.h_prev) + self.bp

        y = p
        return y.transpose(0,1)
