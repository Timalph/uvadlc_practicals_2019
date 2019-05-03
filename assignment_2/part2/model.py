# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.model = torch.nn.LSTM(input_size = vocabulary_size,
                                   hidden_size = lstm_num_hidden,
                                   num_layers = lstm_num_layers,
                                   batch_first = True)
        
        self.batch_size = batch_size
        self.num_layers = lstm_num_layers
        self.num_hidden = lstm_num_hidden
        
        self.vocabulary_size = vocabulary_size
        self.p_fun = torch.nn.Linear(self.num_hidden, self.vocabulary_size)
    def forward(self, x, h = None):
        # Implementation here...
        # Implementation here ...
        #x = x.transpose(1,2)
        #h0 = torch.zeros(self.num_layers, self.batch_size, self.num_hidden)
        #c0 = torch.zeros(self.num_layers, self.batch_size, self.num_hidden)
        
    
        output, (hn, cn) = self.model(x, h)


        p = self.p_fun(output)

        return p, (hn, cn)
