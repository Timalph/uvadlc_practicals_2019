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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config, inp_len):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Currently using: ', device)
    # Initialize the model that we are going to use
    input_length = inp_len
    input_dim = config.input_dim
    num_classes = config.num_classes
    num_hidden = config.num_hidden
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    
    if config.model_type == 'RNN':
    
        model = VanillaRNN(input_length, input_dim, num_hidden, num_classes
                           , batch_size, device).double()
        
    if config.model_type == 'LSTM':
        model = LSTM(input_length, input_dim, num_hidden, num_classes, batch_size, device).double()
    
    
    
    model = model.to(device)
    
    
    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(inp_len+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)  # fixme
    accuracy_list = []
    loss_list = []

## first 100 steps are to generate the test set
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        output = model.forward(batch_inputs.transpose(0,1).double())

        optimizer.zero_grad()
        
        
        output_indices = torch.argmax(output.transpose(0,1), dim=0)
        

        loss_for_backward = criterion(output,batch_targets).to(device)
        loss_for_backward.backward()
        return output_indices
        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        #print(output.shape)
        #print(batch_targets.shape)
        
        optimizer.step()
        
        #loss = criterion.forward(output, batch_targets)
        
        correct_indices = output_indices == batch_targets
        
        
        
        
        #if step == 4000:
        #    return correct_indices, output_indices, batch_targets, batch_inputs
        accuracy = int(sum(correct_indices))/int(len(correct_indices))

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss_for_backward
            ))
            accuracy_list.append(accuracy)
            loss_list.append(loss_for_backward)

        if step == config.train_steps or (len(accuracy_list) > 10 and (sum(accuracy_list[-3:])
        /len(accuracy_list[-3:])) == 1.0):
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    line = ' '.join((str(config.model_type),'Palindrome length:',str(input_length),'Accuracy:',str(accuracy_list),'Loss', str(loss_list)))
    with open('LSTMMMMM.txt', 'a') as file:
                          file.write(line + '\n')

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5   , help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10050, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    for inp_len in [5]:
        train(config, inp_len)