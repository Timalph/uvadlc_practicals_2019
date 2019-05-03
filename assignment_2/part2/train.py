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

########FIX THE ARGPARSER SO THEY CAN RUN IT
########FIX THE TEXTFILE
########UNCOMMENT THE TORCH>DEVICE THING
torch.manual_seed(42)
np.random.seed(42)
################################################################################
def train(config):

    # Initialize the device which to run the model on
    #device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file,config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    

        # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size = config.batch_size,
                                seq_length = config.seq_length,
                                vocabulary_size = dataset.vocab_size)  # fixme
    
    device = 'cuda'

    model.to(device)
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, amsgrad = True)  # fixme
    
    count = 0
    
    train_list_in = []
    train_list_ta = []
    
    test_list_in = []
    test_list_ta = []
    
    #for step, (batch_inputs, batch_targets) in enumerate(data_loader):
    #    print(step, len(train_list_in))
    #    if count%10 == 0:
    #        test_list_in.append(batch_inputs)
    #        test_list_ta.append(batch_targets)
    #    else:
    #        train_list_in.append(batch_inputs)
    #        train_list_ta.append(batch_targets)
    #    count += 1
    #    if len(train_list_in) == 200:
    #        break
    #return test_list_in
    #print(len(train_list_in), len(train_list_ta), len(test_list_in), len(test_list_ta))
    #return 'a'
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        
        
        
        if step % config.print_every != 0 or step == 0:

            t1 = time.time()
            #print(type(step))
            
            
            model.train()
            
            #######################################################
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            
            optimizer.zero_grad()
            
            
            batch_inputz = torch.stack(batch_inputs).to(device)
            batch_input = batch_inputz.transpose(1,0).to(device)
            
            
            zerox = torch.zeros(batch_input.shape[0],
                            batch_input.shape[1],
                            dataset.vocab_size).to(device)
        
            zerox.scatter_(2,batch_input.unsqueeze(2),1).to(device)

            output = model.forward(zerox).to(device)
            #print(output[1])
            targets = torch.stack(batch_targets).to(device)
    
            output_indices = torch.argmax(output, dim=2).to(device)
            output =  output.transpose(0,1).transpose(1,2).to(device)
            #print(output.shape)
            loss_for_backward = criterion(output,targets).to(device)
            #print(targets.shape)
            #return 'a'
            loss_for_backward.backward()
            optimizer.step()
            
            correct_indices = output_indices == targets.transpose(0,1).to(device)
            
            #return correct_indices
            #######################################################
    
            #loss = criterion.forward(output, targets)
        
            
            #accuracy = int(sum(sum(correct_indices)))/int(correct_indices.shape[0]*
            #correct_indices.shape[1])
            #print(type(accuracy),type(loss))
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0 and step != 0:
# =============================================================================
#             acc_list = []
#             for i in range(len(test_list_in)):
#                 batch_inputz = torch.stack(test_list_in[i]).to(device)
#                 batch_input = batch_inputz.transpose(1,0).to(device)
#             
#             
#                 zerox = torch.zeros(batch_input.shape[0],
#                             batch_input.shape[1],
#                             dataset.vocab_size).to(device)
#         
#                 zerox.scatter_(2,batch_input.unsqueeze(2),1).to(device)
#                 
#                 output = model.forward(zerox).to(device)
#                 targets = torch.stack(test_list_ta[i]).to(device)
#         
#                 output_indices = torch.argmax(output, dim=2).to(device)
#                 output =  output.transpose(0,1).transpose(1,2).to(device)
#                 correct_indices = output_indices == targets.transpose(0,1).to(device)
#                 accuracy = int(sum(sum(correct_indices)))/int(correct_indices.shape[0]*
#                               correct_indices.shape[1])
#                 acc_list.append(accuracy)
# =============================================================================
            model.eval()
            
            batch_inputz = torch.stack(batch_inputs).to(device)
            batch_input = batch_inputz.transpose(1,0).to(device)
            
            
            zerox = torch.zeros(batch_input.shape[0],
                            batch_input.shape[1],
                            dataset.vocab_size).to(device)
        
            zerox.scatter_(2,batch_input.unsqueeze(2),1).to(device)

            output = model.forward(zerox).to(device)
            output_indices = torch.argmax(output, dim=2).to(device)
            output =  output.transpose(0,1).transpose(1,2).to(device)
            targets = torch.stack(batch_targets).to(device)
            
            
            loss_for_backward = criterion(output,targets).to(device)
            correct_indices = output_indices == targets.transpose(0,1).to(device)
            
            #accuracy = sum(acc_list) / len(acc_list)
            accuracy = int(sum(sum(correct_indices)))/int(correct_indices.shape[0]*
            correct_indices.shape[1])
            
            print("[{}] Train Step {:04d}/{:f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy,
                    loss_for_backward
            ))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            ## Generate a good sample instead of the same one over and over again
            model.eval()
            
            batch_inputz = torch.stack(batch_inputs).to(device)
            batch_input = batch_inputz.transpose(1,0).to(device)
            zerox = torch.zeros(batch_input.shape[0],
                            batch_input.shape[1],
                            dataset.vocab_size).to(device)
            zerox.scatter_(2,batch_input.unsqueeze(2),1).to(device)
            output = model.forward(zerox).to(device)
            output_indices = torch.argmax(output, dim=2).to(device)
            output =  output.transpose(0,1).transpose(1,2).to(device)
            targets = torch.stack(batch_targets).to(device)
            loss_for_backward = criterion(output,targets).to(device)
            correct_indices = output_indices == targets.transpose(0,1).to(device)
            
            best_sample = np.argmax(np.asarray(sum(correct_indices.t().detach().cpu())))
            print('Real: ', dataset.convert_to_string(np.asarray(batch_input[best_sample].cpu())))
            output = model.forward(zerox).to(device)
            output_indices = torch.argmax(output, dim=2).to(device)
            print('prediction: ', dataset.convert_to_string(np.asarray(output_indices[best_sample].cpu())))
            
            bc = int(sum(correct_indices.t().detach().cpu())[best_sample])/config.seq_length
            if bc > 0.8:
                print(bc)
                return correct_indices
            
            print('This sample had:',bc,'characters right')
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default = 'PP.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=32, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
