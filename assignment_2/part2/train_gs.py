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

def create_zerox(batch_inputs,v, device):
# =============================================================================
#     batch_inputz = torch.stack(batch_inputs).to(device)
#     
#     
#     
#     batch_input = batch_inputz.transpose(1,0).to(device)
#     
#     zerox = torch.zeros(batch_input.shape[0],
#                     batch_input.shape[1],
#                     v).to(device)
#     
#     zerox.scatter_(2,batch_input.unsqueeze(2),1).to(device)
#     return zerox
# =============================================================================
     batch_input = torch.stack(batch_inputs).to(device)
     
     
     zerox = torch.zeros(batch_input.shape[0],
                     batch_input.shape[1],
                     v).to(device)
     
     zerox.scatter_(2,batch_input.unsqueeze(2),1).to(device)
     
     zerox = zerox.transpose(1,0).to(device)

     return zerox    
    
    
def train(config, lr):

    # Initialize the device which to run the model on
    #device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file,config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    

        # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size = config.batch_size,
                                seq_length = config.seq_length,
                                vocabulary_size = dataset.vocab_size,
                                lstm_num_hidden = config.lstm_num_hidden,
                                lstm_num_layers = config.lstm_num_layers
                                
                                
                                )  # fixme
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Currently using: ', device)

    model = model.to(device)
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # fixme
    #optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, amsgrad = True)  # fixme
    #optimizer = torch.optim.Adam(model.parameters(), lr = lr, amsgrad = True)
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)
    
    acc_list = []
    loss_list = []
    
    test_batches_in = []
    test_batches_ta = []
    
    test_acc = []
    
    
    ### Flag for temperature
    temp = True
    temp_value = 2
    
    
    checkpoint = torch.load('model.pth')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        
        
        if step % config.print_every != 0 or step == 0:

            t1 = time.time()
            #print(type(step))
            
            
            #model.train()
            
            #######################################################
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

            
            zerox = create_zerox(batch_inputs, dataset.vocab_size, device)
                
            
            output, _ = model.forward(zerox)#.to(device)


            targets = torch.stack(batch_targets).to(device)
    
            output_indices = torch.argmax(output, dim=2).to(device)
            
            output =  output.transpose(0,1).transpose(1,2).to(device)
            
            
            #print(output.shape, targets.shape)
            #return 'a'
            
            #print(output.transpose(0,2).shape, targets.t().shape)
            #return 'a'
            loss_for_backward = criterion(output.transpose(0,2),targets.t()).to(device)

            optimizer.zero_grad()
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
            #model.eval()
            
            zerox = create_zerox(batch_inputs, dataset.vocab_size, device)

            output, _ = model.forward(zerox)
            
            
            
            output_indices = torch.argmax(output, dim=2).to(device)
            
            
            
            
            output =  output.transpose(0,1).transpose(1,2).to(device)
            targets = torch.stack(batch_targets).to(device)
            
            
            #loss_for_backward = criterion(output,targets).to(device)
            loss_for_backward = criterion(output.transpose(0,2),targets.t()).to(device)
            correct_indices = output_indices == targets.transpose(0,1)#.to(device)
            #return output_indices, targets.transpose(0,1)
            
            
            
            
            
            
            #print(correct_indices.shape)
            #accuracy = sum(acc_list) / len(acc_list)
            #accuracy = int(sum(sum(correct_indices)))/int(correct_indices.numel())
            accuracy = np.array(correct_indices.detach().cpu()).mean()
            
            
            
            
            print("[{}] Train Step {:04d}/{:f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy,
                    loss_for_backward
            ))
            acc_list.append(accuracy)
            loss_list.append(int(loss_for_backward))
            

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            ## Generate a good sample instead of the same one over and over again
            #model.eval()
            
            ### Append every modulo batch to a list of test batches and run 
            ### over that list to test
            
            zerox = create_zerox(batch_inputs, dataset.vocab_size, device)
            
            test_batches_in.append(zerox)
            
            targets = torch.stack(batch_targets).to(device)
            
            test_batches_ta.append(targets)
            
            batch_inputz = torch.stack(batch_inputs).to(device)
            batch_input = batch_inputz.transpose(1,0).to(device)
            
            output,_ = model.forward(zerox)#.to(device)
            output_indices = torch.argmax(output, dim=2).to(device)
            output =  output.transpose(0,1).transpose(1,2).to(device)

            loss_for_backward = criterion(output,targets).to(device)
            correct_indices = output_indices == targets.transpose(0,1).to(device)
            
            best_sample = np.argmax(np.asarray(sum(correct_indices.t().detach().cpu())))
            print('Real: ', dataset.convert_to_string(np.asarray(batch_input[best_sample].cpu())))
            output,_ = model.forward(zerox)#.to(device)
            output_indices = torch.argmax(output, dim=2).to(device)
            print('prediction: ', dataset.convert_to_string(np.asarray(output_indices[best_sample].cpu())))
            
            bc = int(sum(correct_indices.t().detach().cpu())[best_sample])/config.seq_length
            print('This sample had:',bc,'characters right')
            
            output = np.random.randint(dataset.vocab_size)
            letters = [output]
            
            for i in range(config.seq_length-1):
                
                if temp:
                    
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
                    output = int(torch.multinomial(tempered[0][0],1).detach().cpu())
                    #print(output)
                    letters.append(output)
        
                else:
                    
                    zerol = torch.zeros([1,1,dataset.vocab_size])
                    one_hot_letter = torch.tensor(output).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    zerol.scatter_(2,one_hot_letter,1)
                    zerol = zerol.to(device)
                    
                    if i == 0:
                        output, h = model.forward(zerol)
                    else:
                        output, h = model.forward(zerol, h)
                        
                    output = int(torch.argmax(output, dim=2).detach().cpu())
                    letters.append(output)
            abs_step = (runs*10000) + step
            line = ' '.join(('Step:',str(abs_step),dataset.convert_to_string(letters)))
            
            
            with open('GreedyGeneration.txt', 'a') as file:
                          file.write(line + '\n')

            
            
            
            
# =============================================================================
#         if step % (config.sample_every*1000) ==0:
#             avg = []
#             print('Testing over ', len(test_batches_in), 'batches')
#             for z in range(len(test_batches_in)):
#                 ##OUTPUT
#                 output,_ = model.forward(test_batches_in[z])
#                 output_indices = torch.argmax(output, dim=2).to(device)
#                 output =  output.transpose(0,1).transpose(1,2).to(device)
#                 
#                 ##LOSS AND ACCURACY
#                 loss_for_backward = criterion(output,targets).to(device)
#                 correct_indices = output_indices == test_batches_ta[z].transpose(0,1).to(device)
#                 
#                 accuracy = int(sum(sum(correct_indices)))/int(correct_indices.shape[0]*
#                               correct_indices.shape[1])
#                 
#                 avg.append(accuracy)
#                 
#             this_test_acc = sum(avg)/len(avg)
#             print('The test accuracy over ',len(test_batches_in), 'is: ', this_test_acc)
#             test_acc.append(this_test_acc)
#             #if bc > 0.8:
#             #    print(bc)
#             #    #return correct_indices
#             
# =============================================================================
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    line = ' '.join(('Test accuracy:',str(test_acc.append),'Learning rate:',str(lr),'Accuracy:',str(acc_list),'Loss:',str(loss_list)))
    with open('textresults.txt', 'a') as file:
                          file.write(line + '\n')
    
    #hiddenstates = [None]*30
    output = np.random.randint(dataset.vocab_size)
    letters = [output]
    for i in range(400):
            zerol = torch.zeros([1,1,dataset.vocab_size])
            one_hot_letter = torch.tensor(output).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            zerol.scatter_(2,one_hot_letter,1)
            zerol = zerol.to(device)
            if i == 0:
                output, h = model.forward(zerol)
                
                output = int(torch.argmax(output, dim=2).detach().cpu())
                
                
                letters.append(output)
                #hiddenstates[i] = h
            else:
                output, h = model.forward(zerol, h)
                
                output = int(torch.argmax(output, dim=2).detach().cpu())
                
                letters.append(output)
                #hiddenstates[i % 30] = h
    print('Final generation: ', dataset.convert_to_string(letters))
    line = ' '.join(('Accuracy:',str(acc_list),'Loss', str(loss_list)))
    with open('PrideAndPrejudice.txt', 'a') as file:
                          file.write(line + '\n')
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},'model.pth' )
                          
                          
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
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    for i in [0.005]:
        train(config, i)
#    train(config)

#for i in [0.1,0.01,0.001,0.0001,0.00001]:
#     train(config, i)
 


