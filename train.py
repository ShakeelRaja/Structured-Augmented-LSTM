#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Shakeel ur Rehman Raja
Final Project for MSc Data Science (2016/17). City, University of london  

StructLSTM - (Structure Augmented LSTM model)

The code is inspired by the dual softmax model developed by Yoav Zimmerman, 
at http://yoavz.com/music_rnn/. The baseline model has been developed by applying 
a few midifcations to the original code available at 
https://github.com/yoavz/music_rnn

This code performs following function on datasets created in preprocess.py
    Loads a dataset indiciated by preprocess.py 
    Performs zero-padding on the dataset for preserving sequence duration 
    Performs mini-batching on data and creates features and labels dataset
    Removes augmented features from labels and makes it compatible with Language Modelling
    Uses examples and labels for training and validation of the model
    Runs a parametric grid search with defined ranges 
    saves the best models and training output automatically during training

Setting the zero-pad flag in preprocess.py allows the code 
to add a padding to make it a multiple of 128 (mini-batch length).

The code creates a config file with model settings for later use within composer scripts 


**note** preprocessing, training and generation scripts are separate and must be run 
individually 

"""

import os, sys
import argparse
import time
import itertools
import cPickle
import logging
import random
import string
import preprocess

from collections import defaultdict
from random import shuffle


import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

from model import Model, NottinghamModel

def configName(config):
    
    # print configuration settings on console
    def replace_dot(s): return s.replace(".", "p")
    return " num_layers_" + str(config. num_layers) + \
           "_hidden_size_" + str(config.hidden_size) + \
            replace_dot("_melCoef_{}".format(config.melody_coeff)) + \
            replace_dot("_dropout_prob_{}".format(config.dropout_prob)) + \
            replace_dot("_inpdropout_prob_{}".format(config.input_dropout_prob)) + \
            replace_dot("_timeBatchLen_{}".format(config.time_batch_len))+ \
            replace_dot("_cellType_{}".format(config.cell_type))

# This class defines the defailt parameters for training the model, settings in 
# grid search will overwrite these values.             
class DefaultConfig(object):
    
    # default model achitecture
    num_layers = 2
    hidden_size = 200
    melody_coeff = 0.5
    dropout_prob = 0.5
    input_dropout_prob = 1       # set to 1 for no dropout_prob at input layer
    cell_type = 'LSTM'    # use "GRU" for GRUs and "Vanilla" for simple RNN

    # default learning parameters
    max_time_batches= 10
    time_batch_len= 128
    learning_rate = 1e-3
    learning_rate_decay = 0.9
    num_epochs = 250

    # metadata
    dataset = 'softmax'
    modelFile = ''
    
    
    def __repr__(self):
        return """Num Layers: {}, Hidden Size: {}, dropout_prob Prob: {}, Cell Type: {}, Time Batch Len: {}, Learning Rate: {}, Decay: {}""".format(self. num_layers, self.hidden_size,  self.dropout_prob, self.cell_type, self.time_batch_len, self.learning_rate, self.learning_rate_decay)
data = []
targets = []    

# Runs training and validation with model and resturns loss values to main program 
def run_epoch(session, model, batches, training=False, testing=False):

    # shuffle batches
    shuffle(batches)

    # set target tensors for testing and training
    target_tensors = [model.loss, model.final_state]
    if testing:
        target_tensors.append(model.probs)
        batch_probs = defaultdict(list)
    if training:
        target_tensors.append(model.trainStep)

    losses = []
    for data, targets in batches:
        # save state over unrolling time steps
        batch_size = data[0].shape[1]
        nTimeSteps = len(data)
        state = model.get_cell_zero_state(session, batch_size) 
        probs = list()

        for tb_data, tb_targets in zip(data, targets):
            if testing:
                tbd = tb_data
                tbt = tb_targets
            else:
                # shuffle all the batches of input, state, and target *FOR TRAINING ONLY*
                batches = tb_data.shape[1]
                permutations = np.random.permutation(batches)
                tbd = np.zeros_like(tb_data)
                tbd[:, np.arange(batches), :] = tb_data[:, permutations, :]
                tbt = np.zeros_like(tb_targets)
                tbt[:, np.arange(batches), :] = tb_targets[:, permutations, :]
                state[np.arange(batches)] = state[permutations]
            
            # prepare input features and labels for model training
            feed_dict = {
                model.initial_state: state,
                model.inSeq: tbd,
                model.targetSeq: tbt,
            }
            results = session.run(target_tensors, feed_dict=feed_dict)
            
            #save losses and state for training next mini-batch
            losses.append(results[0])
            state = results[1]
            if testing:
                batch_probs[nTimeSteps].append(results[2])
                
    loss = sum(losses) / len(losses)
    
    # return training and validation losses
    if testing:
        return [loss, batch_probs]
    else:
        return loss


def Batch_Data(seqIn, nTimeSteps):

    seq = [s[:(nTimeSteps*time_batch_len)+1, :] for s in seqIn]

    # stack sequences depth wise (along third axis).
    stacked = np.dstack(seq)
    # swap axes so that shape is (SEQ_LENGTH X BATCH_SIZE X inputDim)
    data = np.swapaxes(stacked, 1, 2)
    # roll data -1 along lenth of sequence for next sequence prediction
    targets = np.roll(data, -1, axis=0)
    # cutoff final time step, cut of count from targets
    data = data[:-1, :, :]
    targets = targets[:-1, :, :] #-1 in 3rd dimension to eliminate counter from targets
    
#    assert data.shape == targets.shape #works without counter 

    labels = np.ones((targets.shape[0], targets.shape[1], 2), dtype=np.int32)
    #create melody and harmony labels
    labels[:, :, 0] = np.argmax(targets[:, :, :preprocess.melodyRange], axis=2)
    labels[:, :, 1] = np.argmax(targets[:, :, preprocess.melodyRange:], axis=2)
    targets = labels
    
    # ensure data and target integrity 
    assert targets.shape[:2] == data.shape[:2]
    assert data.shape[0] == nTimeSteps * time_batch_len

    # split sequences into time batches
    tb_data = np.split(data, nTimeSteps, axis=0)
    tb_targets = np.split(targets, nTimeSteps, axis=0)

    return (tb_data, tb_targets)

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

"""
Main program code
"""
    
np.random.seed(0) 
softmax = True     

loc = preprocess.pickleLoc     #location of dataset created in preprocessing
counter = preprocess.counter    # counter flag 
modelLoc = 'models'
runName = time.strftime("%m%d_%H%M")
time_step = 120
modelClass = NottinghamModel

# read data saved in preprocess.py
with open(loc, 'r') as f:
    pickle = cPickle.load(f)
    chordIx = pickle['chordIx']

inputDim = pickle["train"][0].shape[1]  #+1 for counter
print 'Data loaded, with total number of input dimension = {}'.format(inputDim)

# set up run dir for saving training data 
runLoc = os.path.join(modelLoc, runName)
if os.path.exists(runLoc):
    raise Exception("{} already exists, select a different folder", format(runLoc))
os.makedirs(runLoc)

#start logger for training and validation runs
logger = logging.getLogger(__name__) 
logger.handlers = []
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(os.path.join(runLoc, "training.log")))

#setup a grid for trying combinations of hyperparameters
# ranges can be defined as [value1,value2...valuen]
paramGrid = {
    "dropout_prob": [0.8],
    "melody_coeff": [0.5],
    " num_layers": [2],
    "hidden_size": [150],
    "num_epochs": [50],
    "learning_rate": [5e-3],
    "learning_rate_decay": [0.9],
}

# Generate a combination of hyper parameters defined in the grid
runs = list(list(itertools.izip(paramGrid, x)) for x in itertools.product(*paramGrid.itervalues()))
logger.info("a total of {} runs will be performed".format(len(runs)))

# intitliaze training variables
bestConfig = None
bestValidLoss = None
time_batch_len =128
comb = 1

#Run the combinations identified from grid parameters
for combination in runs:
    #load grid values to config
    config = DefaultConfig()
    config.dataset = 'softmax'
    #create model with random name
    config.model_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(12)) + '.model'
    for attr, value in combination:
        setattr(config, attr, value)

    dataSplit = {}
    for dataset in ['train', 'valid']:
    # For testing, use ALL the sequences
    # for counter, use same length of testing and training to allow retention of augmented features
        #if counter == False:         
        if dataset == 'valid':
            max_time_batches= -1
        else:
            max_time_batches= 10
            
        # load data 
        sequences = pickle[dataset]
        metadata = pickle[dataset + '_meta']
        dims = sequences[0].shape[1]
        seqLens = [s.shape[0] for s in sequences]

        avgSeqLen = sum(seqLens) / len(sequences)

        # print basic information about the dataset 
        if comb == 1:
            print "Dataset: {}".format(dataset)
            print "Ave. Sequence Length: {}".format(avgSeqLen)
            print "Max Sequence Length: {}".format(time_batch_len)
            print "Number of sequences: {}".format(len(sequences))
            print "____________________________________"    

        batches = defaultdict(list)
#
       # for zero padding, comment out for truncating sequences
       # creates padding for input data to make it a multiple of mini-batch length
        for sequence in sequences:
            if (sequence.shape[0]-1) % time_batch_len== 0  :
                nTimeSteps = ((sequence.shape[0]-1) // time_batch_len) 
            else:
                #calculate the pad size and create new sequence
                nTimeSteps = ((sequence.shape[0]-1) // time_batch_len) + 1
                pad = np.zeros((nTimeSteps*time_batch_len+1, sequence.shape[1]))
                pad[:sequence.shape[0],:sequence.shape[1]] = sequence
                sequence = pad

            if nTimeSteps < 1:
                continue
            if max_time_batches > 0 and nTimeSteps > max_time_batches:
                continue
            
#        #for truncating sequences, comment out for zero padding and feature augmentation 
#        for sequence in sequences:
#            # -1 because we can't predict the first step
#            nTimeSteps = ((sequence.shape[0]-1) // miniBatchLen) 
#            if nTimeSteps < 1:
#                continue
#            if max_time_batches> 0 and nTimeSteps > max_time_batches:
#                continue

            batches[nTimeSteps].append(sequence)

            # create batches of examples based on sequence length/minibatches
            batchedData =  [Batch_Data(bSeq, tStep) for tStep, bSeq in batches.iteritems()]
            
            #add metadata to batched data (just in case)
            dataSplit[dataset] = {
            "data": batchedData,
            "metadata": metadata,
            }
            dataSplit["inputDim"] = batchedData[0][0][0].shape[2]
    
    # save number of input dimensions to configuration file            
    config.input_dim =dataSplit["inputDim"]
    
    print ''
    print'Combination no. {}'.format(comb)
    print ''
    logger.info(config)
    configPath = os.path.join(runLoc, configName(config) + 'RUN_'+str(comb)+'_'+ '.config')
    with open(configPath, 'w') as f: 
        cPickle.dump(config, f)
#
#%%
    # build tensorflow models for training and validation from model.py
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            train_model = modelClass(config, training=True)
        with tf.variable_scope("model", reuse=True):
            valid_model = modelClass(config, training=False)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=40)
        tf.initialize_all_variables().run()
        
        # training variables initialization
        earlyStopBestLoss = None
        modelSave = False
        saveFlag = False
        tLosses, vLosses = [], []
        start_time = time.time()
        
        # train and validate the model calculating loss at each iteration
        for i in range(config.num_epochs):
            loss = run_epoch(session, train_model, 
               dataSplit["train"]["data"], training=True, testing=False)
            tLosses.append((i, loss))
            if i == 0:
                continue
            # perform validation on trained model
            vLoss= run_epoch(session, valid_model,dataSplit["valid"]["data"], training=False, testing=False)
            vLosses.append((i, vLoss))
            logger.info('Epoch: {}, Train Loss: {}, Valid Loss: {}, Time Per Epoch: {}'.format(\
                    i, loss, vLoss, (time.time() - start_time)/i))
            
            # save current model if new validation loss goes higher or lower than current best validation loss
            if earlyStopBestLoss == None:
                earlyStopBestLoss = vLoss
            elif vLoss < earlyStopBestLoss:
                earlyStopBestLoss = vLoss
                if modelSave:
                    logger.info('Best model seen. model saved')
                    saver.save(session, os.path.join(runLoc, config.model_name))
                    saveFlag = True
            elif not modelSave:
                modelSave = True 
                logger.info('Valid loss increased, previous model was saved')
                saver.save(session, os.path.join(runLoc, config.model_name))
                saveFlag = True
                
        #save model if not saved already
        if not saveFlag:
            saver.save(session, os.path.join(runLoc, config.model_name))
#
        #plot train and validation loss curves
        axes = plt.gca()
        axes.set_ylim([0, 3])
            
        # Save the loss values as a png file in the model folder
        plt.plot([t[0] for t in tLosses], [t[1] for t in tLosses])
        plt.plot([t[0] for t in vLosses], [t[1] for t in vLosses])
        plt.legend(['Train Loss', 'Validation Loss'])
        chart_file_path = os.path.join(runLoc, configName(config) +'_RUN_'+str(comb)+'_'+'.png')
        plt.savefig(chart_file_path)
        plt.clf()

        #log the best model and config   
        logger.info("Config {}, Loss: {}".format(config, earlyStopBestLoss))
        if bestValidLoss == None or earlyStopBestLoss < bestValidLoss:
            logger.info("Found best new model!")
            bestValidLoss = earlyStopBestLoss
            bestConfig = config
        
        comb = comb+1
    
    # identify best loss when training ends
    logger.info("Best Config: {}, Loss: {}".format(bestConfig, bestValidLoss))
