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

This code performs following function on batches of features and labels
    Initiliazes a Tensorflow RNN model based on configuration file
    Identifies input and output dimension based on augmented features (needs manual adjustment)
    Performs the Dual Softmax calculation suggested by (Yoav, 2016)
    Calculates the training and validation loss for each mini-batch and returns values to training script
    ** Note ** the model architecture uses melody-co-efficient of 0.5 to give equal weight to melody and 
                harmony loss    

The code creates a config file with model settings for later use within composer scripts 


**note** preprocessing, training and generation scripts are separate and must be run 
individually 


"""
import tensorflow as tf    
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn, seq2seq
import numpy as np

import preprocess

class Model(object):
    """
    inSeq: a [ T x B x D ] matrix, where T is the time steps in the batch, B is the
               batch size, and D is the amount of dimensions. 
    """
# initialize model parameters from config file    
    def __init__(self, config, training=False):
        self.config = config
        self.miniBatchLen = miniBatchLen = config.time_batch_len
        self.inputDim = inputDim = config.input_dim
        hidSize = config.hidden_size
        numLayers = config.num_layers
        dropOut = config.dropout_prob
        inDropOut = 1
        rnnCell = config.cell_type

        #create a placeholder for input sequence
        self.inSeq = tf.placeholder(tf.float32, shape=[self.miniBatchLen, None, inputDim])
        
        # reduce features from the output (change this to 0 = no augmentation, 1 = counter only
        # and 5 for counter + metrical augmentation)
        outDim= self.inputDim 
 
        # setup variables
        with tf.variable_scope("RNN"):
            outWeight = tf.get_variable("outWeight", [hidSize, outDim])
            outBias = tf.get_variable("outBias", [outDim])
            self.lr = tf.constant(config.learning_rate, name="learning_rate")
            self.lrDecay = tf.constant(config.learning_rate_decay, name="learning_rate_decay")
            
        # create RNN cell type based on definition in config file 
        # set as LSTM by default and can be changed in train.py to GRU or Simple RNN
        def create_cell(input_size):
            if rnnCell == "Vanilla":
                cell_class = rnn_cell.BasicRNNCell
            elif rnnCell == "GRU":
                cell_class = rnn_cell.GRUCell
            elif rnnCell == "LSTM":
                cell_class = rnn_cell.BasicLSTMCell
            else:
                raise Exception("Invalid cell type: {}".format(rnnCell))
            cell = cell_class(hidSize, input_size = input_size)

            #apply output dropout to training data  
            if training:
                return rnn_cell.DropoutWrapper(cell, output_keep_prob = dropOut)
            else:
                return cell

        #create input sequence applying dropout to training data 
        # input drop out has been set to 1 i.e. no dropout for this experiment
        if training:
            self.inDropOut = tf.nn.dropout(self.inSeq, keep_prob = inDropOut)
        else:
            self.inDropOut = self.inSeq

        # create an n layer (num_layer) sized MultiRnnCell, defining sizes for each
        self.cell = rnn_cell.MultiRNNCell([create_cell(inputDim)] + [create_cell(hidSize) for i in range(1, numLayers)])

        # batch size = number of timesteps i.e. 128 , initial 0 state and input+dropout tensor
        batchSize = tf.shape(self.inDropOut)[0]
        self.initial_state = self.cell.zero_state(batchSize, tf.float32)
        inputs = tf.unpack(self.inDropOut)

        # rnn outputs a list of [batchSize x H] outputs
        outputs, self.final_state = rnn.rnn(self.cell, inputs, initial_state=self.initial_state)
        
        # get the outputs, calculate output activations
        outputs = tf.pack(outputs)
        opConcat = tf.reshape(outputs, [-1, hidSize])
        logitConcat = tf.matmul(opConcat, outWeight) + outBias

        #Reshape output tensor
        logits = tf.reshape(logitConcat, [self.miniBatchLen, -1, outDim]) 

        # probabilities of each note
        self.probs = self.calculate_probs(logits)
        self.loss = self.init_loss(logits, logitConcat)
        self.trainStep = tf.train.RMSPropOptimizer(self.lr, decay = self.lrDecay).minimize(self.loss)
    
    # loss calculation and note probabilities without dual softmax 
    def init_loss(self, outputs, _):
        self.targetSeq = tf.placeholder(tf.float32, [self.miniBatchLen, None, self.inputDim])

        batchSize = tf.shape(self.inSeq_dropout)
        crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(outputs, self.targetSeq)
        return tf.reduce_sum(crossEntropy) / self.miniBatchLen / tf.to_float(batchSize)

    def calculate_probs(self, logits):
        return tf.sigmoid(logits)
    
    # initiliaze the model with cell states
    def get_cell_zero_state(self, session, batchSize):
        return self.cell.zero_state(batchSize, tf.float32).eval(session=session)

class NottinghamModel(Model):
    """ 
    Dual softmax formulation as described by (Yoav, 2016)
    Applied to dataset with structure augmented features allowing melodies and 
    harmonies to learn from augmentation.
    
    """
    
    #initiliaze with model 
    def init_loss(self, outputs, opConcat):
        self.targetSeq = tf.placeholder(tf.int64, [self.miniBatchLen, None, 2])
        batchSize = tf.shape(self.targetSeq)[1]

        with tf.variable_scope("RNN"):
            self.melody_coeff = tf.constant(self.config.melody_coeff)

        targets_concat = tf.reshape(self.targetSeq, [-1, 2])
        
        #Dual softmax calculates individual melody and harmony losses
        melLoss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            opConcat[:, :preprocess.melodyRange], \
            targets_concat[:, 0])
        harLoss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            opConcat[:, preprocess.melodyRange:], \
            targets_concat[:, 1])
        losses = tf.add(self.melody_coeff * melLoss, (1 - self.melody_coeff) * harLoss)
        return tf.reduce_sum(losses) / self.miniBatchLen / tf.to_float(batchSize)
    
    # apply softmax to melody and harmony output and append the probabiliy values
    def calculate_probs(self, logits):
        steps = []
        for timeStep in range(self.miniBatchLen):
            softmaxMelody = tf.nn.softmax(logits[timeStep, :, :preprocess.melodyRange])
            softmaxHarmony = tf.nn.softmax(logits[timeStep, :, preprocess.melodyRange:])
            steps.append(tf.concat(1, [softmaxMelody, softmaxHarmony]))
        return tf.pack(steps)

    def assign_melody_coeff(self, session, melody_coeff):
        if melody_coeff < 0.0 or melody_coeff > 1.0:
            raise Exception("Invalid melody coeffecient")

        session.run(tf.assign(self.melody_coeff, melody_coeff))

